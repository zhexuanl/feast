"""RisingWave compute + online-serving engine for Feast (contrib).

Architecture: RisingWave owns real-time computation and online serving via
continuous materialized views; Feast keeps the registry and the point-in-time
training joins. One ``update()`` provisions BOTH the online MV and the offline
Iceberg sink from one feature definition, so online and offline are computed by the
same engine — minimal train/serve skew.

Status: SCAFFOLD. The contract wiring, config, registry-free provisioning, and PIT
delegation are grounded and verified against the Feast/RisingWave source. The SQL
generation and the windowed-agg -> Iceberg composition are NOT yet verified
end-to-end and are gated behind the de-risking spike (see ``README.md`` and the
inline ``UNVERIFIED`` markers).
"""

import logging
from datetime import timedelta
from typing import List, Literal, Optional, Sequence, Union

from feast import (
    BatchFeatureView,
    Entity,
    FeatureView,
    OnDemandFeatureView,
    StreamFeatureView,
)
from feast.data_source import KafkaSource, PushSource
from feast.infra.common.materialization_job import (
    MaterializationJob,
    MaterializationJobStatus,
    MaterializationTask,
)
from feast.infra.compute_engines.base import ComputeEngine
from feast.infra.compute_engines.risingwave.feature_builder import (
    RisingWaveFeatureBuilder,
)
from feast.infra.compute_engines.risingwave.job import (
    RisingWaveMaterializationJob,
)
from feast.infra.compute_engines.risingwave.names import (
    offline_sink_name,
    online_mv_name,
    source_name,
    tiles_name,
)
from feast.infra.compute_engines.risingwave.nodes import (
    build_batch_tile_select,
    build_online_rollup_select,
    build_windowed_agg_select,
)
from feast.infra.compute_engines.dag.context import ColumnInfo
from feast.infra.offline_stores.offline_store import OfflineStore
from feast.infra.online_stores.online_store import OnlineStore
from feast.infra.registry.base_registry import BaseRegistry
from feast.repo_config import FeastConfigBaseModel, RepoConfig

logger = logging.getLogger(__name__)

_ENGINE_PATH = (
    "feast.infra.compute_engines.risingwave.engine.RisingWaveComputeEngine"
)

# Minimal Feast dtype -> RisingWave SQL type. Spike-gated: extend to the full Feast
# type system, and source raw-input column types from the source schema, before
# production. Raw aggregation-input columns default to DOUBLE PRECISION.
_RW_TYPE = {
    "Int64": "BIGINT",
    "Int32": "INT",
    "Float64": "DOUBLE PRECISION",
    "Float32": "REAL",
    "String": "VARCHAR",
    "Bool": "BOOLEAN",
    "Bytes": "BYTEA",
    "UnixTimestamp": "TIMESTAMP",
}


class RisingWaveComputeEngineConfig(FeastConfigBaseModel):
    """Config for the RisingWave compute engine. Set as ``batch_engine`` (and/or the
    per-view ``stream_engine``) in ``feature_store.yaml``."""

    type: Literal[_ENGINE_PATH] = _ENGINE_PATH
    """Full module path to the engine class (no core repo_config.py registration)."""

    host: str = "localhost"
    port: int = 4566  # standalone.rs:331
    database: str = "dev"  # standalone.rs:332
    user: Optional[str] = "root"  # standalone.rs:333
    password: Optional[str] = None

    # Offline Iceberg sink — the well-governed history Feast point-in-time-joins over.
    catalog_name: str = "feast"
    catalog_type: str = "storage"
    warehouse_path: Optional[str] = None
    iceberg_database: str = "feast"
    s3_endpoint: Optional[str] = None
    s3_region: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None

    # Pin EMIT ON WINDOW CLOSE so the online MV and the offline history agree
    # (consistency over freshness). Requires a watermark on the stream source.
    emit_on_window_close: bool = True


def _connect(config):
    import psycopg  # feast's postgres stores use psycopg v3 (postgres.py:19)

    return psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.database,
        user=config.user,
        password=config.password,
        autocommit=True,
    )


def _registry_free_column_info(view) -> ColumnInfo:
    # update() receives no registry and the engine holds none (base.py:40-56), so
    # derive join keys from entity_columns (registry-free; cf local/nodes.py:377-378)
    # rather than registry.get_entity(). feature_cols are the *output* (resolved)
    # names — used only by the offline sink projection.
    return ColumnInfo(
        join_keys=[f.name for f in view.entity_columns],
        feature_cols=[f.name for f in view.features],
        ts_col=view.stream_source.timestamp_field,
        created_ts_col=None,
        field_mapping=None,
    )


def _batch_column_info(view) -> ColumnInfo:
    # Batch analog of _registry_free_column_info: the timestamp is on the BATCH source
    # (view.source), not a stream_source. feature_cols are the resolved output names —
    # carried onto the tile partials by build_batch_tile_select.
    return ColumnInfo(
        join_keys=[f.name for f in view.entity_columns],
        feature_cols=[f.name for f in view.features],
        ts_col=view.source.timestamp_field,
        created_ts_col=None,
        field_mapping=None,
    )


def _aggregation_interval(view) -> timedelta:
    # Feast's Aggregation carries no aggregation_interval (the tile size) — that is a
    # ourfs view-level concept — so it rides on a tag the authoring surface sets.
    secs = view.tags.get("ourfs_aggregation_interval")
    if secs is None:
        raise ValueError(
            f"BatchFeatureView '{view.name}' has no 'ourfs_aggregation_interval' tag: the "
            "tile model needs a tile size, but Feast's Aggregation carries none. Set it via "
            "the ourfs BatchFeatureView authoring surface."
        )
    try:
        return timedelta(seconds=int(secs))
    except (ValueError, TypeError):
        raise ValueError(
            f"BatchFeatureView '{view.name}' tag ourfs_aggregation_interval={secs!r} must be an "
            "integer number of seconds (e.g. '3600' for hourly, '86400' for daily)."
        )


def _sql_str(value: str) -> str:
    # Escape a value going into a single-quoted SQL string literal / connector option. Mirrors
    # Feast's snowflake.py _escape_snowflake_sql_string and RW's own option-quoting rules.
    return str(value).replace("'", "''")


def _iceberg_storage_opts(config) -> List[str]:
    # The catalog + S3 connection options shared by the Iceberg SOURCE (batch read) and SINK
    # (offline history). One source of truth so online and offline never read different storage —
    # all values escaped against the single-quoted option literals.
    #
    # CREDENTIALS: the S3 keys are appended ONLY when set in the engine config. For PRODUCTION S3,
    # leave them unset and run the RisingWave compute node under an IAM role / instance profile /
    # env credentials — the Iceberg connector then uses the ambient AWS credential chain, so no
    # credential is ever embedded in the (catalog-persisted, log-visible) CREATE SOURCE/SINK DDL.
    # Explicit keys are for dev/MinIO only; they are escaped but DO appear in the DDL. (RisingWave's
    # CREATE SECRET store would also hide them, but it is a licensed feature — free tier <=4 cores.)
    opts = [
        f"catalog.name='{_sql_str(config.catalog_name)}'",
        f"catalog.type='{_sql_str(config.catalog_type)}'",
        f"warehouse.path='{_sql_str(config.warehouse_path)}'",
        f"database.name='{_sql_str(config.iceberg_database)}'",
    ]
    for key, val in (
        ("s3.endpoint", config.s3_endpoint),
        ("s3.region", config.s3_region),
        ("s3.access.key", config.s3_access_key),
        ("s3.secret.key", config.s3_secret_key),
    ):
        if val:
            opts.append(f"{key}='{_sql_str(val)}'")
    return opts


def _iceberg_source_ddl(name: str, table: str, config) -> str:
    # A RisingWave Iceberg source infers its schema from the Iceberg metadata, so (unlike
    # the Kafka _source_ddl) it needs NO column list. Validated live (spike stage 5c).
    opts = (
        ["connector='iceberg'"]
        + _iceberg_storage_opts(config)
        + [f"table.name='{_sql_str(table)}'"]
    )
    return f'CREATE SOURCE IF NOT EXISTS "{name}" WITH ({", ".join(opts)})'


def _source_is_retractable(source) -> bool:
    # Append-only (CREATE SOURCE ... FORMAT PLAIN) by default. Retractable upstreams
    # (CREATE TABLE ... FORMAT UPSERT) are spike-gated; when added, return True so the
    # monoid guard in build_windowed_agg_select engages.
    return False


def _source_ddl(name: str, source: KafkaSource, view) -> str:
    # Spike-gated typing: raw aggregation-input columns are not in view.features, and
    # their types are not carried on the FeatureView, so we emit placeholder types.
    # The spike must source real types from the stream/batch source schema.
    cols: List[str] = []
    seen = set()
    for field in view.entity_columns:
        cols.append(f'"{field.name}" {_RW_TYPE.get(str(getattr(field, "dtype", "")), "VARCHAR")}')
        seen.add(field.name)
    for agg in view.aggregations:
        if agg.column and agg.column not in seen:
            cols.append(f'"{agg.column}" DOUBLE PRECISION')  # spike-gated type
            seen.add(agg.column)
    ts = source.timestamp_field
    cols.append(f'"{ts}" TIMESTAMP')

    watermark = ""
    if source.kafka_options.watermark_delay_threshold is not None:
        secs = int(source.kafka_options.watermark_delay_threshold.total_seconds())
        # watermark.slt:5-9
        watermark = f', WATERMARK FOR "{ts}" AS "{ts}" - INTERVAL \'{secs}\' SECOND'

    return (
        f'CREATE SOURCE IF NOT EXISTS "{name}" ({", ".join(cols)}{watermark}) '
        "WITH (connector='kafka', "
        f"properties.bootstrap.server='{_sql_str(source.kafka_options.kafka_bootstrap_servers)}', "
        f"topic='{_sql_str(source.kafka_options.topic)}', scan.startup.mode='earliest') "
        "FORMAT PLAIN ENCODE JSON"  # issue_18308.slt:14-15; non-JSON formats spike-gated
    )


def _materialized_view_ddl(name: str, select: str) -> str:
    return f'CREATE MATERIALIZED VIEW IF NOT EXISTS "{name}" AS {select}'


def _iceberg_sink_ddl(
    name: str, mv: str, column_info: ColumnInfo, config, *, upsert: bool = False
) -> str:
    keys = ", ".join(f'"{k}"' for k in column_info.join_keys)
    feats = ", ".join(f'"{c}"' for c in column_info.feature_cols)
    projection = ", ".join(p for p in (keys, feats) if p)
    # window_END is the event timestamp (NEVER window_start): a window [t, t+w) is
    # only complete at t+w, so an as-of (<=) join can't read it early
    # (time_window.slt:50-61).
    select = f'SELECT {projection}, "window_end" AS event_timestamp FROM "{mv}"'

    opts = (
        ["connector='iceberg'", "create_table_if_not_exists='true'"]
        + _iceberg_storage_opts(config)
        + [f"table.name='{_sql_str(name)}'"]
    )

    if upsert:
        # Composite PK so each (entity, window) bucket is a DISTINCT retained row.
        # NEVER entity-only: that collapses to one row per entity and leaks the latest
        # value to every training label (upsert_table.slt:14-28).
        opts.append("type='upsert'")
        opts.append(f"primary_key='{', '.join(column_info.join_keys)}, window_end'")
    else:
        # Append-only retains the full timestamped history the PIT join needs
        # (append_only_table.slt:29-44).
        opts.append("type='append-only'")
        opts.append("force_append_only='true'")

    return f'CREATE SINK IF NOT EXISTS "{name}" AS {select} WITH ({", ".join(opts)})'


def _drop_ddl(project: str, view) -> List[str]:
    return [
        f'DROP SINK IF EXISTS "{offline_sink_name(project, view.name)}"',
        f'DROP MATERIALIZED VIEW IF EXISTS "{online_mv_name(project, view.name)}"',
        f'DROP SOURCE IF EXISTS "{source_name(project, view.name)}"',
    ]


def _batch_drop_ddl(project: str, view) -> List[str]:
    # Mirror of _drop_ddl for a tile BatchFeatureView: drop the online rollup MV, then the tiles
    # MV it reads, then the source. No Iceberg sink is provisioned (the MVs are read directly).
    return [
        f'DROP MATERIALIZED VIEW IF EXISTS "{online_mv_name(project, view.name)}"',
        f'DROP MATERIALIZED VIEW IF EXISTS "{tiles_name(project, view.name)}"',
        f'DROP SOURCE IF EXISTS "{source_name(project, view.name)}"',
    ]


class RisingWaveComputeEngine(ComputeEngine):
    def __init__(
        self,
        *,
        repo_config: RepoConfig,
        offline_store: OfflineStore,
        online_store: OnlineStore,
        **kwargs,
    ):
        # Training/PIT retrieval is handled by RisingWaveOfflineStore — Feast's
        # provider routes get_historical_features to the offline store, never the
        # compute engine — so the engine needs no offline store of a particular type.
        super().__init__(
            repo_config=repo_config,
            offline_store=offline_store,
            online_store=online_store,
            **kwargs,
        )
        self.config = repo_config.batch_engine

    def update(
        self,
        project: str,
        views_to_delete: Sequence[
            Union[BatchFeatureView, StreamFeatureView, FeatureView]
        ],
        views_to_keep: Sequence[
            Union[BatchFeatureView, StreamFeatureView, FeatureView, OnDemandFeatureView]
        ],
        entities_to_delete: Sequence[Entity],
        entities_to_keep: Sequence[Entity],
    ):
        # DDL-in-update precedent: Snowflake (snowflake_engine.py:101-157). The
        # persistent infra (source + MV + Iceberg sink) is provisioned here; the MV
        # then keeps online features continuously fresh — there is no per-task
        # streaming start in materialize().
        with _connect(self.config) as conn, conn.cursor() as cur:
            cur.execute("set sink_decouple = false")  # required before Iceberg sinks (upsert_table.slt:2)
            for view in views_to_delete:
                if isinstance(view, StreamFeatureView):
                    for stmt in _drop_ddl(project, view):
                        cur.execute(stmt)
                elif isinstance(view, BatchFeatureView) and view.aggregations:
                    for stmt in _batch_drop_ddl(project, view):
                        cur.execute(stmt)
            for view in views_to_keep:
                if isinstance(view, StreamFeatureView):
                    for stmt in self._provision_ddl(project, view):
                        cur.execute(stmt)
                elif isinstance(view, BatchFeatureView) and view.aggregations:
                    # Only AGGREGATING batch FVs map to the tile model. A non-aggregating
                    # batch FV (pure transform) is a later increment; skip it here rather
                    # than mis-provision it.
                    for stmt in self._provision_batch_ddl(project, view):
                        cur.execute(stmt)

    def teardown_infra(
        self,
        project: str,
        fvs: Sequence[Union[BatchFeatureView, StreamFeatureView, FeatureView]],
        entities: Sequence[Entity],
    ):
        with _connect(self.config) as conn, conn.cursor() as cur:
            for view in fvs:
                if isinstance(view, StreamFeatureView):
                    for stmt in _drop_ddl(project, view):
                        cur.execute(stmt)
                elif isinstance(view, BatchFeatureView) and view.aggregations:
                    for stmt in _batch_drop_ddl(project, view):
                        cur.execute(stmt)

    def _provision_ddl(self, project: str, view) -> List[str]:
        """Registry-free DDL for one StreamFeatureView (source + MV + Iceberg sink)."""
        source = view.stream_source
        if isinstance(source, PushSource):
            raise ValueError(
                f"StreamFeatureView '{view.name}' uses a PushSource, which is too thin "
                "to compile to a RisingWave CREATE SOURCE (data_source.py:851-882). "
                "Use a KafkaSource, or route pushes through an external ingestion path."
            )
        if not isinstance(source, KafkaSource):
            raise ValueError(
                "RisingWaveComputeEngine requires a KafkaSource-backed "
                f"StreamFeatureView; '{view.name}' has {type(source).__name__}."
            )

        emit_on_close = bool(self.config.emit_on_window_close)
        has_watermark = source.kafka_options.watermark_delay_threshold is not None
        if emit_on_close and not has_watermark:
            raise ValueError(
                "emit_on_window_close=True requires a watermark on the source "
                f"timestamp, but the KafkaSource for '{view.name}' sets no "
                "watermark_delay_threshold (eowc_group_agg.slt:8-12). Set one, or "
                "disable emit_on_window_close (losing online/offline consistency)."
            )

        column_info = _registry_free_column_info(view)
        src = source_name(project, view.name)
        mv = online_mv_name(project, view.name)
        select = build_windowed_agg_select(
            column_info,
            list(view.aggregations),
            src,
            source_is_retractable=_source_is_retractable(source),
            emit_on_close=emit_on_close,
        )
        return [
            _source_ddl(src, source, view),
            _materialized_view_ddl(mv, select),
            _iceberg_sink_ddl(offline_sink_name(project, view.name), mv, column_info, self.config),
        ]

    def _provision_batch_ddl(self, project: str, view) -> List[str]:
        """Registry-free DDL for one tile-aggregating BatchFeatureView: an Iceberg source, a
        continuous TILES MV (``build_batch_tile_select`` — per-(entity, tile_end) partials), and an
        ONLINE ROLLUP MV over those tiles (``build_online_rollup_select`` — the request-anchored
        ``now()`` window rollup, one row per entity). RisingWave maintains both MVs incrementally as
        the Iceberg table grows (verified live), so there is no scheduler. The point-lookup reads the
        ROLLUP MV (``online_mv_name``) unchanged — the tiles MV is internal plumbing. NO Iceberg sink
        yet (the MVs are read directly; durable tile history is a later increment). Offline training
        rolls the SAME tiles up with ``build_tile_rollup_select`` anchored to each label timestamp."""
        interval = _aggregation_interval(view)
        column_info = _batch_column_info(view)
        aggs = list(view.aggregations)
        src = source_name(project, view.name)
        tiles = tiles_name(project, view.name)
        mv = online_mv_name(project, view.name)
        tile_select = build_batch_tile_select(
            column_info, aggs, src, aggregation_interval=interval
        )
        rollup_select = build_online_rollup_select(
            column_info, aggs, tiles, aggregation_interval=interval
        )
        return [
            _iceberg_source_ddl(src, view.source.table, self.config),
            _materialized_view_ddl(tiles, tile_select),
            _materialized_view_ddl(mv, rollup_select),
        ]

    def _materialize_one(
        self, registry: BaseRegistry, task: MaterializationTask, **kwargs
    ) -> MaterializationJob:
        view = task.feature_view
        job_id = f"{view.name}-{task.start_time}-{task.end_time}"
        if not getattr(view, "offline", False):
            # Online-only: the live MV serves online; nothing to backfill offline.
            return RisingWaveMaterializationJob(job_id, MaterializationJobStatus.SUCCEEDED)
        try:
            builder = RisingWaveFeatureBuilder(
                registry,
                view,
                task,
                source_is_retractable=_source_is_retractable(view.stream_source),
                emit_on_close=False,  # bounded batch backfill, not a streaming MV
            )
            sql = builder.build().to_sql(self.get_execution_context(registry, task))
            # UNVERIFIED end-to-end (spike risk #3): the windowed-agg -> offline-table
            # backfill is not yet proven in-repo. Preferred long-term: read the live
            # sink's Iceberg history so backfill == what was served (risk #8). The
            # bounded [start, end) predicate must be applied here before execution.
            with _connect(self.config) as conn, conn.cursor() as cur:
                cur.execute(sql)
            return RisingWaveMaterializationJob(job_id, MaterializationJobStatus.SUCCEEDED)
        except BaseException as e:
            return RisingWaveMaterializationJob(
                job_id, MaterializationJobStatus.ERROR, error=e
            )

    # NOTE: get_historical_features is intentionally NOT implemented here. Feast's
    # provider routes training/PIT retrieval to the OFFLINE store
    # (RisingWaveOfflineStore), never to the compute engine — implementing it on the
    # engine would be dead code. The base ComputeEngine raises NotImplementedError.
