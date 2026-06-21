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
import re
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
    base_name,
    offline_sink_name,
    online_mv_name,
    online_window_mv_name,
    source_name,
    tiles_name,
)
from feast.infra.compute_engines.risingwave.iceberg_source import (
    is_tile_fv,
    tile_interval,
    view_aggregations,
)
from feast.infra.compute_engines.risingwave.nodes import (
    build_batch_tile_select,
    build_online_rollup_select,
    build_windowed_agg_select,
    group_aggregations_by_window,
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
    # Batch analog of _registry_free_column_info: the timestamp is on the BATCH source (the
    # IcebergSource), not a stream_source. feature_cols are the resolved output names — carried
    # onto the tile partials by build_batch_tile_select.
    return ColumnInfo(
        join_keys=[f.name for f in view.entity_columns],
        feature_cols=[f.name for f in view.features],
        ts_col=view.batch_source.timestamp_field,
        created_ts_col=None,
        field_mapping=None,
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
    # Mirror of _drop_ddl for a tile BatchFeatureView: drop the per-window online rollup MVs, then the
    # tiles MV they read, then the source. No Iceberg sink is provisioned (the MVs are read directly).
    # The window set comes from the SAME group_aggregations_by_window split the engine provisioned with.
    ddl = [
        f'DROP MATERIALIZED VIEW IF EXISTS "{online_window_mv_name(project, view.name, window_secs)}"'
        for window_secs, _ in group_aggregations_by_window(view_aggregations(view))
    ]
    ddl.append(f'DROP MATERIALIZED VIEW IF EXISTS "{tiles_name(project, view.name)}"')
    ddl.append(f'DROP SOURCE IF EXISTS "{source_name(project, view.name)}"')
    return ddl


def _existing_online_window_secs(cur, project: str, view_name: str) -> set:
    """The window-seconds of the per-window online MVs that physically EXIST for a tile view, read from
    RisingWave's pg-compatible catalog. The reconcile diffs this against the DESIRED window set so a
    re-apply that shrinks/changes a view's windows drops the now-removed windows' MVs: Feast routes a
    same-name edited view to ``views_to_keep`` (not ``views_to_delete``) and ``CREATE ... IF NOT EXISTS``
    never removes a no-longer-provisioned window's MV, so it would otherwise run forever, unreachable by
    any future provision/teardown (which only name the current window set)."""
    prefix = f"{base_name(project, view_name)}_online_"
    cur.execute("SELECT matviewname FROM pg_matviews")
    found = set()
    for (name,) in cur.fetchall():
        if name.startswith(prefix) and name.endswith("s"):
            secs = name[len(prefix) : -1]
            if secs.isdigit():
                found.add(int(secs))
    return found


def _deployed_mv_select(cur, name: str) -> Optional[str]:
    """The SELECT of a deployed materialized view as RisingWave stores it (verbatim — verified on
    RW v3.0.0: RW persists ``CREATE MATERIALIZED VIEW <name> AS <select>`` with our SELECT unchanged),
    or None if the MV does not exist. This stored SELECT is an exact definition fingerprint used to
    detect that a kept view's definition changed — the only way to do so, since RW has no CREATE OR
    REPLACE / ALTER ... AS / COMMENT ON, and Feast never tells the engine which kept views changed.

    Assumption: RW round-trips our generated SELECT unchanged (modulo whitespace, which the reconcile
    normalizes). If a future RW version re-rendered the stored definition differently from what we
    generate, the comparison would conservatively see every apply as "changed" and re-materialize each
    time — wasteful, never wrong. If that ever happens, switch to a stored definition hash (a sidecar)
    rather than comparing against RW's rendering."""
    cur.execute(
        "SELECT definition FROM rw_catalog.rw_materialized_views WHERE name = %s", (name,)
    )
    row = cur.fetchone()
    if not row:
        return None
    definition = row[0]
    marker = " AS "  # the MV name precedes the first ' AS '; everything after it is the SELECT
    idx = definition.find(marker)
    return definition[idx + len(marker) :] if idx != -1 else definition


def _deployed_source_table(cur, name: str) -> Optional[str]:
    """The Iceberg ``table.name`` of a deployed source as RisingWave stores it, or None if the source
    does not exist. A tile view's tiles MV reads its source by the (stable) source NAME, so the
    underlying Iceberg table only appears in the ``CREATE SOURCE ... table.name='...'`` definition —
    this is the only way to detect that a kept view was repointed at a different table (which the MV
    definitions cannot reveal). Single quotes in the table are doubled in the DDL; we un-double them."""
    cur.execute("SELECT definition FROM rw_catalog.rw_sources WHERE name = %s", (name,))
    row = cur.fetchone()
    if not row:
        return None
    m = re.search(r"(?:^|[\s,(])table\.name\s*=\s*'((?:[^']|'')*)'", row[0])
    return m.group(1).replace("''", "'") if m else None


def _plan_batch_reconcile(
    *, desired_tiles: str, desired_online: dict, deployed_tiles, deployed_online: dict
):
    """Pure reconcile planner: compare a tile view's DESIRED definitions against the DEPLOYED ones (read
    from RisingWave's catalog) and return ``(full_rebuild, online_drops, online_creates)``.

    ``full_rebuild`` is True when the tiles MV changed (the per-tile PARTIALS changed — a different
    aggregation function/column) or the view is unprovisioned: the caller drops every deployed online MV
    (returned in ``online_drops``) and the tiles MV, then re-provisions the whole graph. Otherwise the
    tiles MV is unchanged (its partials are WINDOW-INDEPENDENT, so adding/removing a window does NOT touch
    it) and only the per-window online MVs are reconciled — drop windows that were removed or whose rollup
    definition changed, create windows that are new or redefined, and leave unchanged windows (and their
    serving) untouched. A materialization-affecting change re-materializes; an unchanged view is a
    no-op (no rebuild, no serving blip)."""

    def norm(sql):
        return None if sql is None else " ".join(sql.split())

    if norm(deployed_tiles) != norm(desired_tiles):
        return True, list(deployed_online), []
    drops = [
        name
        for name, dep in deployed_online.items()
        if name not in desired_online or norm(dep) != norm(desired_online[name])
    ]
    creates = [
        (name, sql)
        for name, sql in desired_online.items()
        if name not in deployed_online or norm(deployed_online[name]) != norm(sql)
    ]
    return False, drops, creates


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
                elif is_tile_fv(view):
                    for stmt in _batch_drop_ddl(project, view):
                        cur.execute(stmt)
            for view in views_to_keep:
                if isinstance(view, StreamFeatureView):
                    for stmt in self._provision_ddl(project, view):
                        cur.execute(stmt)
                elif is_tile_fv(view):
                    self._reconcile_batch_view(cur, project, view)

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
                elif is_tile_fv(view):
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
        """Registry-free DDL for one tile-aggregating BatchFeatureView: an Iceberg source, a continuous
        TILES MV (``build_batch_tile_select`` — per-(entity, tile_end) partials over window-independent
        partials), and ONE ONLINE ROLLUP MV PER DISTINCT WINDOW (``build_online_rollup_select`` — the
        request-anchored ``now()`` window rollup, one row per entity). One tile set is reused across all
        windows; each window gets its own now()-anchored MV because RisingWave rejects now()
        inside a CASE in a two-sided temporal-filter MV. RisingWave maintains every MV incrementally as
        the Iceberg table grows (verified live), so there is no scheduler. The point-lookup reads the
        per-window MV (``online_window_mv_name``) holding the requested feature's window — the tiles MV
        is internal plumbing. NO Iceberg sink yet (the MVs are read directly; durable tile history is a
        later increment). Offline training rolls the SAME tiles up per label timestamp."""
        interval = tile_interval(view)
        column_info = _batch_column_info(view)
        aggs = view_aggregations(view)
        src = source_name(project, view.name)
        tiles = tiles_name(project, view.name)
        ddl = [
            _iceberg_source_ddl(src, view.batch_source.table, self.config),
            _materialized_view_ddl(
                tiles,
                build_batch_tile_select(column_info, aggs, src, aggregation_interval=interval),
            ),
        ]
        for window_secs, window_aggs in group_aggregations_by_window(aggs):
            rollup_select = build_online_rollup_select(
                column_info, window_aggs, tiles, aggregation_interval=interval
            )
            mv = online_window_mv_name(project, view.name, window_secs)
            ddl.append(_materialized_view_ddl(mv, rollup_select))
        return ddl

    def _reconcile_batch_view(self, cur, project: str, view) -> None:
        """Reconcile a KEPT tile view's physical objects to its CURRENT definition (re-materialize on a
        materialization-affecting change; a no-op when unchanged). Feast routes a same-name edited view
        to views_to_keep without telling the engine what changed, and ``CREATE ... IF NOT EXISTS`` would
        silently keep the old MVs — so serving would diverge from the applied definition. RisingWave has
        no CREATE OR REPLACE, so we compare each object's desired definition against the one RW stores
        verbatim in its catalog and drop+recreate only what differs.

        Tiles MV changed (the per-tile partials — a different aggregation function/column) -> full
        re-materialize (drop the online MVs that depend on it, drop the tiles MV, re-provision). Tiles MV
        unchanged (window-independent partials, so adding/removing a window does not touch it) -> reconcile
        only the per-window online MVs: drop removed/redefined windows, create new/redefined ones, leave
        unchanged windows (and their serving) running untouched."""
        interval = tile_interval(view)
        column_info = _batch_column_info(view)
        aggs = view_aggregations(view)
        src = source_name(project, view.name)
        tiles = tiles_name(project, view.name)
        desired_tiles = build_batch_tile_select(
            column_info, aggs, src, aggregation_interval=interval
        )
        desired_online = {
            online_window_mv_name(project, view.name, w): build_online_rollup_select(
                column_info, wa, tiles, aggregation_interval=interval
            )
            for w, wa in group_aggregations_by_window(aggs)
        }
        deployed_online = {
            online_window_mv_name(project, view.name, secs): _deployed_mv_select(
                cur, online_window_mv_name(project, view.name, secs)
            )
            for secs in _existing_online_window_secs(cur, project, view.name)
        }
        full_rebuild, drops, creates = _plan_batch_reconcile(
            desired_tiles=desired_tiles,
            desired_online=desired_online,
            deployed_tiles=_deployed_mv_select(cur, tiles),
            deployed_online=deployed_online,
        )
        # A repointed Iceberg table doesn't show in any MV definition (they read the source by its
        # stable name), so detect it from the source catalog and force a full re-materialize that also
        # drops+recreates the source — else serving would keep reading the OLD table.
        deployed_table = _deployed_source_table(cur, src)
        source_changed = deployed_table is not None and deployed_table != view.batch_source.table
        if source_changed:
            full_rebuild, drops, creates = True, list(deployed_online), []
        for name in drops:  # online MVs (dependents) first
            cur.execute(f'DROP MATERIALIZED VIEW IF EXISTS "{name}"')
        if full_rebuild:
            cur.execute(f'DROP MATERIALIZED VIEW IF EXISTS "{tiles}"')
            if source_changed:
                cur.execute(f'DROP SOURCE IF EXISTS "{src}"')  # so CREATE SOURCE picks up the new table
            for stmt in self._provision_batch_ddl(project, view):
                cur.execute(stmt)
        else:
            for name, select in creates:
                cur.execute(f'CREATE MATERIALIZED VIEW IF NOT EXISTS "{name}" AS {select}')

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
