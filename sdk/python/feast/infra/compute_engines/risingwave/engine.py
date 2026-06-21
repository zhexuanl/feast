"""RisingWave compute + online-serving engine for Feast (contrib).

Architecture: RisingWave owns real-time computation and online serving via
continuous materialized views; Feast keeps the registry and the point-in-time
training joins. One ``update()`` provisions BOTH the online MV and the offline
Iceberg sink from one feature definition, so online and offline are computed by the
same engine — minimal train/serve skew.

Status: SCAFFOLD. The contract wiring, config, registry-free provisioning, and PIT
delegation are grounded and verified against the Feast/RisingWave source. The SQL
generation and the windowed-agg -> Iceberg composition are NOT yet verified
end-to-end; treat them as unproven and validate against live RisingWave before
relying on them in production (see the inline ``UNVERIFIED`` markers).
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
    passthrough_history_source_name,
    source_name,
    tiles_name,
)
from feast.infra.compute_engines.risingwave.iceberg_source import (
    IcebergSource,
    is_passthrough_stream,
    is_passthrough_view,
    is_streaming_tile,
    is_tile_fv,
    is_tile_view,
    tile_interval,
    view_aggregations,
)
from feast.infra.compute_engines.risingwave.nodes import (
    build_batch_tile_select,
    build_latest_row_select,
    build_online_rollup_select,
    build_streaming_tile_select,
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

# Minimal Feast dtype -> RisingWave SQL type. Not yet complete: extend to the full Feast
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

# RisingWave CREATE-clause type -> the canonical name RisingWave reports in information_schema.columns
# (verified live on v3.0.0). Used to compare a deployed source's column types against the desired schema,
# since the catalog reports canonical names ("double precision", "character varying") rather than the
# CREATE-clause form ("DOUBLE PRECISION", "VARCHAR").
_RW_CANONICAL_TYPE = {
    "BIGINT": "bigint",
    "INT": "integer",
    "DOUBLE PRECISION": "double precision",
    "REAL": "real",
    "VARCHAR": "character varying",
    "BOOLEAN": "boolean",
    "BYTEA": "bytea",
    "TIMESTAMP": "timestamp without time zone",
}


def _canonical_type(dtype) -> str:
    # The canonical information_schema type for a Feast dtype, defaulting to VARCHAR's canonical form for an
    # unmapped dtype (matching _passthrough_source_ddl's VARCHAR fallback).
    return _RW_CANONICAL_TYPE.get(_RW_TYPE.get(str(dtype), "VARCHAR"), "character varying")


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


def _passthrough_column_info(view) -> ColumnInfo:
    # Column info for a passthrough view's latest-row MV: feature_cols are the RAW feature columns (carried
    # through unchanged, not resolved aggregation names). The source is the Kafka stream for a streaming
    # passthrough, the Iceberg batch source otherwise. created_ts is intentionally NOT used: the offline
    # as-of read is over an Iceberg history (a streaming passthrough's batch_source, or a batch
    # passthrough's own source) and IcebergSource carries no created_timestamp_column, so to keep online ==
    # offline both sides order by event time alone (latest-value-by-timestamp, the established feature stores attribute model).
    source = view.stream_source if is_passthrough_stream(view) else view.batch_source
    return ColumnInfo(
        join_keys=[f.name for f in view.entity_columns],
        feature_cols=[f.name for f in view.features],
        ts_col=source.timestamp_field,
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
    # the Kafka _source_ddl) it needs NO column list.
    opts = (
        ["connector='iceberg'"]
        + _iceberg_storage_opts(config)
        + [f"table.name='{_sql_str(table)}'"]
    )
    return f'CREATE SOURCE IF NOT EXISTS "{name}" WITH ({", ".join(opts)})'


def _source_is_retractable(source) -> bool:
    # Append-only (CREATE SOURCE ... FORMAT PLAIN) by default. Retractable upstreams
    # (CREATE TABLE ... FORMAT UPSERT) are not yet supported; when added, return True so the
    # monoid guard in build_windowed_agg_select engages.
    return False


def _kafka_source_with(source: KafkaSource) -> str:
    # The Kafka connector WITH clause + encoding shared by every Kafka CREATE SOURCE (aggregation and
    # passthrough). Only JSON encoding is supported for now; non-JSON formats are not yet implemented.
    return (
        "WITH (connector='kafka', "
        f"properties.bootstrap.server='{_sql_str(source.kafka_options.kafka_bootstrap_servers)}', "
        f"topic='{_sql_str(source.kafka_options.topic)}', scan.startup.mode='earliest') "
        "FORMAT PLAIN ENCODE JSON"
    )


def _source_ddl(name: str, source: KafkaSource, view) -> str:
    # Placeholder typing: raw aggregation-input columns are not in view.features, and
    # their types are not carried on the FeatureView, so we emit placeholder types.
    # Real types must instead be sourced from the stream/batch source schema.
    cols: List[str] = []
    seen = set()
    for field in view.entity_columns:
        cols.append(f'"{field.name}" {_RW_TYPE.get(str(getattr(field, "dtype", "")), "VARCHAR")}')
        seen.add(field.name)
    for agg in view.aggregations:
        if agg.column and agg.column not in seen:
            cols.append(f'"{agg.column}" DOUBLE PRECISION')  # placeholder type pending real source-schema types
            seen.add(agg.column)
    ts = source.timestamp_field
    cols.append(f'"{ts}" TIMESTAMP')

    watermark = ""
    if source.kafka_options.watermark_delay_threshold is not None:
        secs = int(source.kafka_options.watermark_delay_threshold.total_seconds())
        # subtract the watermark delay from the event timestamp to bound out-of-order lateness
        watermark = f', WATERMARK FOR "{ts}" AS "{ts}" - INTERVAL \'{secs}\' SECOND'

    return (
        f'CREATE SOURCE IF NOT EXISTS "{name}" ({", ".join(cols)}{watermark}) '
        + _kafka_source_with(source)
    )


def _passthrough_source_ddl(name: str, source: KafkaSource, view) -> str:
    # The Kafka CREATE SOURCE for a passthrough (non-aggregated) stream view: the column list is the entity
    # keys + the RAW feature columns (typed from the declared schema, not a placeholder — a passthrough
    # column IS a source column) + the event timestamp. No watermark: the latest-row MV is a Group-TopN over
    # an append-only source, which needs no window/watermark. No created-timestamp column: a passthrough
    # orders by event time alone (see _passthrough_column_info).
    cols: List[str] = []
    seen = set()
    for field in view.entity_columns:
        cols.append(f'"{field.name}" {_RW_TYPE.get(str(getattr(field, "dtype", "")), "VARCHAR")}')
        seen.add(field.name)
    for feature in view.features:
        if feature.name not in seen:
            cols.append(f'"{feature.name}" {_RW_TYPE.get(str(getattr(feature, "dtype", "")), "VARCHAR")}')
            seen.add(feature.name)
    if source.timestamp_field not in seen:
        cols.append(f'"{source.timestamp_field}" TIMESTAMP')
    return (
        f'CREATE SOURCE IF NOT EXISTS "{name}" ({", ".join(cols)}) '
        + _kafka_source_with(source)
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
    # only complete at t+w, so an as-of (<=) join can't read it early.
    select = f'SELECT {projection}, "window_end" AS event_timestamp FROM "{mv}"'

    opts = (
        ["connector='iceberg'", "create_table_if_not_exists='true'"]
        + _iceberg_storage_opts(config)
        + [f"table.name='{_sql_str(name)}'"]
    )

    if upsert:
        # Composite PK so each (entity, window) bucket is a DISTINCT retained row.
        # NEVER entity-only: that collapses to one row per entity and leaks the latest
        # value to every training label.
        opts.append("type='upsert'")
        opts.append(f"primary_key='{', '.join(column_info.join_keys)}, window_end'")
    else:
        # Append-only retains the full timestamped history the PIT join needs.
        opts.append("type='append-only'")
        opts.append("force_append_only='true'")

    return f'CREATE SINK IF NOT EXISTS "{name}" AS {select} WITH ({", ".join(opts)})'


def _drop_ddl(project: str, view) -> List[str]:
    return [
        f'DROP SINK IF EXISTS "{offline_sink_name(project, view.name)}"',
        f'DROP MATERIALIZED VIEW IF EXISTS "{online_mv_name(project, view.name)}"',
        f'DROP SOURCE IF EXISTS "{source_name(project, view.name)}"',
    ]


def _passthrough_drop_ddl(project: str, view) -> List[str]:
    # Teardown for a passthrough view, dependents first: the online MV name is shared with the plain-stream
    # shape, which also provisions a "{base}_offline" Iceberg sink reading that MV. So a view re-applied
    # from an aggregating stream to a passthrough would leave that sink behind — and since the sink depends
    # on the MV, the un-CASCADEd MV drop would fail. Drop the sink first (a no-op for a passthrough view
    # that never had one), then the latest-row MV, then the source.
    return [
        f'DROP SINK IF EXISTS "{offline_sink_name(project, view.name)}"',
        f'DROP MATERIALIZED VIEW IF EXISTS "{online_mv_name(project, view.name)}"',
        f'DROP SOURCE IF EXISTS "{source_name(project, view.name)}"',
        # The offline-history Iceberg source exists only for a streaming passthrough; IF EXISTS makes this
        # a no-op for a batch passthrough (whose online source IS the history) or an online-only stream.
        f'DROP SOURCE IF EXISTS "{passthrough_history_source_name(project, view.name)}"',
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
    """The SELECT of a deployed materialized view as RisingWave stores it (verbatim: RisingWave
    persists ``CREATE MATERIALIZED VIEW <name> AS <select>`` with our SELECT unchanged),
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
    definitions cannot reveal). Single quotes in the table are doubled in the DDL; we un-double them.

    Note: unlike a materialized view (whose SELECT RisingWave stores VERBATIM), a source's WITH clause is
    RE-RENDERED in the catalog (spaces around ``=``, expanded types), so we EXTRACT the
    ``table.name`` option with a spacing-tolerant regex rather than comparing the whole definition."""
    cur.execute("SELECT definition FROM rw_catalog.rw_sources WHERE name = %s", (name,))
    row = cur.fetchone()
    if not row:
        return None
    m = re.search(r"(?:^|[\s,(])table\.name\s*=\s*'((?:[^']|'')*)'", row[0])
    return m.group(1).replace("''", "'") if m else None


def _deployed_kafka_source_opts(cur, name: str) -> Optional[tuple]:
    """The Kafka connector options of a deployed source as RisingWave stores them — the tuple
    ``(topic, bootstrap_servers, watermark_secs)`` — or None if the source does not exist. A stream (or
    streaming-tile) view reads its source by the (stable) source NAME, so a repointed topic, a moved
    bootstrap server, or a changed watermark delay live ONLY in the ``CREATE SOURCE`` definition, never in
    any materialized-view SELECT — reading them back from the catalog is the only way to detect that a kept
    view's source changed (a repointed topic would keep feeding stale data; a changed watermark would
    silently shift late-event admission and break train/serve parity).

    Like ``_deployed_source_table``, a source's WITH clause is RE-RENDERED in the catalog (spaces around
    ``=``, expanded types) rather than stored verbatim, so each option is EXTRACTED with a spacing-tolerant
    regex; single quotes doubled in the DDL are un-doubled. The watermark delay is parsed from the column
    list's ``WATERMARK FOR <ts> AS <ts> - INTERVAL '<n>' SECOND`` (the only INTERVAL a Kafka source DDL
    carries); a source with no watermark yields None for that slot."""
    cur.execute("SELECT definition FROM rw_catalog.rw_sources WHERE name = %s", (name,))
    row = cur.fetchone()
    if not row:
        return None
    definition = row[0]
    topic = re.search(r"(?:^|[\s,(])topic\s*=\s*'((?:[^']|'')*)'", definition)
    bootstrap = re.search(
        r"(?:^|[\s,(])properties\.bootstrap\.server\s*=\s*'((?:[^']|'')*)'", definition
    )
    watermark = re.search(r"INTERVAL\s+'(\d+)'\s+SECOND", definition)
    return (
        topic.group(1).replace("''", "'") if topic else None,
        bootstrap.group(1).replace("''", "'") if bootstrap else None,
        int(watermark.group(1)) if watermark else None,
    )


def _desired_kafka_source_opts(source: KafkaSource) -> tuple:
    """The desired ``(topic, bootstrap_servers, watermark_secs)`` from a view's KafkaSource, in the same
    shape ``_deployed_kafka_source_opts`` reads back so the two compare directly. ``watermark_secs`` is the
    integer-second watermark delay (matching ``_source_ddl``'s ``INTERVAL '<n>' SECOND``), or None when the
    source sets no watermark."""
    opts = source.kafka_options
    wm = opts.watermark_delay_threshold
    return (
        opts.topic,
        opts.kafka_bootstrap_servers,
        int(wm.total_seconds()) if wm is not None else None,
    )


def _deployed_source_columns(cur, name: str) -> Optional[dict]:
    """The ``{column: canonical SQL type}`` map of a deployed source as RisingWave reports it in
    information_schema, or None if the source does not exist. A passthrough Kafka source declares its raw
    feature columns with explicit types; a feature dtype change (same column name) shows in NO MV SELECT (the
    latest-row MV projects columns by name only), so it is read back here to detect that the source schema
    changed. information_schema reports canonical type names, so it is compared against ``_canonical_type``
    rather than the re-rendered CREATE SOURCE form."""
    cur.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s", (name,)
    )
    rows = cur.fetchall()
    return {name: dtype for name, dtype in rows} if rows else None


def _desired_passthrough_columns(view) -> dict:
    """The desired ``{column: canonical SQL type}`` a passthrough Kafka source declares — entity keys + raw
    feature columns + the event timestamp — in the same canonical form ``_deployed_source_columns`` reads
    back, so a feature dtype change is detected on reconcile."""
    source = view.stream_source
    cols: dict = {}
    for field in view.entity_columns:
        cols[field.name] = _canonical_type(getattr(field, "dtype", ""))
    for feature in view.features:
        cols.setdefault(feature.name, _canonical_type(getattr(feature, "dtype", "")))
    cols.setdefault(source.timestamp_field, "timestamp without time zone")
    return cols


def _norm_sql(sql):
    """Whitespace-normalize a SELECT for definition comparison (RW stores our SELECT verbatim modulo
    whitespace). None stays None so a missing deployed object never compares equal to a desired one."""
    return None if sql is None else " ".join(sql.split())


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

    norm = _norm_sql
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
            cur.execute("set sink_decouple = false")  # required before creating Iceberg sinks
            for view in views_to_delete:
                # is_streaming_tile FIRST: a streaming-tile view IS a StreamFeatureView, but its physical
                # objects are the tile graph (N online MVs + tiles MV + source, no Iceberg sink), so it
                # tears down via _batch_drop_ddl, not _drop_ddl.
                if is_streaming_tile(view):
                    for stmt in _batch_drop_ddl(project, view):
                        cur.execute(stmt)
                elif is_passthrough_view(view):
                    for stmt in _passthrough_drop_ddl(project, view):
                        cur.execute(stmt)
                elif isinstance(view, StreamFeatureView):
                    for stmt in _drop_ddl(project, view):
                        cur.execute(stmt)
                elif is_tile_fv(view):
                    for stmt in _batch_drop_ddl(project, view):
                        cur.execute(stmt)
            for view in views_to_keep:
                # Precedence is load-bearing: a streaming-tile AND a streaming-passthrough view are both
                # StreamFeatureViews, so is_streaming_tile and is_passthrough_view must be checked BEFORE the
                # generic StreamFeatureView branch — else a passthrough (no-aggregation) view would hit the
                # windowed-agg path, which rejects an empty aggregation set. is_passthrough_view also precedes
                # is_tile_fv, but the two are mutually exclusive (a tile view has aggregations).
                if is_streaming_tile(view):
                    self._reconcile_streaming_tile_view(cur, project, view)
                elif is_passthrough_view(view):
                    self._reconcile_passthrough_view(cur, project, view)
                elif isinstance(view, StreamFeatureView):
                    self._reconcile_stream_view(cur, project, view)
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
                # is_streaming_tile FIRST (it is also a StreamFeatureView): tear down the tile graph.
                if is_streaming_tile(view):
                    for stmt in _batch_drop_ddl(project, view):
                        cur.execute(stmt)
                elif is_passthrough_view(view):
                    for stmt in _passthrough_drop_ddl(project, view):
                        cur.execute(stmt)
                elif isinstance(view, StreamFeatureView):
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
                "watermark_delay_threshold. Set one, or "
                "disable emit_on_window_close (losing online/offline consistency)."
            )

        column_info = _registry_free_column_info(view)
        src = source_name(project, view.name)
        mv = online_mv_name(project, view.name)
        return [
            _source_ddl(src, source, view),
            _materialized_view_ddl(mv, self._stream_mv_select(project, view)),
            _iceberg_sink_ddl(offline_sink_name(project, view.name), mv, column_info, self.config),
        ]

    def _stream_mv_select(self, project: str, view) -> str:
        """The EOWC windowed-aggregation SELECT for a stream view's online MV — the ONE definition shared
        by provisioning and reconcile so the two cannot drift."""
        return build_windowed_agg_select(
            _registry_free_column_info(view),
            list(view.aggregations),
            source_name(project, view.name),
            source_is_retractable=_source_is_retractable(view.stream_source),
            emit_on_close=bool(self.config.emit_on_window_close),
        )

    def _reconcile_stream_view(self, cur, project: str, view) -> None:
        """Reconcile a KEPT StreamFeatureView to its current definition (re-materialize on a change;
        no-op when unchanged) — the stream analogue of ``_reconcile_batch_view``. ``CREATE ... IF NOT
        EXISTS`` would keep the old EOWC MV, so a changed aggregation would silently serve/train under
        the OLD definition. Compare the MV's deployed SELECT (RW catalog, stored verbatim) against the
        desired one; on a change drop the graph (sink -> MV -> source, dependents first) and re-provision.

        A repointed topic/bootstrap or a changed watermark delay lives only in the ``CREATE SOURCE``
        definition (the EOWC MV reads the source by its stable name), not in any MV SELECT, so it is read
        back from the source catalog (``_deployed_kafka_source_opts``) and triggers the same drop+reprovision
        — else a repointed topic keeps feeding stale data, or a changed watermark silently shifts late-event
        admission and breaks train/serve parity. The drop+reprovision already drops and recreates the source.

        Both online serving AND offline training read this EOWC MV (the offline source is a PostgreSQL
        query over it), so re-materializing the MV keeps online == offline under the new
        definition — no skew.

        Iceberg-sink follow-up (deferred): the ``{base}_offline`` Iceberg sink is a durable
        copy training does NOT read today — so its table retaining pre-change rows after a re-materialize
        is harmless now. RisingWave has no sink-level table reset (only ``create_table_if_not_exists``;
        no drop/overwrite/truncate), so if the offline read ever migrates to the Iceberg sink (i.e. once
        MV retention is introduced), the re-materialize must purge that table out-of-band (catalog
        drop, or a definition-versioned table name) to keep the durable archive consistent."""
        mv = online_mv_name(project, view.name)
        src = source_name(project, view.name)
        desired = self._stream_mv_select(project, view)
        mv_changed = _norm_sql(_deployed_mv_select(cur, mv)) != _norm_sql(desired)
        # A repointed topic/bootstrap or a changed watermark shows in NO MV SELECT (the EOWC MV reads the
        # source by its stable name), so read the source opts back from the catalog and reprovision on a
        # difference too. ``deployed is None`` (unprovisioned) is handled by mv_changed (the MV is absent).
        deployed_opts = _deployed_kafka_source_opts(cur, src)
        source_changed = (
            deployed_opts is not None
            and deployed_opts != _desired_kafka_source_opts(view.stream_source)
        )
        if mv_changed or source_changed:
            for stmt in _drop_ddl(project, view):
                cur.execute(stmt)
            for stmt in self._provision_ddl(project, view):
                cur.execute(stmt)

    def _passthrough_mv_select(self, project: str, view) -> str:
        """The latest-row SELECT for a passthrough view's online MV — the ONE definition shared by
        provisioning and reconcile so the two cannot drift."""
        return build_latest_row_select(
            _passthrough_column_info(view), source_name(project, view.name)
        )

    def _provision_passthrough_ddl(self, project: str, view) -> List[str]:
        """Registry-free DDL for one passthrough (non-aggregated) feature view: a source + ONE latest-row
        online MV (the newest row per entity, served by the same point-lookup as an aggregation MV). A
        streaming passthrough reads a Kafka source that declares the raw feature columns; a batch passthrough
        reads an Iceberg source that infers them. No window/tile and no watermark: the latest-row MV is a
        Group-TopN over an append-only source. No Iceberg sink.

        Offline training reads the RAW history with an as-of cut, not this MV. A batch passthrough's own
        Iceberg source IS that history. A streaming passthrough's Kafka stream is not queryable as history,
        so when its stream source declares an Iceberg batch_source (the historical log backing the stream)
        we ALSO provision an Iceberg source over it for the offline read; absent one, the view is
        online-only and offline retrieval is refused."""
        src = source_name(project, view.name)
        mv = online_mv_name(project, view.name)
        if is_passthrough_stream(view):
            source = view.stream_source
            if isinstance(source, PushSource):
                raise ValueError(
                    f"passthrough StreamFeatureView '{view.name}' uses a PushSource, which is too thin to "
                    "compile to a RisingWave CREATE SOURCE. Use a KafkaSource."
                )
            if not isinstance(source, KafkaSource):
                raise ValueError(
                    "RisingWaveComputeEngine requires a KafkaSource-backed passthrough stream view; "
                    f"'{view.name}' has {type(source).__name__}."
                )
            ddl = [
                _passthrough_source_ddl(src, source, view),
                _materialized_view_ddl(mv, self._passthrough_mv_select(project, view)),
            ]
            history = getattr(source, "batch_source", None)
            if isinstance(history, IcebergSource):
                ddl.append(
                    _iceberg_source_ddl(
                        passthrough_history_source_name(project, view.name), history.table, self.config
                    )
                )
            return ddl
        source_ddl = _iceberg_source_ddl(src, view.batch_source.table, self.config)
        return [source_ddl, _materialized_view_ddl(mv, self._passthrough_mv_select(project, view))]

    def _reconcile_passthrough_view(self, cur, project: str, view) -> None:
        """Reconcile a KEPT passthrough view to its current definition (re-materialize on a change; no-op
        when unchanged) — the passthrough analogue of ``_reconcile_stream_view``. ``CREATE ... IF NOT
        EXISTS`` would keep the old latest-row MV, so a changed feature set would silently serve the OLD
        definition; compare the MV's deployed SELECT (RW catalog, stored verbatim) against the desired one
        and, on a change, drop the MV + source and re-provision.

        A repointed source (a Kafka topic/bootstrap/watermark change, or an Iceberg-table repoint) lives only
        in the CREATE SOURCE definition, not in the MV SELECT, so it is read back from the source catalog and
        triggers the same drop+reprovision — else the MV keeps reading the stale source."""
        mv = online_mv_name(project, view.name)
        src = source_name(project, view.name)
        desired = self._passthrough_mv_select(project, view)
        mv_changed = _norm_sql(_deployed_mv_select(cur, mv)) != _norm_sql(desired)
        if is_passthrough_stream(view):
            # (topic, bootstrap): a passthrough source carries NO watermark (the latest-row MV is a
            # Group-TopN, not an EOWC window), so the watermark slot is not part of its identity.
            deployed_opts = _deployed_kafka_source_opts(cur, src)
            opts_changed = (
                deployed_opts is not None
                and deployed_opts[:2] != _desired_kafka_source_opts(view.stream_source)[:2]
            )
            # Column schema: a passthrough Kafka source declares the raw feature columns with explicit types,
            # so a feature dtype change (same name) must rebuild the source — it shows in no MV SELECT.
            deployed_cols = _deployed_source_columns(cur, src)
            schema_changed = deployed_cols is not None and any(
                deployed_cols.get(name) != dtype
                for name, dtype in _desired_passthrough_columns(view).items()
            )
            # The offline-history Iceberg source must exist and point at the stream's batch_source table:
            # rebuild if it is MISSING (an Iceberg batch_source added to a previously online-only view) or
            # REPOINTED (a different table) — neither shows in the MV SELECT or the Kafka opts.
            history = getattr(view.stream_source, "batch_source", None)
            history_changed = False
            if isinstance(history, IcebergSource):
                deployed_history = _deployed_source_table(
                    cur, passthrough_history_source_name(project, view.name)
                )
                history_changed = deployed_history is None or deployed_history != history.table
            source_changed = opts_changed or schema_changed or history_changed
        else:
            # A batch passthrough reads an Iceberg source that INFERS its schema, so a feature dtype change
            # needs no source rebuild; only a table repoint (read from the source catalog) does.
            deployed_table = _deployed_source_table(cur, src)
            source_changed = deployed_table is not None and deployed_table != view.batch_source.table
        if mv_changed or source_changed:
            for stmt in _passthrough_drop_ddl(project, view):
                cur.execute(stmt)
            for stmt in self._provision_passthrough_ddl(project, view):
                cur.execute(stmt)

    def _provision_batch_ddl(self, project: str, view) -> List[str]:
        """Registry-free DDL for one tile-aggregating BatchFeatureView: an Iceberg source, a continuous
        TILES MV (``build_batch_tile_select`` — per-(entity, tile_end) partials over window-independent
        partials), and ONE ONLINE ROLLUP MV PER DISTINCT WINDOW (``build_online_rollup_select`` — the
        request-anchored ``now()`` window rollup, one row per entity). One tile set is reused across all
        windows; each window gets its own now()-anchored MV because RisingWave rejects now()
        inside a CASE in a two-sided temporal-filter MV. RisingWave maintains every MV incrementally as
        the Iceberg table grows (validated end-to-end on RisingWave v3.0.0), so there is no scheduler. The point-lookup reads the
        per-window MV (``online_window_mv_name``) holding the requested feature's window — the tiles MV
        is internal plumbing. NO Iceberg sink yet: the MVs are read directly, and durable tile history
        is not provisioned. Offline training rolls the SAME tiles up per label timestamp."""
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

    def _provision_streaming_tile_ddl(self, project: str, view) -> List[str]:
        """Registry-free DDL for one STREAMING tile feature view: a watermarked Kafka source, a
        continuous EOWC TILES MV (``build_streaming_tile_select`` — per-(entity, tile_end)
        window-INDEPENDENT partials tumbled at the aggregation_interval), and ONE now()-anchored online
        rollup MV PER DISTINCT WINDOW (``build_online_rollup_select``). Same topology as the batch tile
        path (one tiles MV + N rollup MVs, NO Iceberg sink — the offline tile PIT reads the tiles MV
        directly); only the tile SOURCE differs: a watermarked Kafka source tumbled EMIT ON WINDOW CLOSE
        vs an Iceberg ``date_trunc`` GROUP BY. The rollup MVs and the offline PIT are byte-identical to
        the batch path, so serving/training reuse the same code.

        EOWC tiles require a watermark on the source timestamp: a late event is dropped once at its tile
        boundary, which is exactly what keeps online == offline (both read the same EOWC tiles).
        This is intrinsic to the tile model, so — unlike a plain stream MV, whose EOWC is opt-in via
        ``emit_on_window_close`` — the watermark is ALWAYS required here; reject a source that sets none
        (the EOWC tiles would never emit)."""
        source = view.stream_source
        if isinstance(source, PushSource):
            raise ValueError(
                f"streaming-tile view '{view.name}' uses a PushSource, which is too thin to compile to "
                "a RisingWave CREATE SOURCE (data_source.py:851-882). Use a KafkaSource."
            )
        if not isinstance(source, KafkaSource):
            raise ValueError(
                "RisingWaveComputeEngine requires a KafkaSource-backed streaming-tile view; "
                f"'{view.name}' has {type(source).__name__}."
            )
        if source.kafka_options.watermark_delay_threshold is None:
            raise ValueError(
                "a streaming-tile view's EOWC tiles require a watermark on the source timestamp (a late "
                "event is dropped once at its tile boundary, keeping online == offline), but the "
                f"KafkaSource for '{view.name}' sets no watermark_delay_threshold. Set one."
            )
        # enable_tiling without a tiling_hop_size: is_streaming_tile is True (it keys on enable_tiling +
        # aggregations) but tile_interval is None — fail loud with the fix rather than an opaque
        # 'NoneType' AttributeError deep in the tile SQL builder. (StreamFeatureView leaves
        # tiling_hop_size None when unset — its 5-min default is only a local for window validation.)
        if view.tiling_hop_size is None:
            raise ValueError(
                f"streaming-tile view '{view.name}' has enable_tiling=True but no tiling_hop_size (the "
                "tile interval). Set tiling_hop_size to 1 hour or 1 day (the streaming tile grid)."
            )

        interval = tile_interval(view)
        column_info = _registry_free_column_info(view)
        aggs = view_aggregations(view)
        src = source_name(project, view.name)
        tiles = tiles_name(project, view.name)
        ddl = [
            _source_ddl(src, source, view),
            _materialized_view_ddl(
                tiles,
                build_streaming_tile_select(column_info, aggs, src, aggregation_interval=interval),
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

    def _reconcile_streaming_tile_view(self, cur, project: str, view) -> None:
        """Reconcile a KEPT streaming-tile view to its CURRENT definition — the streaming analogue of
        ``_reconcile_batch_view`` (and the tile analogue of ``_reconcile_stream_view``). Feast routes a
        same-name edited view to views_to_keep without saying what changed, and ``CREATE ... IF NOT
        EXISTS`` would keep the old MVs, so serving/training would diverge from the applied definition.
        RisingWave has no CREATE OR REPLACE, so compare each object's desired definition against the one
        RW stores verbatim in its catalog and drop+recreate only what differs — the SAME pure planner
        (``_plan_batch_reconcile``) the batch tile path uses, since the tile graph is identical (one tiles
        MV + N per-window online MVs); only the tiles MV's source/SELECT differs.

        EOWC tiles MV changed (different per-tile partials) -> full re-materialize (drop the online MVs,
        the tiles MV, re-provision). Tiles MV unchanged (window-independent partials, so adding/removing a
        window does not touch it) -> reconcile only the per-window online MVs.

        A repointed topic/bootstrap or a changed watermark delay does not show in any MV definition (the
        tiles MV reads the source by its stable name), so it is read back from the source catalog
        (``_deployed_kafka_source_opts``) and, on a difference, forces a full re-materialize (which already
        drops+recreates the source) — else the tiles keep reading the stale topic, or admit late events
        under the old watermark, diverging online from offline."""
        interval = tile_interval(view)
        column_info = _registry_free_column_info(view)
        aggs = view_aggregations(view)
        src = source_name(project, view.name)
        tiles = tiles_name(project, view.name)
        desired_tiles = build_streaming_tile_select(
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
        # A repointed topic/bootstrap or a changed watermark lives only in the CREATE SOURCE definition (the
        # tiles MV reads the source by its stable name), so detect it from the source catalog and force a
        # full re-materialize — which already drops+recreates the source below — else the tiles keep reading
        # the stale topic, or admit late events under the old watermark, diverging online from offline.
        deployed_opts = _deployed_kafka_source_opts(cur, src)
        source_changed = (
            deployed_opts is not None
            and deployed_opts != _desired_kafka_source_opts(view.stream_source)
        )
        if source_changed:
            full_rebuild, drops, creates = True, list(deployed_online), []
        for name in drops:  # online MVs (dependents) first
            cur.execute(f'DROP MATERIALIZED VIEW IF EXISTS "{name}"')
        if full_rebuild:
            cur.execute(f'DROP MATERIALIZED VIEW IF EXISTS "{tiles}"')
            # Drop the source too (unlike the batch reconcile): a streaming source declares its agg-input
            # columns EXPLICITLY (_source_ddl), so a partials change that adds an aggregation over a NEW
            # input column also changes the source schema — and CREATE SOURCE IF NOT EXISTS would keep the
            # old columns, so the rebuilt tiles MV would reference a column the source lacks. Dropping it
            # (the source is dependency-free once the tiles MV is gone) lets CREATE SOURCE pick up the new
            # columns. The batch reconcile needs no analogue: an Iceberg source infers its schema, so a new
            # column is read without re-creating the source.
            cur.execute(f'DROP SOURCE IF EXISTS "{src}"')
            for stmt in self._provision_streaming_tile_ddl(project, view):
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
        if is_tile_view(view):
            # A tile view's offline training reads the live tiles MV directly (the tile PIT rollup), and that
            # MV already holds the full history its source carries — there is no durable offline table to
            # backfill. The plain windowed-agg backfill below would instead compute non-tile values (skewing
            # offline from the tile rollup served online) and target a staging table a tile view never
            # provisions, so a tile view materializes offline to a no-op.
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
            # UNVERIFIED end-to-end: the windowed-agg -> offline-table backfill is not yet
            # proven in-repo. Preferred long-term: read the live sink's Iceberg history so
            # backfill == what was served. The bounded [start, end) predicate must be
            # applied here before execution.
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
