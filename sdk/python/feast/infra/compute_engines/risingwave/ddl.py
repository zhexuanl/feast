"""RisingWave DDL-string builders + column-info helpers + the desired-online-MV planner.

Pure string/plan builders for the RisingWave engine: column-info derivation, the CREATE
SOURCE / MATERIALIZED VIEW / SINK DDL, the teardown DDL, and ``_desired_online_mvs`` (the
single home of the v2 serving split). Depends on ``engine_config`` for the dtype maps and on
the shared name + SQL-builder modules; it does NOT touch the database (the reconcile readers in
``reconcile.py`` do). ``engine.py`` re-exports every symbol here so existing imports keep
resolving.
"""

from typing import List

from feast.data_source import KafkaSource
from feast.infra.compute_engines.dag.context import ColumnInfo
from feast.infra.compute_engines.risingwave.aggregation_carriers import (
    group_aggregations_by_window_offset,
    group_lifetime_aggregations,
    is_lifetime_agg,
    is_series_agg,
    view_agg_filter_cols,
    view_agg_lifetime,
    view_agg_offsets,
    view_agg_series,
    view_secondary_key,
)
from feast.infra.compute_engines.risingwave.engine_config import _RW_TYPE
from feast.infra.compute_engines.risingwave.iceberg_source import (
    is_passthrough_stream,
    view_aggregations,
)
from feast.infra.compute_engines.risingwave.names import (
    offline_sink_name,
    online_cumulative_mv_name,
    online_lifetime_mv_name,
    online_mv_name,
    online_series_mv_name,
    online_window_mv_name,
    passthrough_history_source_name,
    source_name,
    tiles_name,
)
from feast.infra.compute_engines.risingwave.tile_plan import TilePlan


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
    # offline both sides order by event time alone (latest-value-by-timestamp).
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
    # The aggregation secondary key is a raw GROUP BY column the tiles MV references but which is neither a
    # join key nor an aggregation input, so it must be declared on the source too (placeholder type, like
    # the agg inputs above) — else the streaming tiles MV fails to bind it.
    sk = view_secondary_key(view)
    if sk and sk not in seen:
        cols.append(f'"{sk}" VARCHAR')
        seen.add(sk)
    ts = source.timestamp_field
    cols.append(f'"{ts}" TIMESTAMP')
    seen.add(ts)
    # A filtered aggregation's FILTER(WHERE ...) may reference a raw column that is neither a join key, an
    # aggregation input, nor the timestamp (e.g. a transaction_code). It is undeclared above, so the tiles
    # MV could not bind it — declare each such column with its carried source type. Empty for an unfiltered
    # view (the carrier is absent), so this leaves the unfiltered CREATE SOURCE byte-identical.
    for col, rw_type in view_agg_filter_cols(view).items():
        if col not in seen:
            cols.append(f'"{col}" {rw_type or "VARCHAR"}')
            seen.add(col)

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
    # The (window, offset) set comes from the SAME split the engine provisioned with.
    aggs = view_aggregations(view)
    lifetimes = view_agg_lifetime(view)
    series = view_agg_series(view)
    # a series aggregation has no own rollup MV (it reuses the tiles MV), so exclude it from the per-window
    # drop set, same as a lifetime aggregation.
    windowed_aggs = [a for a in aggs if not is_lifetime_agg(a, lifetimes) and not is_series_agg(a, series)]
    ddl = [
        f'DROP MATERIALIZED VIEW IF EXISTS "{online_window_mv_name(project, view.name, w, off)}"'
        for (w, off), _ in group_aggregations_by_window_offset(windowed_aggs, view_agg_offsets(view))
    ]
    ddl += [
        f'DROP MATERIALIZED VIEW IF EXISTS "{online_lifetime_mv_name(project, view.name, floor)}"'
        for floor, _ in group_lifetime_aggregations(aggs, lifetimes)
    ]
    # the cumulative MV (v2 invertible serving) reads the tiles MV, so drop it before the tiles MV; a
    # bare DROP IF EXISTS is a no-op for a view that never provisioned one (secondary-key / non-invertible).
    ddl.append(f'DROP MATERIALIZED VIEW IF EXISTS "{online_cumulative_mv_name(project, view.name)}"')
    # the series snapshot MV (step==interval window-series) reads the tiles MV, so drop it before the tiles
    # MV; a bare DROP IF EXISTS no-ops for a view that never provisioned one.
    ddl.append(f'DROP MATERIALIZED VIEW IF EXISTS "{online_series_mv_name(project, view.name)}"')
    ddl.append(f'DROP MATERIALIZED VIEW IF EXISTS "{tiles_name(project, view.name)}"')
    ddl.append(f'DROP SOURCE IF EXISTS "{source_name(project, view.name)}"')
    return ddl


def _desired_online_mvs(
    project: str,
    view_name: str,
    column_info: ColumnInfo,
    aggs: list,
    tiles: str,
    *,
    aggregation_interval,
    agg_params,
    secondary_key,
    offsets,
    lifetimes,
    series,
    filters=None,
) -> dict:
    """The desired ``{mv_name: SELECT}`` for a tile view's online rollup MVs. The v2 serving split now lives
    in ``TilePlan`` (``tile_plan.py``) — the single structured home shared by provisioning, reconcile, the
    offline read, and serving-shard derivation, so they cannot drift. This stays as the stable entry point
    that ``engine.py`` re-exports and the unit suite imports; it is a thin delegation to
    ``TilePlan.online_mvs()`` and is byte-identical to its former hand-coded body (pinned by
    ``test_tile_plan.test_online_mvs_equals_desired_online_mvs``)."""
    return TilePlan.from_inputs(
        project, view_name, column_info, aggs, tiles,
        aggregation_interval=aggregation_interval, agg_params=agg_params, secondary_key=secondary_key,
        offsets=offsets, lifetimes=lifetimes, series=series,
    ).online_mvs()
