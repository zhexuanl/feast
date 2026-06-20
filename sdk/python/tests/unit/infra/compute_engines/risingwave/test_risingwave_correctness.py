"""Adversarial unit tests for the RisingWave compute engine.

These tests do NOT cover the happy path. Each one tries to make the engine emit an
incorrect, leaky, or unsafe artifact and asserts that it refuses or produces the
safe form. They encode the 5 blockers + correctness invariants found in the design
review and pin what the de-risking spike must keep green.

They run without a live RisingWave: the SQL builders and the provisioning guards are
pure (no DB connection).
"""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from feast.aggregation import Aggregation
from feast.data_format import JsonFormat
from feast.data_source import KafkaSource, PushSource
from feast.infra.compute_engines.risingwave.engine import (
    RisingWaveComputeEngine,
    _aggregation_interval,
    _batch_drop_ddl,
    _iceberg_sink_ddl,
    _iceberg_source_ddl,
    _iceberg_storage_opts,
)
from feast.infra.compute_engines.risingwave.offline_store import (
    RisingWaveOfflineStore,
    RisingWaveOfflineStoreConfig,
    _entity_df_to_sql,
)
from feast.infra.compute_engines.risingwave.nodes import (
    RWFilterNode,
    RWJoinNode,
    build_batch_tile_select,
    build_offline_tile_pit_query,
    build_online_rollup_select,
    build_tile_rollup_select,
    build_windowed_agg_select,
)
from feast.infra.compute_engines.dag.context import ColumnInfo, ExecutionContext
from feast.infra.compute_engines.dag.model import DAGFormat
from feast.infra.compute_engines.dag.node import DAGNode
from feast.infra.compute_engines.dag.value import DAGValue
from feast.infra.compute_engines.utils import ENTITY_ROW_ID, ENTITY_TS_ALIAS
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
    EntitySelectMode,
)


def _column_info(feature_cols=("amount_sum_3600s",)):
    return ColumnInfo(
        join_keys=["user_id"],
        feature_cols=list(feature_cols),
        ts_col="event_ts",
        created_ts_col=None,
        field_mapping=None,
    )


def _agg(function, window_seconds=3600, column="amount"):
    return Aggregation(
        column=column,
        function=function,
        time_window=timedelta(seconds=window_seconds),
    )


def _kafka_source(watermark=True):
    return KafkaSource(
        name="txn_stream",
        timestamp_field="event_ts",
        message_format=JsonFormat(schema_json=""),
        kafka_bootstrap_servers="localhost:9092",
        topic="txn",
        watermark_delay_threshold=timedelta(seconds=30) if watermark else None,
    )


def _stream_view(source, aggs, offline=True):
    return SimpleNamespace(
        name="user_txn",
        stream_source=source,
        aggregations=list(aggs),
        entity_columns=[SimpleNamespace(name="user_id", dtype="String")],
        features=[SimpleNamespace(name=a.resolved_name(a.time_window)) for a in aggs],
        offline=offline,
    )


def _engine(emit_on_window_close=True):
    engine = RisingWaveComputeEngine.__new__(RisingWaveComputeEngine)
    engine.config = SimpleNamespace(
        emit_on_window_close=emit_on_window_close,
        catalog_name="feast",
        catalog_type="storage",
        warehouse_path="s3a://feast/wh",
        iceberg_database="feast",
        s3_endpoint=None,
        s3_region=None,
        s3_access_key=None,
        s3_secret_key=None,
    )
    return engine


# --- PIT history: window_end timestamping, append-only / composite-PK retention ---


def test_offline_sink_timestamps_by_window_end_not_window_start():
    sql = _iceberg_sink_ddl("p_v_offline", "p_v_online", _column_info(), _engine().config)
    assert '"window_end" AS event_timestamp' in sql
    # window_start would expose a still-open window to inclusive as-of joins.
    assert "window_start" not in sql


def test_offline_sink_defaults_to_append_only_history():
    sql = _iceberg_sink_ddl("p_v_offline", "p_v_online", _column_info(), _engine().config)
    assert "type='append-only'" in sql
    assert "force_append_only='true'" in sql


def test_offline_sink_upsert_uses_composite_pk_never_entity_only():
    sql = _iceberg_sink_ddl(
        "p_v_offline", "p_v_online", _column_info(), _engine().config, upsert=True
    )
    assert "type='upsert'" in sql
    # entity-only PK would collapse history and leak the latest value to every label.
    assert "primary_key='user_id, window_end'" in sql


# --- Retraction: monoid guard over a retractable source ---


@pytest.mark.parametrize("function", ["min", "max", "count_distinct"])
def test_monoid_aggregation_over_retractable_source_is_rejected(function):
    with pytest.raises(ValueError, match="monoid"):
        build_windowed_agg_select(
            _column_info(),
            [_agg(function)],
            "src",
            source_is_retractable=True,
            emit_on_close=False,
        )


@pytest.mark.parametrize("function", ["sum", "count", "mean"])
def test_abelian_aggregation_over_retractable_source_is_allowed(function):
    sql = build_windowed_agg_select(
        _column_info(),
        [_agg(function)],
        "src",
        source_is_retractable=True,
        emit_on_close=False,
    )
    assert "tumble(" in sql


def test_monoid_aggregation_over_append_only_source_is_allowed():
    sql = build_windowed_agg_select(
        _column_info(),
        [_agg("max")],
        "src",
        source_is_retractable=False,
        emit_on_close=False,
    )
    assert "max(amount)" in sql


# --- Window semantics ---


def test_windowed_select_groups_by_and_emits_window_end():
    sql = build_windowed_agg_select(
        _column_info(),
        [_agg("sum")],
        "src",
        source_is_retractable=False,
        emit_on_close=False,
    )
    assert "GROUP BY window_start, window_end" in sql


def test_mixed_windows_in_one_view_are_rejected():
    aggs = [_agg("sum", 3600), _agg("count", 86400)]
    with pytest.raises(ValueError, match="single"):
        build_windowed_agg_select(
            _column_info(),
            aggs,
            "src",
            source_is_retractable=False,
            emit_on_close=False,
        )


def test_emit_on_window_close_is_appended_only_when_requested():
    base = build_windowed_agg_select(
        _column_info(), [_agg("sum")], "src",
        source_is_retractable=False, emit_on_close=False,
    )
    eowc = build_windowed_agg_select(
        _column_info(), [_agg("sum")], "src",
        source_is_retractable=False, emit_on_close=True,
    )
    assert "EMIT ON WINDOW CLOSE" not in base
    assert eowc.endswith("EMIT ON WINDOW CLOSE")


# --- Aggregation breadth: allow-list + streaming-safe stddev/variance + approx_count_distinct ---


@pytest.mark.parametrize("function", ["median", "foobar", "approx_percentile", "first", "last"])
def test_unsupported_aggregation_function_is_rejected_at_apply(function):
    # Unknown / unsupported functions must fail at apply with a clear message, not reach
    # RisingWave as raw SQL and fail at parse time.
    with pytest.raises(ValueError, match="Unsupported aggregation function"):
        build_windowed_agg_select(
            _column_info(), [_agg(function)], "src",
            source_is_retractable=False, emit_on_close=False,
        )


@pytest.mark.parametrize("function", ["stddev_pop", "stddev_samp", "var_pop", "var_samp"])
def test_stddev_and_variance_emit_native_rw_sql(function):
    sql = build_windowed_agg_select(
        _column_info(), [_agg(function)], "src",
        source_is_retractable=False, emit_on_close=False,
    )
    assert f"{function}(amount) AS {function}_amount_3600s" in sql


@pytest.mark.parametrize("function", ["stddev_pop", "stddev_samp", "var_pop", "var_samp"])
def test_stddev_and_variance_are_not_monoids_over_retractable_source(function):
    # RisingWave decomposes these into streaming sum(x)/sum(x*x)/count (Abelian group), so it
    # can retract them over an upsert source — they must NOT trip the monoid guard.
    sql = build_windowed_agg_select(
        _column_info(), [_agg(function)], "src",
        source_is_retractable=True, emit_on_close=False,
    )
    assert "tumble(" in sql


def test_approx_count_distinct_emits_native_sql_and_is_monoid():
    sql = build_windowed_agg_select(
        _column_info(), [_agg("approx_count_distinct")], "src",
        source_is_retractable=False, emit_on_close=False,
    )
    assert "approx_count_distinct(amount) AS approx_count_distinct_amount_3600s" in sql
    # HLL has no inverse -> monoid -> rejected over a retractable source.
    with pytest.raises(ValueError, match="monoid"):
        build_windowed_agg_select(
            _column_info(), [_agg("approx_count_distinct")], "src",
            source_is_retractable=True, emit_on_close=False,
        )


# --- Batch tile aggregation (established feature stores tile model: partial aggregates + retrieval rollup) ---
# Validated end-to-end live on RW v3.0.0: spike/sql/05c_batch_tiles.sql.


def test_batch_tile_select_buckets_by_interval_and_stamps_tile_end():
    sql = build_batch_tile_select(
        _column_info(), [_agg("sum", 2592000)], "src", aggregation_interval=timedelta(days=1)
    )
    # 1-day tiles, stamped by tile_end (the event-time upper boundary of the tile).
    assert "date_trunc('day', event_ts) + INTERVAL '1 day' AS tile_end" in sql
    # the tile holds the PARTIAL sum under the final feature's resolved name.
    assert "sum(amount) AS sum_amount_2592000s" in sql
    assert "GROUP BY user_id, date_trunc('day', event_ts)" in sql


def test_batch_tile_count_partial_is_count():
    sql = build_batch_tile_select(
        _column_info(), [_agg("count", 2592000)], "src", aggregation_interval=timedelta(days=1)
    )
    assert "count(amount) AS count_amount_2592000s" in sql


def test_batch_tile_rollup_recombines_partials_in_request_anchored_window():
    sql = build_tile_rollup_select(
        _column_info(), [_agg("sum", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), as_of_sql="$1",
    )
    # sum partials roll up with sum, under the same feature name.
    assert "sum(sum_amount_2592000s) AS sum_amount_2592000s" in sql
    # request-anchored sliding window: end = the most-recent aggregation_interval boundary <= as_of.
    assert "tile_end > date_trunc('day', $1) - INTERVAL '2592000' SECOND" in sql
    assert "tile_end <= date_trunc('day', $1)" in sql
    assert "GROUP BY user_id" in sql


def test_batch_tile_count_rollup_combiner_is_sum():
    sql = build_tile_rollup_select(
        _column_info(), [_agg("count", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), as_of_sql="$1",
    )
    # COUNT tiles recombine by SUMMING the per-tile counts (not count()).
    assert "sum(count_amount_2592000s) AS count_amount_2592000s" in sql


def test_batch_tile_min_max_roll_up_with_min_max():
    tile = build_batch_tile_select(
        _column_info(), [_agg("max", 2592000)], "src", aggregation_interval=timedelta(days=1)
    )
    assert "max(amount) AS max_amount_2592000s" in tile
    roll = build_tile_rollup_select(
        _column_info(), [_agg("max", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), as_of_sql="$1",
    )
    assert "max(max_amount_2592000s) AS max_amount_2592000s" in roll


@pytest.mark.parametrize("function", ["mean", "stddev_pop", "count_distinct", "approx_count_distinct"])
def test_batch_tile_rejects_non_additive_aggregation(function):
    # mean/stddev (need sum+count / sum+sumsq+count partials) and count_distinct/approx (non-additive)
    # cannot roll up from simple per-tile partials yet — rejected with a clear message.
    with pytest.raises(ValueError, match="additive"):
        build_batch_tile_select(
            _column_info(), [_agg(function, 2592000)], "src", aggregation_interval=timedelta(days=1)
        )


def test_batch_tile_rejects_non_standard_interval():
    with pytest.raises(ValueError, match="aggregation_interval"):
        build_batch_tile_select(
            _column_info(), [_agg("sum", 2592000)], "src", aggregation_interval=timedelta(minutes=15)
        )


# --- online rollup MV: continuous now()-anchored window over the tiles (point-looked-up) ---


def test_online_rollup_uses_plain_now_two_sided_window():
    sql = build_online_rollup_select(
        _column_info(), [_agg("sum", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1),
    )
    # plain now() (NOT date_trunc(now()), which RW rejects in a two-sided temporal-filter MV)
    assert "tile_end > now() - INTERVAL '2592000' SECOND" in sql
    assert "tile_end <= now()" in sql
    assert "date_trunc" not in sql
    # combiner rolls per-tile sum partials up under the same feature name
    assert "sum(sum_amount_2592000s) AS sum_amount_2592000s" in sql
    # one row per entity; window_end is the PIT stamp the point-lookup ORDER BYs
    assert "max(tile_end) AS window_end" in sql
    assert "GROUP BY user_id" in sql


def test_online_rollup_count_combiner_is_sum():
    sql = build_online_rollup_select(
        _column_info(), [_agg("count", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1),
    )
    assert "sum(count_amount_2592000s) AS count_amount_2592000s" in sql


@pytest.mark.parametrize("function", ["mean", "stddev_pop", "count_distinct"])
def test_online_rollup_rejects_non_additive_aggregation(function):
    with pytest.raises(ValueError, match="additive"):
        build_online_rollup_select(
            _column_info(), [_agg(function, 2592000)], "tiles",
            aggregation_interval=timedelta(days=1),
        )


def test_online_rollup_rejects_window_not_multiple_of_interval():
    # a 1-hour window over 1-day tiles is not a whole number of tiles -> online/offline can't agree
    with pytest.raises(ValueError, match="multiple"):
        build_online_rollup_select(
            _column_info(), [_agg("sum", 3600)], "tiles",
            aggregation_interval=timedelta(days=1),
        )


def test_tile_rollup_offline_also_rejects_window_not_multiple_of_interval():
    # the same equivalence guard applies to the offline (date_trunc) rollup
    with pytest.raises(ValueError, match="multiple"):
        build_tile_rollup_select(
            _column_info(), [_agg("sum", 3600)], "tiles",
            aggregation_interval=timedelta(days=1), as_of_sql="$1",
        )


# --- offline tile PIT: floor-anchored range-agg join, per entity-row label ---


def test_offline_tile_pit_anchors_window_at_floor_of_each_label():
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="proj_v_tiles",
        column_info=_column_info(), aggregations=[_agg("sum", 259200)],
        aggregation_interval=timedelta(days=1),
    )
    assert "LEFT JOIN proj_v_tiles t" in sql  # LEFT so no-tile rows still appear (NULL feature)
    assert 't."user_id" = e."user_id"' in sql  # range-join on the entity key
    # window anchored at floor(THIS ROW's label), not a global now() and not the latest tile
    assert "t.tile_end > date_trunc('day', e.\"event_timestamp\") - INTERVAL '259200' SECOND" in sql
    assert "t.tile_end <= date_trunc('day', e.\"event_timestamp\")" in sql
    assert "sum(t.sum_amount_259200s) AS sum_amount_259200s" in sql  # combiner rolls partials up
    assert 'GROUP BY e."user_id", e."event_timestamp"' in sql  # one output row per entity-label


def test_offline_tile_pit_count_combiner_is_sum():
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[_agg("count", 259200)], aggregation_interval=timedelta(days=1),
    )
    assert "sum(t.count_amount_259200s) AS count_amount_259200s" in sql


@pytest.mark.parametrize("function", ["mean", "stddev_pop"])
def test_offline_tile_pit_rejects_non_additive(function):
    with pytest.raises(ValueError, match="additive"):
        build_offline_tile_pit_query(
            "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
            tiles_relation="t", column_info=_column_info(),
            aggregations=[_agg(function, 259200)], aggregation_interval=timedelta(days=1),
        )


def test_offline_tile_pit_does_not_apply_ttl_only_the_window_bounds():
    # For an aggregation FV the time_window IS the lookback bound (Chronon); ttl is NOT a
    # second bound. Pin it: exactly one tile_end lower bound (the window), no extra ttl filter.
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[_agg("sum", 259200)], aggregation_interval=timedelta(days=1),
    )
    assert sql.count("t.tile_end >") == 1  # only the window lower bound, no ttl lower bound
    assert "ttl" not in sql.lower()


# --- Provisioning guards ---


def test_pushsource_stream_view_is_rejected():
    # isinstance check fires before any attribute access, so __new__ is enough.
    push = PushSource.__new__(PushSource)
    view = _stream_view(push, [_agg("sum")])
    with pytest.raises(ValueError, match="PushSource"):
        _engine()._provision_ddl("proj", view)


def test_emit_on_window_close_requires_a_source_watermark():
    view = _stream_view(_kafka_source(watermark=False), [_agg("sum")])
    with pytest.raises(ValueError, match="watermark"):
        _engine(emit_on_window_close=True)._provision_ddl("proj", view)


def test_provision_emits_source_mv_and_iceberg_sink():
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])
    source_sql, mv_sql, sink_sql = _engine(emit_on_window_close=True)._provision_ddl(
        "proj", view
    )
    assert source_sql.startswith("CREATE SOURCE")
    assert "WATERMARK FOR" in source_sql
    assert mv_sql.startswith("CREATE MATERIALIZED VIEW")
    assert mv_sql.endswith("EMIT ON WINDOW CLOSE")
    assert sink_sql.startswith("CREATE SINK")
    assert "connector='iceberg'" in sink_sql
    assert '"window_end" AS event_timestamp' in sink_sql


# --- Phase 6 Inc 2: BATCH feature view provisioning (Iceberg source -> tiles MV) ---


def _iceberg_batch_source(table="txn_ice", ts="event_ts"):
    # Stands in for the Feast custom Iceberg batch source (the real DataSource is a
    # follow-up): _provision_batch_ddl only reads .table + .timestamp_field off it.
    return SimpleNamespace(name=table, table=table, timestamp_field=ts)


def _batch_view(aggs, interval_secs=86400, name="user_txn_daily"):
    return SimpleNamespace(
        name=name,
        source=_iceberg_batch_source(),
        aggregations=list(aggs),
        entity_columns=[SimpleNamespace(name="user_id", dtype="String")],
        features=[SimpleNamespace(name=a.resolved_name(a.time_window)) for a in aggs],
        tags={"ourfs_aggregation_interval": str(interval_secs)},
        offline=True,
    )


def test_provision_batch_emits_source_tiles_mv_and_online_rollup_mv():
    # daily (86400s) tiles, 3-day (259200s) window
    agg = _agg("sum", window_seconds=259200)
    view = _batch_view([agg])
    ddl = _engine()._provision_batch_ddl("proj", view)
    assert len(ddl) == 3  # source + tiles MV + online rollup MV; NO Iceberg sink (MVs read directly)
    source_sql, tiles_sql, rollup_sql = ddl
    feat = agg.resolved_name(agg.time_window)

    assert source_sql.startswith("CREATE SOURCE")
    assert "connector='iceberg'" in source_sql
    assert "table.name='txn_ice'" in source_sql
    assert "WATERMARK" not in source_sql  # a batch source has no watermark

    # tiles MV: internal _tiles name, per-(entity, tile_end) partials over the iceberg source
    assert tiles_sql.startswith("CREATE MATERIALIZED VIEW")
    assert '"proj_user_txn_daily_tiles"' in tiles_sql
    assert "date_trunc('day'" in tiles_sql
    assert "tile_end" in tiles_sql
    assert feat in tiles_sql  # the partial carries the FINAL feature's resolved_name

    # online rollup MV: the point-looked-up _online name, plain now() window over the tiles MV
    assert rollup_sql.startswith("CREATE MATERIALIZED VIEW")
    assert '"proj_user_txn_daily_online"' in rollup_sql
    assert "FROM proj_user_txn_daily_tiles" in rollup_sql  # reads FROM the tiles MV
    assert "now() - INTERVAL '259200' SECOND" in rollup_sql
    assert "tile_end <= now()" in rollup_sql
    assert "date_trunc" not in rollup_sql  # RW rejects two-sided date_trunc(now()) in an MV
    assert "max(tile_end) AS window_end" in rollup_sql  # PIT stamp for the point-lookup
    assert f"sum({feat}) AS {feat}" in rollup_sql  # combiner rolls partials up under one name


def test_provision_batch_requires_aggregation_interval_tag():
    view = _batch_view([_agg("sum")])
    view.tags = {}  # Feast Aggregation carries no interval; the tile size must come from a tag
    with pytest.raises(ValueError, match="aggregation_interval"):
        _engine()._provision_batch_ddl("proj", view)


def test_aggregation_interval_rejects_non_integer_tag():
    view = _batch_view([_agg("sum")])
    view.tags = {"ourfs_aggregation_interval": "daily"}  # must be integer seconds
    with pytest.raises(ValueError, match="integer number of seconds"):
        _aggregation_interval(view)


def test_iceberg_source_ddl_escapes_single_quotes_in_table_name():
    config = _engine().config
    sql = _iceberg_source_ddl("src", "my'table", config)
    assert "table.name='my''table'" in sql  # escaped, not a broken/injectable literal


def _config_with_s3(access="minio-access", secret="sup3r's3cret"):
    return SimpleNamespace(
        emit_on_window_close=True, catalog_name="feast", catalog_type="storage",
        warehouse_path="s3a://feast/wh", iceberg_database="feast",
        s3_endpoint="http://minio:9000", s3_region="us-east-1",
        s3_access_key=access, s3_secret_key=secret,
    )


def test_iceberg_opts_omit_s3_credentials_for_ambient_chain():
    # PRODUCTION path: no explicit keys in config -> keys omitted from DDL -> RW uses the node's
    # ambient AWS credential chain (IAM role / env). No credential in the catalog-persisted DDL.
    opts = " ".join(_iceberg_storage_opts(_engine().config))  # _engine() has s3_* = None
    assert "s3.access.key" not in opts
    assert "s3.secret.key" not in opts
    assert "catalog.name='feast'" in opts  # non-credential storage opts still present


def test_iceberg_opts_escape_explicit_dev_credentials():
    # DEV/MinIO path: explicit keys are escaped against the single-quoted option literal.
    config = _config_with_s3(secret="sup3r's3cret")
    opts = " ".join(_iceberg_storage_opts(config))
    assert "s3.secret.key='sup3r''s3cret'" in opts  # escaped, not broken/injectable


@pytest.mark.parametrize("function", ["mean", "stddev_pop"])
def test_provision_batch_rejects_non_additive_aggregation(function):
    view = _batch_view([_agg(function)])
    with pytest.raises(ValueError, match="additive"):
        _engine()._provision_batch_ddl("proj", view)


def test_batch_drop_ddl_drops_both_mvs_and_source_with_no_sink():
    view = _batch_view([_agg("sum")])
    stmts = _batch_drop_ddl("proj", view)
    # drop order: online rollup MV, then the tiles MV it reads, then the source
    assert stmts[0] == 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_online"'
    assert stmts[1] == 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_tiles"'
    assert stmts[2].startswith("DROP SOURCE")
    assert not any("SINK" in s for s in stmts)  # none provisioned, none to drop


# (Removed: engine.get_historical_features tests — retrieval is the offline store's
#  job now, exercised by the RisingWaveOfflineStore tests at the bottom of this file.)


# --- New surface helpers: drive the pure SQL-builder DAG nodes without a live DB ---


class _StubInputNode(DAGNode):
    """A single upstream node whose output DAGValue is pre-seeded into the context.

    The RisingWave nodes pull their input via ``get_single_input_value`` keyed by the
    input node's name, so we register one stub and stash its DAGValue in
    ``context.node_outputs``.
    """

    def __init__(self, name, value: DAGValue):
        super().__init__(name)
        self._value = value

    def execute(self, context):  # pragma: no cover - never executed in these tests
        return self._value


def _rw_value(relation: str, columns, *, metadata=None) -> DAGValue:
    return DAGValue(
        data=relation,
        format=DAGFormat.RISINGWAVE,
        metadata={**(metadata or {}), "columns": list(columns)},
    )


def _context_with_input(input_node: _StubInputNode, *, entity_df=None) -> ExecutionContext:
    """Minimal ExecutionContext carrying only what the nodes read.

    Built via ``__new__`` so we do not need a real RepoConfig / OfflineStore — the
    join/filter nodes only touch ``project``, ``entity_df`` and ``node_outputs``.
    """
    context = ExecutionContext.__new__(ExecutionContext)
    context.project = "proj"
    context.entity_df = entity_df
    context.node_outputs = {input_node.name: input_node._value}
    return context


# --- (b) RWFilterNode applies an INCLUSIVE PIT cut (ts <= entity ts), not strict < ---


def _filter_column_info():
    return ColumnInfo(
        join_keys=["user_id"],
        feature_cols=["amount_sum_3600s"],
        ts_col="window_end",
        created_ts_col=None,
        field_mapping=None,
    )


def test_filter_pit_cut_is_inclusive_on_window_end_not_strict_or_window_start():
    # Feature relation already aggregated: window_end is the effective event timestamp,
    # carried via metadata. The PIT cut must be `window_end <= __entity_event_timestamp`.
    columns = ["user_id", "amount_sum_3600s", "window_end", ENTITY_TS_ALIAS]
    upstream = _StubInputNode(
        "agg",
        _rw_value(
            "(SELECT ...)",
            columns,
            metadata={"event_timestamp_column": "window_end", "aggregated": True},
        ),
    )
    node = RWFilterNode(
        "filter", SimpleNamespace(), _filter_column_info(), inputs=[upstream]
    )
    out = node.execute(_context_with_input(upstream))
    sql = out.data
    assert f'"window_end" <= "{ENTITY_TS_ALIAS}"' in sql
    # Inclusive only: a strict cut would drop the row whose window closes exactly at the
    # label time. Strip the inclusive operator and assert no bare strict `<` remains.
    assert "<" not in sql.replace("<=", "")
    # window_start would admit a still-open window before it closes.
    assert "window_start" not in sql


def test_filter_pit_cut_is_skipped_when_disabled_even_with_entity_ts_present():
    # The pre-aggregation filter on the aggregated-PIT path must NOT emit a raw-ts cut
    # (that would leak partial/future-dated windows); include_pit_cut=False suppresses it.
    columns = ["user_id", "event_ts", ENTITY_TS_ALIAS]
    ci = ColumnInfo(
        join_keys=["user_id"],
        feature_cols=["amount"],
        ts_col="event_ts",
        created_ts_col=None,
        field_mapping=None,
    )
    upstream = _StubInputNode("src", _rw_value("rel", columns))
    node = RWFilterNode(
        "filter", SimpleNamespace(), ci, inputs=[upstream], include_pit_cut=False
    )
    out = node.execute(_context_with_input(upstream))
    # No predicate at all -> the node returns its input unchanged (no WHERE emitted).
    assert ENTITY_TS_ALIAS not in out.data
    assert "WHERE" not in out.data


def test_filter_ttl_lower_bound_is_inclusive_and_anchored_on_entity_ts():
    columns = ["user_id", "window_end", ENTITY_TS_ALIAS]
    upstream = _StubInputNode(
        "agg",
        _rw_value(
            "(SELECT ...)",
            columns,
            metadata={"event_timestamp_column": "window_end"},
        ),
    )
    node = RWFilterNode(
        "filter",
        SimpleNamespace(),
        _filter_column_info(),
        inputs=[upstream],
        ttl=timedelta(hours=1),
    )
    sql = node.execute(_context_with_input(upstream)).data
    assert f'"window_end" <= "{ENTITY_TS_ALIAS}"' in sql
    assert f'"window_end" >= "{ENTITY_TS_ALIAS}" - INTERVAL \'3600\' SECOND' in sql


# --- (c) RWJoinNode produces a LEFT JOIN on the join keys (entity spine on the left) ---


def test_entity_spine_join_is_left_join_on_join_keys():
    entity_df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    feature_columns = ["user_id", "amount_sum_3600s", "window_end"]
    upstream = _StubInputNode(
        "features",
        _rw_value(
            "(SELECT user_id, amount_sum_3600s, window_end FROM agg)",
            feature_columns,
            metadata={"event_timestamp_column": "window_end"},
        ),
    )
    node = RWJoinNode(
        "join", SimpleNamespace(), _filter_column_info(), inputs=[upstream]
    )
    out = node.execute(_context_with_input(upstream, entity_df=entity_df))
    sql = out.data
    # Spine is the LEFT side (alias e), features the RIGHT (alias f) -> a LEFT JOIN so
    # entity rows without a matching feature row are retained (no inner-join row loss).
    assert "LEFT JOIN" in sql
    assert "INNER JOIN" not in sql
    assert sql.count(" JOIN ") == 1
    # ON predicate is keyed on the view join key, qualified e.<key> = f.<key>.
    assert 'e."user_id" = f."user_id"' in sql
    assert out.metadata["join_type"] == "left"
    assert out.metadata["joined_on"] == ["user_id"]


def test_entity_spine_join_left_join_on_composite_join_keys():
    ci = ColumnInfo(
        join_keys=["user_id", "merchant_id"],
        feature_cols=["amount_sum_3600s"],
        ts_col="window_end",
        created_ts_col=None,
        field_mapping=None,
    )
    entity_df = pd.DataFrame(
        {
            "user_id": [1],
            "merchant_id": [9],
            "event_timestamp": pd.to_datetime(["2024-01-01"]),
        }
    )
    feature_columns = ["user_id", "merchant_id", "amount_sum_3600s", "window_end"]
    upstream = _StubInputNode("features", _rw_value("(SELECT ... FROM agg)", feature_columns))
    node = RWJoinNode("join", SimpleNamespace(), ci, inputs=[upstream])
    sql = node.execute(_context_with_input(upstream, entity_df=entity_df)).data
    assert "LEFT JOIN" in sql
    # Both keys ANDed in the ON clause; neither key silently dropped.
    assert 'e."user_id" = f."user_id"' in sql
    assert 'e."merchant_id" = f."merchant_id"' in sql
    assert " AND " in sql


def test_entity_spine_join_rejects_missing_entity_df():
    upstream = _StubInputNode(
        "features", _rw_value("(SELECT ... FROM agg)", ["user_id", "amount_sum_3600s"])
    )
    node = RWJoinNode("join", SimpleNamespace(), _filter_column_info(), inputs=[upstream])
    with pytest.raises(RuntimeError, match="requires an entity_df"):
        node.execute(_context_with_input(upstream, entity_df=None))


# --- RisingWaveOfflineStore: inline the entity_df; never upload a temp table
#     (RisingWave INSERTs are async, so an uploaded entity table is empty at query time)


def test_offline_store_config_defaults_to_embed_query_and_disabled_ssl():
    cfg = RisingWaveOfflineStoreConfig(
        host="localhost", port=4566, database="dev", user="root", password=""
    )
    assert cfg.entity_select_mode == EntitySelectMode.embed_query
    assert cfg.sslmode == "disable"


def test_entity_df_to_sql_is_a_bare_select_with_no_table_upload():
    df = pd.DataFrame(
        {
            "user_id": pd.Series(["u1"], dtype="object"),
            "event_timestamp": pd.to_datetime(["2026-06-18T12:00:00+00:00"]),
        }
    )
    sql = _entity_df_to_sql(df)
    assert sql.startswith("SELECT ")
    assert "CREATE TABLE" not in sql and "INSERT" not in sql  # never uploads
    assert "TIMESTAMPTZ" in sql  # tz-aware label is cast
    assert '"user_id"' in sql and '"event_timestamp"' in sql


def test_get_historical_features_inlines_dataframe_instead_of_uploading():
    # A DataFrame entity_df must be converted to inline SQL before delegating to the
    # parent, so the parent uses its embed_query/CTE path — never the temp-table upload
    # that RisingWave's async INSERTs leave empty.
    from unittest.mock import patch

    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "event_timestamp": pd.to_datetime(["2026-06-18T12:00:00+00:00"]),
        }
    )
    target = (
        "feast.infra.offline_stores.contrib.postgres_offline_store.postgres."
        "PostgreSQLOfflineStore.get_historical_features"
    )
    with patch(target) as parent:
        RisingWaveOfflineStore.get_historical_features(
            config=MagicMock(),
            feature_views=[],
            feature_refs=[],
            entity_df=df,
            registry=MagicMock(),
            project="proj",
        )
    assert parent.called
    passed = parent.call_args.kwargs["entity_df"]
    assert isinstance(passed, str) and passed.startswith("SELECT ")
