"""Adversarial unit tests for the RisingWave compute engine.

These tests do NOT cover the happy path. Each one tries to make the engine emit an
incorrect, leaky, or unsafe artifact and asserts that it refuses or produces the
safe form. They encode the correctness invariants of the engine and
pin the behavior that must stay green.

They run without a live RisingWave: the SQL builders and the provisioning guards are
pure (no DB connection).
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from feast.aggregation import Aggregation
from feast.data_format import JsonFormat
from feast.data_source import KafkaSource, PushSource
from feast.infra.common.materialization_job import (
    MaterializationJobStatus,
    MaterializationTask,
)
from feast.infra.compute_engines.risingwave.engine import (
    RisingWaveComputeEngine,
    _batch_drop_ddl,
    _deployed_kafka_source_opts,
    _deployed_mv_select,
    _deployed_source_table,
    _desired_kafka_source_opts,
    _desired_online_mvs,
    _existing_online_mv_names,
    _iceberg_sink_ddl,
    _iceberg_source_ddl,
    _iceberg_storage_opts,
    _passthrough_drop_ddl,
    _plan_batch_reconcile,
)
from feast.infra.compute_engines.risingwave.aggregation_carriers import encode_agg_series
from feast.infra.compute_engines.risingwave.names import (
    SERIES_SNAPSHOT_ENDS_COL,
    online_series_mv_name,
)
from feast.infra.compute_engines.risingwave.iceberg_source import (
    IcebergSource,
    is_passthrough_fv,
    is_passthrough_stream,
    is_passthrough_view,
    is_tile_view,
)
from feast.infra.compute_engines.risingwave.offline_store import (
    RisingWaveOfflineStore,
    RisingWaveOfflineStoreConfig,
    _entity_df_to_sql,
)
from feast.infra.compute_engines.risingwave.nodes import (
    RWFilterNode,
    RWJoinNode,
    _assert_tile_supported,
    _partials_for,
    _recombine_expr,
    _tile_value_expr,
    build_batch_tile_select,
    build_cumulative_read_query,
    build_cumulative_tile_select,
    build_latest_row_select,
    build_lifetime_rollup_select,
    build_offline_tile_pit_query,
    build_online_rollup_select,
    build_passthrough_pit_query,
    build_series_snapshot_select,
    build_streaming_tile_select,
    build_tile_rollup_select,
    build_windowed_agg_select,
    group_lifetime_aggregations,
    snapshot_series_aggs,
    encode_agg_filter_cols,
    encode_agg_filters,
    encode_agg_lifetime,
    encode_agg_offsets,
    encode_agg_params,
    encode_secondary_key,
    group_aggregations_by_window,
    group_aggregations_by_window_offset,
    view_agg_offsets,
)
from feast.infra.compute_engines.dag.context import ColumnInfo, ExecutionContext
from feast.infra.compute_engines.dag.model import DAGFormat
from feast.infra.compute_engines.dag.node import DAGNode
from feast.infra.compute_engines.dag.value import DAGValue
from feast.infra.compute_engines.utils import ENTITY_ROW_ID, ENTITY_TS_ALIAS
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
    EntitySelectMode,
)
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
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


def _stream_tile_view(source, aggs, interval_secs=86400):
    # A streaming-tile view = a StreamFeatureView carrying Feast's NATIVE enable_tiling + tiling_hop_size
    # (the tile interval) alongside its aggregations. The engine reads tiling_hop_size as the tile size
    # (tile_interval) and is_streaming_tile keys on enable_tiling + aggregations. Windows are > interval
    # (the feature-view authoring layer guarantees this).
    view = _stream_view(source, aggs)
    view.enable_tiling = True
    view.tiling_hop_size = timedelta(seconds=interval_secs)
    return view


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


def test_latest_row_select_is_group_topn_newest_per_entity():
    # A passthrough column is a raw value, served as the newest row per entity (no aggregation, no window).
    # The latest-row MV is a Group-TopN: ROW_NUMBER() OVER (PARTITION BY keys ORDER BY ts DESC) keeping rn=1.
    sql = build_latest_row_select(
        ColumnInfo(
            join_keys=["user_id"],
            feature_cols=["amount", "country"],
            ts_col="event_ts",
            created_ts_col=None,
            field_mapping=None,
        ),
        "src",
    )
    assert "ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_ts DESC)" in sql
    assert sql.rstrip().endswith("= 1")  # keep only the newest row per entity
    assert "GROUP BY" not in sql and "tumble(" not in sql  # passthrough: no aggregation, no window
    # projects entity keys + raw feature columns + the event timestamp (servable by the point-lookup)
    assert sql.startswith("SELECT user_id, amount, country, event_ts FROM")


def test_latest_row_select_breaks_ties_on_created_timestamp_like_the_offline_read():
    # When the source defines a created timestamp, the latest-row MV must break same-event-timestamp ties by
    # created_ts DESC — the SAME order the offline as-of read uses — so online == offline on ties.
    sql = build_latest_row_select(
        ColumnInfo(
            join_keys=["user_id"],
            feature_cols=["amount"],
            ts_col="event_ts",
            created_ts_col="created_ts",
            field_mapping=None,
        ),
        "src",
    )
    assert "ORDER BY event_ts DESC, created_ts DESC" in sql


def test_latest_row_select_projects_each_column_once_when_a_feature_equals_the_timestamp():
    # A passthrough schema may name a feature the same as the timestamp (or an entity key); the projection
    # must list it once, or CREATE MATERIALIZED VIEW fails on an ambiguous duplicate output column.
    sql = build_latest_row_select(
        ColumnInfo(
            join_keys=["user_id"],
            feature_cols=["amount", "event_ts"],  # 'event_ts' coincides with the ts column
            ts_col="event_ts",
            created_ts_col=None,
            field_mapping=None,
        ),
        "src",
    )
    assert sql.startswith("SELECT user_id, amount, event_ts FROM")  # event_ts appears once, not twice
    assert sql.count("SELECT user_id, amount, event_ts,") == 1  # inner projection also deduped


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


@pytest.mark.parametrize("function", ["median", "foobar"])
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


def test_approx_percentile_emits_within_group_ordered_sql():
    # approx_percentile is parameterized (quantile + precision). Both ride out-of-band in agg_params
    # keyed by the aggregation's resolved_name (feast.Aggregation carries no param field). RisingWave's
    # form is approx_percentile(<p>, <relative_error>) WITHIN GROUP (ORDER BY <col>), where the
    # precision maps to relative_error = 1 / precision (precision 100 => 0.01, RisingWave's default).
    agg = _agg("approx_percentile")
    out = agg.resolved_name(agg.time_window)
    sql = build_windowed_agg_select(
        _column_info(), [agg], "src",
        source_is_retractable=False, emit_on_close=True,
        agg_params={out: [0.95, 100]},
    )
    assert f"approx_percentile(0.95, 0.01) WITHIN GROUP (ORDER BY amount) AS {out}" in sql
    # higher precision -> tighter relative_error
    sql2 = build_windowed_agg_select(
        _column_info(), [agg], "src",
        source_is_retractable=False, emit_on_close=True,
        agg_params={out: [0.95, 500]},
    )
    assert "approx_percentile(0.95, 0.002) WITHIN GROUP (ORDER BY amount)" in sql2


def test_approx_percentile_without_quantile_param_is_rejected():
    # approx_percentile cannot be emitted without its quantile; an aggregation that reaches the
    # builder with no agg_params entry must fail fast, not produce invalid SQL.
    with pytest.raises(ValueError, match="quantile"):
        build_windowed_agg_select(
            _column_info(), [_agg("approx_percentile")], "src",
            source_is_retractable=False, emit_on_close=True,
        )


def test_approx_percentile_is_monoid_and_tile_rejected():
    # No inverse -> monoid -> rejected over a retractable source; and no additive partial -> not
    # tile-decomposable, so it stays rejected by the tile model (plain/EOWC path only).
    with pytest.raises(ValueError, match="monoid"):
        build_windowed_agg_select(
            _column_info(), [_agg("approx_percentile")], "src",
            source_is_retractable=True, emit_on_close=True,
            agg_params={_agg("approx_percentile").resolved_name(timedelta(seconds=3600)): [0.5]},
        )
    with pytest.raises(ValueError, match="not supported"):
        _assert_tile_supported([_agg("approx_percentile")])


@pytest.mark.parametrize(
    "function,order,distinct",
    [
        ("last", "DESC", False),
        ("first", "ASC", False),
        ("last_distinct", "DESC", True),
        ("first_distinct", "ASC", True),
    ],
)
def test_sequence_aggregates_emit_ordered_array_agg_slice(function, order, distinct):
    # Sequence aggregates are Array-valued: the n most-recent (last*) / earliest (first*) values,
    # ordered. n rides the same agg_params carrier as approx_percentile's quantile. The _distinct
    # variants wrap array_distinct (RisingWave rejects array_agg(DISTINCT ... ORDER BY <other col>)).
    agg = _agg(function)
    out = agg.resolved_name(agg.time_window)
    sql = build_windowed_agg_select(
        _column_info(), [agg], "src",
        source_is_retractable=False, emit_on_close=True,
        agg_params={out: [3]},
    )
    inner = f"array_agg(amount ORDER BY event_ts {order})"
    if distinct:
        inner = f"array_distinct({inner})"
    assert f"({inner})[1:3] AS {out}" in sql


def test_sequence_aggregate_without_n_param_is_rejected():
    with pytest.raises(ValueError, match="needs an n"):
        build_windowed_agg_select(
            _column_info(), [_agg("last")], "src",
            source_is_retractable=False, emit_on_close=True,
        )


def test_sequence_aggregate_is_tile_supported():
    # Sequence aggregates now tile via a bounded top-n-per-tile partial re-merged at rollup (the
    # per-tile array is bounded to n). No raise.
    _assert_tile_supported([_agg("last")])


def test_windowed_path_rejects_duplicate_resolved_output_names():
    # Two aggregations resolving to the SAME output column collide on one AS alias (and, for a
    # parameterized agg, the resolved_name-keyed param carrier clobbers one of the params). The
    # plain/EOWC path is the ONLY path approx_percentile runs on, so the guard must live in the
    # shared builder — rejecting clearly at apply for BOTH the online MV and the offline backfill,
    # not as an opaque CREATE MATERIALIZED VIEW failure.
    agg = _agg("approx_percentile")
    out = agg.resolved_name(agg.time_window)
    with pytest.raises(ValueError, match="duplicate output column"):
        build_windowed_agg_select(
            _column_info(), [agg, _agg("approx_percentile")], "src",
            source_is_retractable=False, emit_on_close=True,
            agg_params={out: [0.5, 100]},
        )


def test_stream_mv_select_threads_approx_percentile_quantile_from_view_tags():
    # The quantile rides in the view's ourfs_agg_params tag; the engine decodes it and renders the
    # parameterized SQL, so a re-applied / registry-rehydrated view reproduces the same MV SELECT.
    agg = _agg("approx_percentile")
    out = agg.resolved_name(agg.time_window)
    view = _stream_view(_kafka_source(watermark=True), [agg])
    view.tags = encode_agg_params({out: [0.9, 100]})
    sql = _engine()._stream_mv_select("proj", view)
    assert f"approx_percentile(0.9, 0.01) WITHIN GROUP (ORDER BY amount) AS {out}" in sql


def test_agg_lifetime_carrier_round_trips_with_optional_floor():
    # The lifetime carrier marks which aggregations are lifetime (presence of the resolved_name), with an
    # optional floor epoch (None = no floor); unlike the param/offset carriers, a None value is kept (it
    # is a meaningful "lifetime, no floor"). is_lifetime_agg reads membership back by resolved_name.
    from feast.infra.compute_engines.risingwave.nodes import (
        AGG_LIFETIME_TAG,
        is_lifetime_agg,
        view_agg_lifetime,
    )

    assert encode_agg_lifetime({}) == {}
    enc = encode_agg_lifetime({"sum_amount": None, "spend_since": 1767225600})
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])
    view.tags = enc
    assert view_agg_lifetime(view) == {"sum_amount": None, "spend_since": 1767225600}
    # a lifetime aggregation carries no window; resolved_name is suffix-less, so membership resolves it
    lifetime = Aggregation(column="amount", function="sum", time_window=None)
    windowed = _agg("sum", 604800)
    lifetimes = view_agg_lifetime(view)
    assert is_lifetime_agg(lifetime, lifetimes) is True  # resolved_name 'sum_amount' in the carrier
    assert is_lifetime_agg(windowed, lifetimes) is False  # 'sum_amount_604800s' not in the carrier
    assert AGG_LIFETIME_TAG in enc


def test_agg_offset_carrier_round_trips_and_drops_zero():
    # The per-aggregation window offset rides the engine-owned offset tag (parallel to the param tag),
    # keyed by resolved_name. A zero offset (the trailing window) is omitted so a non-shifted view's
    # tags are left untouched; the inverse decodes the shifted entries back to whole seconds.
    assert encode_agg_offsets({"sum_amount_604800s": 0}) == {}
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])
    view.tags = encode_agg_offsets({"prev_week_sum": -604800, "trailing_sum": 0})
    assert view_agg_offsets(view) == {"prev_week_sum": -604800}
    assert view_agg_offsets(_stream_view(_kafka_source(watermark=True), [_agg("sum")])) == {}


# --- Batch tile aggregation (tile model: partial aggregates + retrieval rollup) ---
# The tile model (window-independent partial aggregates materialized once, then recombined per
# window at retrieval) is validated end-to-end on RisingWave v3.0.0.


def test_tile_partials_conform_to_feast_tiling_ir_decomposition():
    # Feast's tiling module is the CANONICAL reference for the partial-aggregate IR algebra (which
    # sub-aggregates each function tiles into). Pin OUR _partials_for to it for every function Feast's
    # tiling covers, so the shared algebra cannot drift (we render to SQL; Feast's orchestrator is pandas,
    # not reusable for pushdown — but the decomposition must agree). Feast returns ir_columns=None for an
    # algebraic function (the value IS the partial) and a list for a composite one. Our pop/samp variants
    # map to Feast's base name: the IR SET is identical, only the recombine denominator differs.
    from feast.aggregation.tiling.base import get_ir_metadata_for_aggregation

    feast_base = {"var_pop": "var", "var_samp": "var", "stddev_pop": "std", "stddev_samp": "std"}

    def feast_ir_count(function):
        _, meta = get_ir_metadata_for_aggregation(
            Aggregation(column="amount", function=feast_base.get(function, function)), "amount"
        )
        return 1 if meta.ir_columns is None else len(meta.ir_columns)

    for fn in ["sum", "count", "min", "max", "mean", "var_pop", "var_samp", "stddev_pop", "stddev_samp"]:
        ours = len(_partials_for(Aggregation(column="amount", function=fn)))
        assert ours == feast_ir_count(fn), f"{fn}: our {ours} partials != Feast's {feast_ir_count(fn)} IRs"
    # the exact families, not just the counts: mean = {sum, count}; variance/stddev = {sum, count, sumsq}
    assert {n for n, _ in _partials_for(_agg("mean"))} == {"sum_amount", "count_amount"}
    assert {n for n, _ in _partials_for(_agg("var_pop"))} == {"sum_amount", "count_amount", "sumsq_amount"}

    # Our EXTENSIONS are deliberately BEYOND Feast's tiling: it REJECTS count_distinct outright (and omits
    # sequence), so we own those (exact set-union / bounded top-n). Assert the boundary is real — if Feast
    # later adds count_distinct tiling, this raise-check fails and we fold it into the shared algebra.
    with pytest.raises(ValueError, match="does not support tiling"):
        get_ir_metadata_for_aggregation(Aggregation(column="amount", function="count_distinct"), "amount")


def test_batch_tile_select_buckets_by_interval_and_stamps_tile_end():
    sql = build_batch_tile_select(
        _column_info(), [_agg("sum", 2592000)], "src", aggregation_interval=timedelta(days=1)
    )
    # 1-day tiles, stamped by tile_end (the event-time upper boundary of the tile).
    assert "date_trunc('day', event_ts) + INTERVAL '1 day' AS tile_end" in sql
    # the tile holds the WINDOW-INDEPENDENT partial sum (one sum_amount serves every window).
    assert "sum(amount) AS sum_amount" in sql
    assert "GROUP BY user_id, date_trunc('day', event_ts)" in sql


def test_batch_tile_count_partial_is_count():
    sql = build_batch_tile_select(
        _column_info(), [_agg("count", 2592000)], "src", aggregation_interval=timedelta(days=1)
    )
    assert "count(amount) AS count_amount" in sql


def test_batch_tile_rollup_recombines_partials_in_request_anchored_window():
    sql = build_tile_rollup_select(
        _column_info(), [_agg("sum", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), as_of_sql="$1",
    )
    # sum partials (window-independent sum_amount) roll up with sum, output under the per-window name.
    assert "sum(sum_amount) AS sum_amount_2592000s" in sql
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
    assert "sum(count_amount) AS count_amount_2592000s" in sql


def test_batch_tile_min_max_roll_up_with_min_max():
    tile = build_batch_tile_select(
        _column_info(), [_agg("max", 2592000)], "src", aggregation_interval=timedelta(days=1)
    )
    assert "max(amount) AS max_amount" in tile
    roll = build_tile_rollup_select(
        _column_info(), [_agg("max", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), as_of_sql="$1",
    )
    assert "max(max_amount) AS max_amount_2592000s" in roll


@pytest.mark.parametrize("function", ["approx_count_distinct"])
def test_batch_tile_rejects_unmergeable_aggregation(function):
    # approx_count_distinct (HLL) has no mergeable sketch across tiles — rejected clearly. (Exact
    # count_distinct DOES tile, via a per-tile distinct set; see the count_distinct tile tests below.)
    with pytest.raises(ValueError, match="not supported"):
        build_batch_tile_select(
            _column_info(), [_agg(function, 2592000)], "src", aggregation_interval=timedelta(days=1)
        )


@pytest.mark.parametrize(
    "function,order,distinct",
    [("last", "DESC", False), ("first", "ASC", False),
     ("last_distinct", "DESC", True), ("first_distinct", "ASC", True)],
)
def test_sequence_tile_partial_is_bounded_topn_per_tile(function, order, distinct):
    # The per-tile partial is the tile's OWN top-n (bounded to n), named with n so last(3) and
    # last(5) on one column are distinct partials. n rides agg_params keyed by resolved_name.
    agg = _agg(function, 2592000)
    out = agg.resolved_name(agg.time_window)
    sql = build_batch_tile_select(
        _column_info(), [agg], "src", aggregation_interval=timedelta(days=1), agg_params={out: [3]}
    )
    inner = f"array_agg(amount ORDER BY event_ts {order})"
    if distinct:
        inner = f"array_distinct({inner})"
    assert f"({inner})[1:3] AS {function}_amount_3" in sql


def test_sequence_tile_rollup_flattens_topn_in_tile_order_then_slices():
    # Rollup concatenates the per-tile top-n arrays in tile_end order (array_flatten preserves it),
    # then slices to n — the n most-recent across the window, bounded.
    agg = _agg("last", 2592000)
    out = agg.resolved_name(agg.time_window)
    sql = build_online_rollup_select(
        _column_info(), [agg], "tiles", aggregation_interval=timedelta(days=1), agg_params={out: [3]}
    )
    assert f"(array_flatten(array_agg(last_amount_3 ORDER BY tile_end DESC)))[1:3] AS {out}" in sql


def test_count_distinct_tile_partial_is_nullsafe_distinct_set():
    # The per-tile partial is the tile's DISTINCT SET (an array), NULL-filtered so the union+count
    # matches count(distinct <col>) which excludes NULL.
    sql = build_batch_tile_select(
        _column_info(), [_agg("count_distinct", 2592000)], "src", aggregation_interval=timedelta(days=1)
    )
    assert "array_agg(DISTINCT amount) FILTER (WHERE amount IS NOT NULL) AS distinct_amount" in sql


def test_count_distinct_tile_rollup_unions_sets_then_counts():
    # Rollup unions the per-tile distinct arrays (array_flatten) and counts distinct elements; an empty
    # window -> NULL (NULLIF) so the offline LEFT-JOIN matches the online MV's absent-entity NULL.
    out = "count_distinct_amount_2592000s"
    online = build_online_rollup_select(
        _column_info(), [_agg("count_distinct", 2592000)], "tiles", aggregation_interval=timedelta(days=1)
    )
    assert f"NULLIF(cardinality(array_distinct(array_flatten(array_agg(distinct_amount)))), 0) AS {out}" in online


def test_batch_tile_mean_emits_sum_and_count_partials():
    sql = build_batch_tile_select(
        _column_info(), [_agg("mean", 2592000)], "src", aggregation_interval=timedelta(days=1)
    )
    # mean reuses the window-independent sum_amount + count_amount partials (no per-window __sm/__cnt).
    assert "sum(amount) AS sum_amount" in sql
    assert "count(amount) AS count_amount" in sql


def test_mean_rollup_recombines_sum_over_count():
    sql = build_tile_rollup_select(
        _column_info(), [_agg("mean", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), as_of_sql="$1",
    )
    assert (
        "sum(sum_amount) / NULLIF(sum(count_amount), 0) AS mean_amount_2592000s" in sql
    )


def test_batch_tile_stddev_emits_sumsq_partial():
    sql = build_batch_tile_select(
        _column_info(), [_agg("stddev_pop", 2592000)], "src", aggregation_interval=timedelta(days=1)
    )
    assert "sum(amount * amount) AS sumsq_amount" in sql


def test_stddev_pop_rollup_is_sqrt_of_population_variance():
    sql = build_tile_rollup_select(
        _column_info(), [_agg("stddev_pop", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), as_of_sql="$1",
    )
    assert "sqrt(" in sql  # stddev = sqrt(variance)
    assert "sumsq_amount" in sql  # uses the window-independent sum-of-squares partial
    assert "NULLIF(sum(count_amount), 0)) AS stddev_pop_amount_2592000s" in sql  # /n
    # GREATEST(..., 0) clamps the centered sum-of-squares so rounding can't yield a negative variance /
    # a sqrt-of-negative runtime failure (matches RisingWave's own native var/stddev plan).
    assert "GREATEST(" in sql


def test_var_samp_rollup_divides_by_n_minus_1():
    sql = build_tile_rollup_select(
        _column_info(), [_agg("var_samp", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), as_of_sql="$1",
    )
    assert "NULLIF(sum(count_amount) - 1, 0) AS var_samp_amount_2592000s" in sql
    assert "GREATEST(" in sql  # non-negative variance clamp (no negative variance from cancellation)


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
    # combiner rolls the window-independent sum_amount partial up under the per-window feature name
    assert "sum(sum_amount) AS sum_amount_2592000s" in sql
    # one row per entity; window_end is the PIT stamp the point-lookup ORDER BYs
    assert "max(tile_end) AS window_end" in sql
    assert "GROUP BY user_id" in sql


def test_online_rollup_count_combiner_is_sum():
    sql = build_online_rollup_select(
        _column_info(), [_agg("count", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1),
    )
    assert "sum(count_amount) AS count_amount_2592000s" in sql


@pytest.mark.parametrize("function", ["approx_count_distinct"])
def test_online_rollup_rejects_non_additive_aggregation(function):
    with pytest.raises(ValueError, match="not supported"):
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


def test_online_rollup_is_single_window_one_mv_per_window():
    # Single-window-per-MV boundary: the now()-anchored online MV is per-window (RW rejects now() inside a CASE in a
    # two-sided temporal-filter MV, so multi-window online = N separate per-window MVs the engine loops
    # over). The builder therefore takes a SINGLE window; two windows is rejected, not silently merged.
    # (Multi-window OFFLINE, by contrast, is one query — see the multi-window PIT tests above.)
    with pytest.raises(ValueError, match="single non-null time_window"):
        build_online_rollup_select(
            _column_info(), [_agg("sum", 259200), _agg("sum", 2592000)], "tiles",
            aggregation_interval=timedelta(days=1),
        )


# --- window offset: a window shifted into the past (week-over-week / lag) -------------------------


def test_online_rollup_offset_zero_is_byte_identical():
    # offset=0 (the default) must emit EXACTLY today's two-sided now() window — the reconcile compares
    # the stored MV SELECT verbatim, so any drift would needlessly re-materialize every existing MV.
    base = build_online_rollup_select(
        _column_info(), [_agg("sum", 2592000)], "tiles", aggregation_interval=timedelta(days=1)
    )
    with_zero = build_online_rollup_select(
        _column_info(), [_agg("sum", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), offset_secs=0,
    )
    assert base == with_zero
    assert "tile_end > now() - INTERVAL '2592000' SECOND AND tile_end <= now()" in base


def test_online_rollup_offset_shifts_both_bounds_into_the_past():
    # A 7d window at offset -7d is the PREVIOUS week (now-14d, now-7d]: the lower bound deepens by
    # |offset| and the upper bound retreats from now() to now()-|offset|.
    sql = build_online_rollup_select(
        _column_info(), [_agg("sum", 604800)], "tiles",
        aggregation_interval=timedelta(days=1), offset_secs=-604800,
    )
    assert (
        "tile_end > now() - INTERVAL '1209600' SECOND AND tile_end <= now() - INTERVAL '604800' SECOND"
        in sql
    )
    assert "tile_end <= now()," not in sql and "tile_end <= now() GROUP" not in sql  # upper is shifted


def test_group_aggregations_by_window_offset_splits_same_window_by_offset():
    # Two aggregates share a 7d window but differ in offset (trailing vs previous week). They cannot
    # share one now()-anchored MV (the WHERE bounds differ), so the (window, offset) grouping splits them.
    trailing = _agg("sum", 604800)
    prev = Aggregation(column="amount", function="sum", time_window=timedelta(days=7), name="prev_week_sum")
    offsets = {prev.resolved_name(prev.time_window): -604800}
    groups = group_aggregations_by_window_offset([trailing, prev], offsets)
    assert [key for key, _ in groups] == [(604800, -604800), (604800, 0)]  # ascending; offset sorts first
    by_key = {key: aggs for key, aggs in groups}
    assert by_key[(604800, 0)] == [trailing]
    assert by_key[(604800, -604800)] == [prev]


def test_offline_tile_pit_offset_adds_upper_bound_and_extends_join():
    prev = Aggregation(column="amount", function="sum", time_window=timedelta(days=7), name="prev_week_sum")
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[_agg("sum", 604800), prev],
        aggregation_interval=timedelta(days=1),
        offsets={prev.resolved_name(prev.time_window): -604800},
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    # the trailing 7d agg keeps its lower-only CASE (offset 0 = byte-identical)
    assert (
        f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '604800' SECOND THEN t.sum_amount END) "
        "AS sum_amount_604800s" in sql
    )
    # the previous-week agg gets BOTH bounds: (end-14d, end-7d]
    assert (
        f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '1209600' SECOND "
        f"AND t.tile_end <= {end} - INTERVAL '604800' SECOND THEN t.sum_amount END) AS prev_week_sum"
        in sql
    )
    # the join now reads back to the deepest tile any agg needs: 7d + |offset 7d| = 14d
    assert f"t.tile_end > {end} - INTERVAL '1209600' SECOND AND t.tile_end <= {end}" in sql


def test_online_rollup_rejects_offset_not_multiple_of_interval():
    # The offset shifts the window by a count of tiles, so a non-multiple |offset| (1h on a 1-day tile)
    # would make the online now()-anchored bound and the offline floor-anchored bound select different
    # tiles -> online != offline. Guard it at the builder, not only at the authoring factory.
    with pytest.raises(ValueError, match="offset"):
        build_online_rollup_select(
            _column_info(), [_agg("sum", 604800)], "tiles",
            aggregation_interval=timedelta(days=1), offset_secs=-3600,
        )


def test_offline_tile_pit_rejects_offset_not_multiple_of_interval():
    prev = Aggregation(column="amount", function="sum", time_window=timedelta(days=7), name="prev")
    with pytest.raises(ValueError, match="offset"):
        build_offline_tile_pit_query(
            "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
            tiles_relation="t", column_info=_column_info(), aggregations=[prev],
            aggregation_interval=timedelta(days=1),
            offsets={prev.resolved_name(prev.time_window): -3600},  # 1h not a multiple of 1-day
        )


def test_offline_tile_pit_offset_zero_is_byte_identical():
    # offsets=None / all-zero must reproduce today's multi-window CASE exactly (back-compat).
    aggs = [_agg("sum", 259200), _agg("sum", 2592000)]
    base = build_offline_tile_pit_query(
        "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(), aggregations=aggs,
        aggregation_interval=timedelta(days=1),
    )
    with_empty = build_offline_tile_pit_query(
        "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(), aggregations=aggs,
        aggregation_interval=timedelta(days=1), offsets={},
    )
    assert base == with_empty


# --- aggregation secondary key: a per-key Map breakdown -------------------------------------------


def test_secondary_key_tiles_add_the_group_by_dimension():
    batch = build_batch_tile_select(
        _column_info(), [_agg("sum", 2592000)], "src",
        aggregation_interval=timedelta(days=1), secondary_key="ad_id",
    )
    assert '"ad_id", date_trunc' in batch  # secondary key projected before tile_end
    assert 'GROUP BY user_id, "ad_id", date_trunc' in batch
    stream = build_streaming_tile_select(
        _column_info(), [_agg("sum", 2592000)], "src",
        aggregation_interval=timedelta(days=1), secondary_key="ad_id",
    )
    assert '"ad_id", window_end AS tile_end' in stream
    assert 'GROUP BY window_start, window_end, user_id, "ad_id" EMIT ON WINDOW CLOSE' in stream


def test_secondary_key_rejects_clash_with_join_key_or_output():
    # the secondary key is a SEPARATE GROUP BY dimension; colliding it with a join key (or an aggregation
    # output / the timestamp) would double-list the column — reject at build time, not as an opaque RW error.
    with pytest.raises(ValueError, match="distinct raw column"):
        build_batch_tile_select(
            _column_info(), [_agg("sum", 2592000)], "src",
            aggregation_interval=timedelta(days=1), secondary_key="user_id",  # == the join key
        )


def test_secondary_key_byte_identical_when_absent():
    # no secondary key -> the tile/rollup SQL is unchanged (back-compat with every non-breakdown view).
    base = build_batch_tile_select(_column_info(), [_agg("sum", 2592000)], "src", aggregation_interval=timedelta(days=1))
    assert base == build_batch_tile_select(
        _column_info(), [_agg("sum", 2592000)], "src", aggregation_interval=timedelta(days=1), secondary_key=None
    )


def test_online_rollup_secondary_key_nests_jsonb_object_agg():
    sql = build_online_rollup_select(
        _column_info(), [_agg("sum", 2592000)], "tiles",
        aggregation_interval=timedelta(days=1), secondary_key="ad_id",
    )
    # inner: per (entity, secondary_key) recombine over the SAME now() window
    assert 'sum(sum_amount) AS sum_amount_2592000s' in sql
    assert "tile_end > now() - INTERVAL '2592000' SECOND AND tile_end <= now()" in sql
    assert 'GROUP BY user_id, "ad_id"' in sql
    # outer: collapse the secondary key into a per-aggregation Map, NULL keys filtered, empty -> NULL
    assert (
        'NULLIF(jsonb_object_agg("ad_id", sum_amount_2592000s) '
        "FILTER (WHERE \"ad_id\" IS NOT NULL), '{}'::jsonb) AS sum_amount_2592000s" in sql
    )
    assert "GROUP BY user_id" in sql.rsplit("FROM", 1)[1]


def test_offline_pit_secondary_key_nests_jsonb_object_agg():
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(), aggregations=[_agg("sum", 259200)],
        aggregation_interval=timedelta(days=1), secondary_key="ad_id",
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    # inner: per (entity row, secondary_key) windowed recombine via the LEFT JOIN
    assert 't."ad_id" AS "ad_id"' in sql
    assert f'GROUP BY e."user_id", e."event_timestamp", t."ad_id"' in sql
    # outer: per-entity-row Map, NULL secondary key (a join miss) filtered, empty -> NULL
    assert (
        'NULLIF(jsonb_object_agg("ad_id", sum_amount_259200s) '
        "FILTER (WHERE \"ad_id\" IS NOT NULL), '{}'::jsonb) AS sum_amount_259200s" in sql
    )


# --- lifetime window: all-history rollup, lower bound dropped --------------------------------------


def _lifetime_agg(function="sum", column="amount"):
    # a lifetime aggregation lowers to a feast.Aggregation with NO time_window; its resolved name is
    # suffix-less ({fn}_{col}) and it is marked lifetime by the carrier.
    return Aggregation(column=column, function=function, time_window=None)


def test_lifetime_rollup_one_sided_now_bound_no_lower():
    agg = _lifetime_agg("sum")
    sql = build_lifetime_rollup_select(_column_info(), [agg], "tiles")
    assert "WHERE tile_end <= now() GROUP BY user_id" in sql  # one-sided: no lower bound
    assert "now() - INTERVAL" not in sql  # nothing trailing
    assert "sum(sum_amount) AS sum_amount" in sql  # recombine identical to the windowed rollup
    assert "max(tile_end) AS window_end" in sql


def test_lifetime_rollup_with_floor_adds_lower_bound():
    sql = build_lifetime_rollup_select(_column_info(), [_lifetime_agg("sum")], "tiles", lifetime_start_secs=1767225600)
    assert "WHERE tile_end <= now() AND tile_end > (to_timestamp(1767225600) AT TIME ZONE 'UTC')" in sql


def test_group_lifetime_aggregations_by_floor_none_first():
    a = _lifetime_agg("sum")  # resolved_name sum_amount
    b = Aggregation(column="amount", function="sum", time_window=None, name="since_jan")
    c = Aggregation(column="amount", function="count", time_window=None, name="cnt_lifetime")
    lifetimes = {"sum_amount": None, "since_jan": 1767225600, "cnt_lifetime": None}
    groups = group_lifetime_aggregations([a, b, c], lifetimes)
    assert [floor for floor, _ in groups] == [None, 1767225600]  # no-floor group first, then floored
    by_floor = {floor: aggs for floor, aggs in groups}
    assert by_floor[None] == [a, c]
    assert by_floor[1767225600] == [b]


def test_offline_pit_lifetime_drops_join_lower_bound_and_reads_all_tiles():
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(), aggregations=[_lifetime_agg("sum")],
        aggregation_interval=timedelta(days=1), lifetimes={"sum_amount": None},
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    # the join keeps only the upper bound (<= end); the lower bound is gone — lifetime reads all history
    assert f"AND t.tile_end <= {end} GROUP BY" in sql
    assert "t.tile_end >" not in sql.split("LEFT JOIN")[1].split("GROUP BY")[0]  # no lower bound anywhere in the join
    # a no-floor lifetime agg recombines the raw partial (no CASE narrowing)
    assert "sum(t.sum_amount) AS sum_amount" in sql


def test_offline_pit_lifetime_mixed_with_windowed():
    # a windowed 7d sum alongside a lifetime sum: windowed keeps its CASE; the join still drops its lower
    # bound (lifetime needs all history); the lifetime floored agg gets a tile_end > floor CASE.
    w = _agg("sum", 604800)
    life = Aggregation(column="amount", function="sum", time_window=None, name="since_jan")
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(), aggregations=[w, life],
        aggregation_interval=timedelta(days=1), lifetimes={"since_jan": 1767225600},
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    assert f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '604800' SECOND THEN t.sum_amount END) AS sum_amount_604800s" in sql
    assert "sum(CASE WHEN t.tile_end > (to_timestamp(1767225600) AT TIME ZONE 'UTC') THEN t.sum_amount END) AS since_jan" in sql
    assert f"AND t.tile_end <= {end} GROUP BY" in sql  # join lower bound dropped


# --- offline tile PIT: floor-anchored range-agg join, per entity-row label -------------------------


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
    # each agg recombines its OWN window via a CASE on tile_end over the window-independent partial
    assert "THEN t.sum_amount END) AS sum_amount_259200s" in sql
    assert 'GROUP BY e."user_id", e."event_timestamp"' in sql  # one output row per entity-label


def test_offline_tile_pit_count_combiner_is_sum():
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[_agg("count", 259200)], aggregation_interval=timedelta(days=1),
    )
    assert "THEN t.count_amount END) AS count_amount_259200s" in sql


@pytest.mark.parametrize("function", ["approx_count_distinct"])
def test_offline_tile_pit_rejects_non_additive(function):
    with pytest.raises(ValueError, match="not supported"):
        build_offline_tile_pit_query(
            "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
            tiles_relation="t", column_info=_column_info(),
            aggregations=[_agg(function, 259200)], aggregation_interval=timedelta(days=1),
        )


def test_offline_tile_pit_does_not_apply_ttl_only_the_window_bounds():
    # For an aggregation FV the time_window IS the lookback bound (windowed-aggregation semantics); ttl is NOT a
    # second bound. Pin it: exactly one tile_end lower bound (the window), no extra ttl filter.
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[_agg("sum", 259200)], aggregation_interval=timedelta(days=1),
    )
    # two window-derived lower bounds — the join (max window) + the per-agg CASE — both '259200' (the
    # window); NO third ttl bound. Pin: every lower bound is the window interval, and no 'ttl' filter.
    assert sql.count("t.tile_end >") == 2
    assert sql.count("- INTERVAL '259200' SECOND") == 2
    assert "ttl" not in sql.lower()


# --- Streaming tiles: the EOWC-tumble twin of build_batch_tile_select ---


def test_streaming_tile_select_mirrors_batch_partials_via_eowc_tumble():
    # The streaming tiles MV emits the SAME deduped window-independent partials as the batch tiles MV
    # (shared _tile_partials_projection), materialized by an EOWC TUMBLE at the interval instead of date_trunc.
    aggs = [_agg("sum", 259200), _agg("mean", 604800)]  # partials dedup -> sum_amount, count_amount
    proj = "sum(amount) AS sum_amount, count(amount) AS count_amount"
    sql = build_streaming_tile_select(_column_info(), aggs, "src", aggregation_interval=timedelta(days=1))
    assert proj in sql
    assert proj in build_batch_tile_select(_column_info(), aggs, "src", aggregation_interval=timedelta(days=1))
    assert "tumble(src, event_ts, INTERVAL '86400' SECOND)" in sql  # EOWC tumble AT the interval
    assert "window_end AS tile_end" in sql  # tile_end = the tumble close boundary
    assert "GROUP BY window_start, window_end, user_id" in sql
    assert sql.rstrip().endswith("EMIT ON WINDOW CLOSE")


@pytest.mark.parametrize("secs", [604800, 900, 2592000])  # week, 15min, 30d
def test_streaming_tile_select_rejects_non_grid_aligned_interval(secs):
    # epoch-anchored TUMBLE must match the offline date_trunc floor -> only minute/hour/day are grid-aligned
    # (week mis-grids: TUMBLE epoch-Thursday vs date_trunc ISO-Monday; 15min/30d are not date_trunc units).
    with pytest.raises(ValueError, match="1 hour .* or 1 day"):
        build_streaming_tile_select(
            _column_info(), [_agg("sum", 259200)], "src", aggregation_interval=timedelta(seconds=secs)
        )


def test_batch_tile_select_supports_minute_interval():
    # 'minute' is a native, epoch-aligned date_trunc unit, so a 1-minute batch tile is valid — the
    # sub-hour grain fraud velocity features need (e.g. txn count over 1m/5m/30m windows).
    sql = build_batch_tile_select(
        _column_info(), [_agg("count", 300)], "src", aggregation_interval=timedelta(minutes=1)
    )
    assert "date_trunc('minute', event_ts) + INTERVAL '1 minute' AS tile_end" in sql
    assert "GROUP BY user_id, date_trunc('minute', event_ts)" in sql


def test_streaming_tile_select_supports_minute_interval():
    # a 60s EOWC TUMBLE is epoch-anchored to the SAME grid as date_trunc('minute'), so minute streaming
    # tiles keep online == offline — the high-freshness fraud-velocity grain.
    sql = build_streaming_tile_select(
        _column_info(), [_agg("count", 300)], "src", aggregation_interval=timedelta(minutes=1)
    )
    assert "tumble(src, event_ts, INTERVAL '60' SECOND)" in sql
    assert sql.rstrip().endswith("EMIT ON WINDOW CLOSE")


def test_streaming_tile_select_rejects_unsupported_and_empty():
    with pytest.raises(ValueError, match="not supported"):  # same tile-supported contract as batch
        build_streaming_tile_select(
            _column_info(), [_agg("approx_count_distinct", 259200)], "src", aggregation_interval=timedelta(hours=1)
        )
    with pytest.raises(ValueError, match="at least one aggregation"):
        build_streaming_tile_select(_column_info(), [], "src", aggregation_interval=timedelta(hours=1))


# --- Multi-window from ONE tile set (tiles reused across time-windows) ---


def test_batch_tile_partials_are_window_independent_and_deduped():
    # sum@3d + sum@30d + mean@7d on the SAME column share ONE sum_amount partial; mean adds count_amount.
    sql = build_batch_tile_select(
        _column_info(),
        [_agg("sum", 259200), _agg("sum", 2592000), _agg("mean", 604800)],
        "src", aggregation_interval=timedelta(days=1),
    )
    assert sql.count("AS sum_amount") == 1  # one partial for both sum windows AND mean's sum
    assert sql.count("AS count_amount") == 1  # one partial for mean's count
    assert "sum(amount) AS sum_amount" in sql
    assert "count(amount) AS count_amount" in sql
    # partials carry NO per-window suffix (that lives only on the rollup OUTPUT names)
    assert "sum_amount_" not in sql and "count_amount_" not in sql


def test_offline_tile_pit_multi_window_rolls_each_window_from_one_tile_set():
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[_agg("sum", 259200), _agg("sum", 2592000)],
        aggregation_interval=timedelta(days=1),
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    # the JOIN reads tiles up to the MAX window (30d) once
    assert f"t.tile_end > {end} - INTERVAL '2592000' SECOND AND t.tile_end <= {end}" in sql
    # each window recombines the SAME sum_amount partial, narrowed to its own window by a CASE
    assert (
        f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '259200' SECOND THEN t.sum_amount END) "
        "AS sum_amount_259200s" in sql
    )
    assert (
        f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '2592000' SECOND THEN t.sum_amount END) "
        "AS sum_amount_2592000s" in sql
    )


def test_offline_tile_pit_multi_window_mean_uses_filtered_sum_and_count():
    # mean@7d alongside sum@30d: mean recombines sum_amount/count_amount BOTH filtered to its 7d window,
    # while the join is bounded by the max (30d) window.
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[_agg("mean", 604800), _agg("sum", 2592000)],
        aggregation_interval=timedelta(days=1),
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    case_sum = f"CASE WHEN t.tile_end > {end} - INTERVAL '604800' SECOND THEN t.sum_amount END"
    case_cnt = f"CASE WHEN t.tile_end > {end} - INTERVAL '604800' SECOND THEN t.count_amount END"
    assert f"sum({case_sum}) / NULLIF(sum({case_cnt}), 0) AS mean_amount_604800s" in sql
    assert "INTERVAL '2592000' SECOND AND t.tile_end <=" in sql  # join bounded by the 30d max window


def test_group_aggregations_by_window_groups_distinct_windows_ascending():
    # the window-only grouping (offline precondition + no-offset harnesses): distinct windows, ascending.
    a3d = _agg("sum", 259200)
    a30d = _agg("sum", 2592000)
    mean7d = _agg("mean", 604800)
    groups = group_aggregations_by_window([a30d, a3d, mean7d, a3d])
    assert [secs for secs, _ in groups] == [259200, 604800, 2592000]  # ascending, distinct
    by_secs = {secs: aggs for secs, aggs in groups}
    assert by_secs[259200] == [a3d, a3d]  # both 3d aggs land in one group
    assert by_secs[604800] == [mean7d]


def test_offline_tile_pit_multi_window_still_requires_window_multiple_of_interval():
    # distinct windows are allowed, but EACH must be a whole number of tiles
    with pytest.raises(ValueError, match="multiple"):
        build_offline_tile_pit_query(
            "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
            tiles_relation="t", column_info=_column_info(),
            aggregations=[_agg("sum", 259200), _agg("sum", 3600)],  # 1h is not a multiple of 1-day
            aggregation_interval=timedelta(days=1),
        )


def test_offline_tile_pit_series_emits_array_of_per_step_recombines_oldest_first():
    # A window-series fans one aggregation into L windows -> ARRAY of L per-step recombines over the ONE
    # shared tile set, ordered OLDEST window first. Each element is the per-window recombine narrowed to
    # its step (end - W - i*step, end - i*step], the L-fold copy of the offset CASE; the join reads back
    # to the deepest step once.
    series_agg = Aggregation(column="amount", function="sum", time_window=None, name="daily_sum_3")
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[series_agg],
        aggregation_interval=timedelta(days=1),
        series={"daily_sum_3": [86400, 86400, 3]},
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    oldest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '259200' SECOND AND t.tile_end <= {end} - INTERVAL '172800' SECOND THEN t.sum_amount END)"
    middle = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '172800' SECOND AND t.tile_end <= {end} - INTERVAL '86400' SECOND THEN t.sum_amount END)"
    newest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '86400' SECOND THEN t.sum_amount END)"
    assert f"ARRAY[{oldest}, {middle}, {newest}] AS daily_sum_3" in sql
    # the join reads back to the deepest step once: window + (L-1)*step = 1d + 2*1d = 3d
    assert f"t.tile_end > {end} - INTERVAL '259200' SECOND AND t.tile_end <= {end}" in sql


def test_offline_tile_pit_series_count_combiner_is_sum_of_tile_counts():
    # count over a step recombines by SUMMING per-tile counts; an empty step (no tiles) -> NULL element
    # (sum over no rows), matching established feature stores' empty-window = None and the online assembled array.
    series_agg = Aggregation(column="amount", function="count", time_window=None, name="daily_count_2")
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[series_agg],
        aggregation_interval=timedelta(days=1),
        series={"daily_count_2": [86400, 86400, 2]},
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    oldest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '172800' SECOND AND t.tile_end <= {end} - INTERVAL '86400' SECOND THEN t.count_amount END)"
    newest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '86400' SECOND THEN t.count_amount END)"
    assert f"ARRAY[{oldest}, {newest}] AS daily_count_2" in sql


def test_offline_tile_pit_series_mixes_with_a_windowed_agg():
    # a series alongside a plain windowed agg: both roll up from the ONE tile set; the join reads back to
    # the deeper of (the windowed window) and (the series depth).
    series_agg = Aggregation(column="amount", function="sum", time_window=None, name="daily_sum_5")
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[_agg("sum", 86400), series_agg],
        aggregation_interval=timedelta(days=1),
        series={"daily_sum_5": [86400, 86400, 5]},
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    assert "ARRAY[" in sql and "AS daily_sum_5" in sql
    # the windowed 1d agg still emits its scalar recombine
    assert f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '86400' SECOND THEN t.sum_amount END) AS sum_amount_86400s" in sql
    # join reads back to the deepest: series depth 5d (432000) > the 1d windowed agg
    assert f"t.tile_end > {end} - INTERVAL '432000' SECOND AND t.tile_end <= {end}" in sql


def test_offline_tile_pit_series_max_overlap_recombines_over_window_tiles():
    # MAX is served by the SAME single-scan recombine as the invertible family: each OVERLAPPING element
    # is max(CASE WHEN <its window> THEN t.max_amount END) over ALL tiles in the window — not a one-tile
    # pick. window 2d > step 1d (overlap), L=2.
    ser = Aggregation(column="amount", function="max", time_window=None, name="roll2d_max_2")
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[ser], aggregation_interval=timedelta(days=1),
        series={"roll2d_max_2": [172800, 86400, 2]},  # window 2d, step 1d (overlap)
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    oldest = f"max(CASE WHEN t.tile_end > {end} - INTERVAL '259200' SECOND AND t.tile_end <= {end} - INTERVAL '86400' SECOND THEN t.max_amount END)"  # (end-3d, end-1d]
    newest = f"max(CASE WHEN t.tile_end > {end} - INTERVAL '172800' SECOND THEN t.max_amount END)"  # (end-2d, end]
    assert f"ARRAY[{oldest}, {newest}] AS roll2d_max_2" in sql


def test_offline_tile_pit_series_window_smaller_than_step_leaves_gaps():
    # window < step is a sparse (gapped) series: each element samples a 1-day window, but consecutive
    # windows are 2 days apart, so a day between them is covered by no element. Valid (each element is its
    # own window); confirm the per-element bounds reflect the gaps (window 1d, step 2d, L=2).
    ser = Aggregation(column="amount", function="sum", time_window=None, name="sparse_sum_2")
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[ser], aggregation_interval=timedelta(days=1),
        series={"sparse_sum_2": [86400, 172800, 2]},  # window 1d, step 2d (gaps)
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    oldest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '259200' SECOND AND t.tile_end <= {end} - INTERVAL '172800' SECOND THEN t.sum_amount END)"  # (end-3d, end-2d]
    newest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '86400' SECOND THEN t.sum_amount END)"  # (end-1d, end]; the (end-2d,end-1d] day is sampled by NO element
    assert f"ARRAY[{oldest}, {newest}] AS sparse_sum_2" in sql


def test_offline_tile_pit_series_rejects_step_not_multiple_of_interval():
    # the series step must be a whole number of tiles (the same parity invariant the window obeys), so a
    # non-multiple step would make the offline floor-anchored element diverge from the online assembled one.
    bad = Aggregation(column="amount", function="sum", time_window=None, name="bad_series")
    with pytest.raises(ValueError, match="multiple"):
        build_offline_tile_pit_query(
            "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
            tiles_relation="t", column_info=_column_info(),
            aggregations=[bad], aggregation_interval=timedelta(days=1),
            series={"bad_series": [3600, 3600, 4]},  # 1h step on a 1-day tile
        )


def test_offline_tile_pit_series_coarse_step_recombines_multiple_tiles_per_element():
    # step = 2*interval (k=2): each element's window spans TWO tiles and the CASE recombine sums both —
    # a single-scan recombine, NOT a one-tile placement. window == step here (non-overlapping coarse).
    ser = Aggregation(column="amount", function="sum", time_window=None, name="biday_sum_2")
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[ser], aggregation_interval=timedelta(days=1),
        series={"biday_sum_2": [172800, 172800, 2]},  # 2-day window == 2-day step on a 1-day tile
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    # element 0 (oldest): window (end-4d, end-2d]; element 1 (newest): (end-2d, end]
    oldest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '345600' SECOND AND t.tile_end <= {end} - INTERVAL '172800' SECOND THEN t.sum_amount END)"
    newest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '172800' SECOND THEN t.sum_amount END)"
    assert f"ARRAY[{oldest}, {newest}] AS biday_sum_2" in sql
    assert f"t.tile_end > {end} - INTERVAL '345600' SECOND AND t.tile_end <= {end}" in sql  # reads back 4d


def test_offline_tile_pit_series_overlapping_windows():
    # OVERLAPPING: window (2d) > step (1d). Consecutive elements' windows overlap (share a tile); each
    # element independently re-selects its own (end-2d-i*1d, end-i*1d] tile set — double-counting across
    # elements is intended (overlap). The deepest read-back is window + (L-1)*step = 2d + 2*1d = 4d.
    ser = Aggregation(column="amount", function="sum", time_window=None, name="roll2d_sum_3")
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[ser], aggregation_interval=timedelta(days=1),
        series={"roll2d_sum_3": [172800, 86400, 3]},  # window 2d, step 1d (overlap), length 3
    )
    end = "date_trunc('day', e.\"event_timestamp\")"
    oldest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '345600' SECOND AND t.tile_end <= {end} - INTERVAL '172800' SECOND THEN t.sum_amount END)"  # (end-4d, end-2d]
    middle = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '259200' SECOND AND t.tile_end <= {end} - INTERVAL '86400' SECOND THEN t.sum_amount END)"  # (end-3d, end-1d]
    newest = f"sum(CASE WHEN t.tile_end > {end} - INTERVAL '172800' SECOND THEN t.sum_amount END)"  # (end-2d, end]
    assert f"ARRAY[{oldest}, {middle}, {newest}] AS roll2d_sum_3" in sql
    assert f"t.tile_end > {end} - INTERVAL '345600' SECOND AND t.tile_end <= {end}" in sql  # reads back 4d once


def test_offline_tile_pit_series_rejects_non_positive_length():
    # a length-0 series would emit an empty ARRAY[] literal RisingWave rejects; fail fast at the builder.
    bad = Aggregation(column="amount", function="sum", time_window=None, name="empty_series")
    with pytest.raises(ValueError, match="length"):
        build_offline_tile_pit_query(
            "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
            tiles_relation="t", column_info=_column_info(),
            aggregations=[bad], aggregation_interval=timedelta(days=1),
            series={"empty_series": [86400, 86400, 0]},
        )


def test_offline_tile_pit_rejects_empty_aggregations():
    # same fail-fast guard as the other three tile builders (clear message, not a cryptic max([]) error)
    with pytest.raises(ValueError, match="at least one aggregation"):
        build_offline_tile_pit_query(
            "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
            tiles_relation="t", column_info=_column_info(),
            aggregations=[], aggregation_interval=timedelta(days=1),
        )


def test_tile_rejects_zero_length_window():
    # timedelta(0) is non-null and 0 % interval == 0, so it slips past the None + multiple guards and
    # would emit an always-empty (end, end] range -> silently all-NULL. Reject it with a clear message.
    zero = Aggregation(column="amount", function="sum", time_window=timedelta(0))
    with pytest.raises(ValueError, match="positive whole multiple"):
        build_online_rollup_select(
            _column_info(), [zero], "tiles", aggregation_interval=timedelta(days=1)
        )
    with pytest.raises(ValueError, match="positive whole multiple"):
        build_offline_tile_pit_query(
            "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
            tiles_relation="t", column_info=_column_info(),
            aggregations=[zero], aggregation_interval=timedelta(days=1),
        )


def test_tile_rejects_duplicate_output_names():
    # two aggregations sharing an explicit name resolve to the SAME output column (resolved_name uses
    # the name verbatim, ignoring the window) -> duplicate SELECT alias. Reject before emitting bad SQL.
    a = Aggregation(column="amount", function="sum", time_window=timedelta(days=3), name="total")
    b = Aggregation(column="amount", function="sum", time_window=timedelta(days=30), name="total")
    with pytest.raises(ValueError, match="duplicate output column"):
        build_offline_tile_pit_query(
            "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
            tiles_relation="t", column_info=_column_info(),
            aggregations=[a, b], aggregation_interval=timedelta(days=1),
        )


def test_batch_tile_rejects_entity_column_colliding_with_partial_name():
    # an entity/join key literally named like a tile partial ('sum_amount') would make the tiles MV have
    # two identically-named columns -> RW rejects the DDL. Fail fast with a clear message instead.
    ci = ColumnInfo(
        join_keys=["sum_amount"], feature_cols=["amount"], ts_col="event_ts",
        created_ts_col=None, field_mapping=None,
    )
    with pytest.raises(ValueError, match="collide with tile partial"):
        build_batch_tile_select(
            ci, [_agg("sum", 2592000)], "src", aggregation_interval=timedelta(days=1)
        )


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


class _ReconcileCur:
    # Records executed DDL; answers the MV-definition lookup (rw_materialized_views) with deployed_def, the
    # source lookup (rw_sources) with source_def — or history_def for a "*_history" source name — and the
    # source column-schema lookup (information_schema.columns) with source_cols. None => the row is absent.
    def __init__(self, deployed_def, source_def=None, source_cols=None, history_def=None):
        self.executed = []
        self._deployed = deployed_def
        self._source_def = source_def
        self._source_cols = source_cols
        self._history_def = history_def
        self._row = None
        self._cols_query = False

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self._cols_query = "information_schema.columns" in sql
        if sql.startswith("SELECT definition"):
            if "rw_sources" in sql:
                name = params[0] if params else ""
                val = self._history_def if name.endswith("_history") else self._source_def
            else:
                val = self._deployed
            self._row = (val,) if val is not None else None

    def fetchone(self):
        return self._row

    def fetchall(self):
        return list(self._source_cols) if (self._cols_query and self._source_cols) else []


def _rendered_kafka_source_def(topic="txn", bootstrap="localhost:9092", watermark_secs=30):
    # The CREATE SOURCE definition as RisingWave RE-RENDERS it in rw_catalog.rw_sources (verified live on
    # v3.0.0): spaces around '=' in the WITH clause, the WATERMARK in the column list, option names (incl.
    # dotted) preserved, single quotes doubled. Used to prove the spacing-tolerant extraction matches the
    # real rendering, not just the no-space form the engine emits.
    t = topic.replace("'", "''")
    b = bootstrap.replace("'", "''")
    return (
        'CREATE SOURCE "proj_user_txn_src" ("user_id" VARCHAR, "amount" DOUBLE, '
        '"event_ts" TIMESTAMP, WATERMARK FOR "event_ts" AS "event_ts" - '
        f"INTERVAL '{watermark_secs}' SECOND) WITH (connector = 'kafka', "
        f"properties.bootstrap.server = '{b}', topic = '{t}', "
        "scan.startup.mode = 'earliest') FORMAT PLAIN ENCODE JSON"
    )


def test_deployed_kafka_source_opts_extracts_from_rerendered_definition():
    # RW re-renders the WITH clause (spaces around '=') and doubles single quotes, so the extraction must be
    # spacing-tolerant and un-double — verified against the real v3.0.0 rendering shape.
    cur = _ReconcileCur(None, source_def=_rendered_kafka_source_def(topic="a'b", watermark_secs=45))
    assert _deployed_kafka_source_opts(cur, "proj_user_txn_src") == ("a'b", "localhost:9092", 45)


def test_deployed_kafka_source_opts_none_when_source_absent():
    # An absent source row (a never-provisioned view) reads back None, so source_changed stays False and the
    # MV-absence drives provisioning instead (never a spurious "changed" against a missing source).
    cur = _ReconcileCur(None, source_def=None)
    assert _deployed_kafka_source_opts(cur, "proj_user_txn_src") is None


def test_desired_kafka_source_opts_none_watermark_when_unset():
    # A source with no watermark yields None for the watermark slot, matching what the catalog reads back
    # (no INTERVAL in the DDL) — so a watermark-less source reconciles to a no-op, not a phantom change.
    view = _stream_view(_kafka_source(watermark=False), [_agg("sum")])
    assert _desired_kafka_source_opts(view.stream_source) == ("txn", "localhost:9092", None)


def test_reconcile_stream_view_noop_when_definition_unchanged():
    # A kept stream view whose deployed EOWC MV matches the desired SELECT must NOT be touched (no
    # rebuild, no serving blip) — the verbatim-catalog comparison sees no change.
    eng = _engine(emit_on_window_close=True)
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])
    desired = eng._stream_mv_select("proj", view)
    cur = _ReconcileCur(f"CREATE MATERIALIZED VIEW proj_user_txn_online AS {desired}")
    eng._reconcile_stream_view(cur, "proj", view)
    assert not any(s.startswith(("DROP", "CREATE")) for s in cur.executed)  # only the catalog SELECT ran


def test_reconcile_stream_view_reprovisions_on_definition_change():
    # A changed aggregation => the deployed MV SELECT differs => drop the graph (sink -> MV -> source,
    # dependents first) and re-provision (source -> MV -> sink). Without this the old MV would persist.
    eng = _engine(emit_on_window_close=True)
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])
    cur = _ReconcileCur("CREATE MATERIALIZED VIEW proj_user_txn_online AS SELECT stale")
    eng._reconcile_stream_view(cur, "proj", view)
    drops = [s for s in cur.executed if s.startswith("DROP")]
    creates = [s for s in cur.executed if s.startswith("CREATE")]
    assert [s.split()[1] for s in drops] == ["SINK", "MATERIALIZED", "SOURCE"]  # dependents first
    assert [s.split()[1] for s in creates] == ["SOURCE", "MATERIALIZED", "SINK"]  # base first


def test_reconcile_stream_view_noop_when_source_unchanged():
    # MV unchanged AND the deployed source — RE-RENDERED by RW (spaces around '=') — extracts to the SAME
    # (topic, bootstrap, watermark) as the view's KafkaSource => no spurious rebuild. This pins that the
    # spacing-tolerant extraction does not see the re-render as a change (a verbatim compare would).
    eng = _engine(emit_on_window_close=True)
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])
    desired_mv = eng._stream_mv_select("proj", view)
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_txn_online AS {desired_mv}",
        source_def=_rendered_kafka_source_def(),  # topic=txn, bootstrap=localhost:9092, watermark=30
    )
    eng._reconcile_stream_view(cur, "proj", view)
    assert not any(s.startswith(("DROP", "CREATE")) for s in cur.executed)  # only the catalog SELECTs ran


def test_reconcile_stream_view_reprovisions_on_topic_change():
    # MV unchanged but the source was repointed at a different topic — invisible in any MV SELECT, caught
    # from the source catalog. Drop the graph (sink -> MV -> source) + reprovision so serving stops reading
    # the stale topic.
    eng = _engine(emit_on_window_close=True)
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])
    desired_mv = eng._stream_mv_select("proj", view)
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_txn_online AS {desired_mv}",
        source_def=_rendered_kafka_source_def(topic="OLD_TOPIC"),
    )
    eng._reconcile_stream_view(cur, "proj", view)
    drops = [s for s in cur.executed if s.startswith("DROP")]
    creates = [s for s in cur.executed if s.startswith("CREATE")]
    assert [s.split()[1] for s in drops] == ["SINK", "MATERIALIZED", "SOURCE"]  # dependents first
    assert [s.split()[1] for s in creates] == ["SOURCE", "MATERIALIZED", "SINK"]  # base first


def test_reconcile_stream_view_reprovisions_on_watermark_change():
    # A changed watermark_delay_threshold shifts late-event admission (train/serve parity) but shows in NO
    # MV SELECT — only in the source's WATERMARK clause. Detect it from the source catalog and reprovision.
    eng = _engine(emit_on_window_close=True)
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])  # desired watermark = 30s
    desired_mv = eng._stream_mv_select("proj", view)
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_txn_online AS {desired_mv}",
        source_def=_rendered_kafka_source_def(watermark_secs=99),  # deployed 99s != desired 30s
    )
    eng._reconcile_stream_view(cur, "proj", view)
    assert any(s.startswith("DROP SOURCE") for s in cur.executed)
    assert any(s.startswith("CREATE SOURCE") for s in cur.executed)


# --- BATCH feature view provisioning (Iceberg source -> tiles MV) ---


def _batch_view(aggs, interval_secs=86400, name="user_txn_daily"):
    # A tile feature view = a plain FeatureView whose batch_source is an IcebergSource carrying the
    # tile spec (aggregations + interval). The engine reads it all off batch_source.
    return SimpleNamespace(
        name=name,
        batch_source=IcebergSource(
            table="txn_ice",
            timestamp_field="event_ts",
            aggregations=list(aggs),
            aggregation_interval=timedelta(seconds=interval_secs),
        ),
        entity_columns=[SimpleNamespace(name="user_id", dtype="String")],
        features=[SimpleNamespace(name=a.resolved_name(a.time_window)) for a in aggs],
        offline=True,
    )


def test_provision_batch_emits_source_tiles_mv_and_online_rollup_mv():
    # daily (86400s) tiles, 3-day (259200s) window. A NON-invertible aggregation (max) keeps the v1
    # per-(window) now()-anchored online rollup MV — the cumulative path is only for invertible aggs.
    agg = _agg("max", window_seconds=259200)
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
    assert "max(amount) AS max_amount" in tiles_sql  # window-independent partial (not per-window)

    # online rollup MV: the point-looked-up per-window name, plain now() window over the tiles MV
    assert rollup_sql.startswith("CREATE MATERIALIZED VIEW")
    assert '"proj_user_txn_daily_online_259200s"' in rollup_sql  # per-window MV name (window in suffix)
    assert "FROM proj_user_txn_daily_tiles" in rollup_sql  # reads FROM the tiles MV
    assert "now() - INTERVAL '259200' SECOND" in rollup_sql
    assert "tile_end <= now()" in rollup_sql
    assert "date_trunc" not in rollup_sql  # RW rejects two-sided date_trunc(now()) in an MV
    assert "max(tile_end) AS window_end" in rollup_sql  # PIT stamp for the point-lookup
    assert f"max(max_amount) AS {feat}" in rollup_sql  # rolls the window-independent partial up


def test_provision_batch_emits_one_online_mv_per_window():
    # multi-window: ONE tiles MV (shared, deduped partials) + one now()-anchored online MV PER window.
    # NON-invertible max keeps the v1 per-(window) MV mechanics (the cumulative path is invertible-only),
    # so this pins one MV per distinct window.
    aggs = [_agg("max", 259200), _agg("max", 2592000), _agg("max", 604800)]
    view = _batch_view(aggs)
    ddl = _engine()._provision_batch_ddl("proj", view)
    assert len(ddl) == 1 + 1 + 3  # source + tiles + 3 per-window online MVs (3d, 7d, 30d)
    joined = "\n".join(ddl)
    # exactly ONE tiles MV (window-independent partials shared across all windows)
    assert joined.count('"proj_user_txn_daily_tiles"') == 1
    # one online MV per DISTINCT window, named with the window suffix
    for w in (259200, 604800, 2592000):
        assert f'"proj_user_txn_daily_online_{w}s"' in joined
    # each per-window MV rolls the ONE shared max_amount tile partial up to its own window
    assert "max(max_amount) AS max_amount_2592000s" in joined
    assert "max(max_amount) AS max_amount_604800s" in joined


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


@pytest.mark.parametrize("function", ["approx_count_distinct"])
def test_provision_batch_rejects_non_additive_aggregation(function):
    view = _batch_view([_agg(function)])
    with pytest.raises(ValueError, match="not supported"):
        _engine()._provision_batch_ddl("proj", view)


def test_batch_drop_ddl_drops_both_mvs_and_source_with_no_sink():
    # NON-invertible max -> the v1 per-(window) online rollup MV (no cumulative MV), so the drop set is
    # the per-window MV, then the tiles MV, then the source.
    view = _batch_view([_agg("max", 259200)])
    stmts = _batch_drop_ddl("proj", view)
    # drop order: per-window online rollup MV(s), then the tiles MV they read, then the source
    assert stmts[0] == 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_online_259200s"'
    assert 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_tiles"' in stmts
    assert any(s.startswith("DROP SOURCE") for s in stmts)
    # the tiles MV is dropped AFTER the per-window online MV that reads it, and BEFORE the source
    assert stmts.index('DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_online_259200s"') < stmts.index(
        'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_tiles"'
    )
    assert not any("SINK" in s for s in stmts)  # none provisioned, none to drop


def test_existing_online_mv_names_matches_only_this_views_window_and_offset_mvs():
    # Catalog-driven reconcile: identify this tile view's per-(window, offset) online MVs (to drop orphans
    # when a re-apply shrinks the set) WITHOUT matching the tiles MV, another view, or a non-numeric tail.
    # Both the trailing-window form and the shifted ``_off{secs}s`` form must match.
    class _Cur:
        def execute(self, _sql):
            self._rows = [
                ("proj_v_online_259200s",),            # trailing-window match
                ("proj_v_online_2592000s",),           # trailing-window match
                ("proj_v_online_604800s_off604800s",), # shifted (offset) match
                ("proj_v_online_lifetime",),           # lifetime (no floor) match
                ("proj_v_online_lifetime_from1767225600s",),  # floored lifetime match
                ("proj_v_tiles",),                     # not an online MV
                ("proj_v_online_xs",),                 # non-numeric window -> ignore
                ("proj_other_online_5s",),             # different view -> ignore
            ]

        def fetchall(self):
            return self._rows

    assert _existing_online_mv_names(_Cur(), "proj", "v") == {
        "proj_v_online_259200s",
        "proj_v_online_2592000s",
        "proj_v_online_604800s_off604800s",
        "proj_v_online_lifetime",
        "proj_v_online_lifetime_from1767225600s",
    }


def test_deployed_mv_select_strips_create_prefix_and_handles_absent():
    # RW stores the MV definition VERBATIM as 'CREATE MATERIALIZED VIEW <name> AS <select>'; the
    # reconcile compares the <select> against the freshly-built one. Pin the prefix-strip + absent case.
    class _Cur:
        def __init__(self, row):
            self._row = row

        def execute(self, _sql, _params):
            pass

        def fetchone(self):
            return self._row

    deployed = _deployed_mv_select(
        _Cur(("CREATE MATERIALIZED VIEW v_online_3s AS SELECT a, sum(x) AS s FROM t",)), "v_online_3s"
    )
    assert deployed == "SELECT a, sum(x) AS s FROM t"  # keeps the inner ' AS ' aliases
    assert _deployed_mv_select(_Cur(None), "missing") is None


def test_deployed_source_table_extracts_iceberg_table_name():
    # A repointed Iceberg table only shows in the CREATE SOURCE ... table.name='...' definition; pin the
    # extraction (incl. the doubled-single-quote un-escaping) against the engine's own DDL format.
    class _Cur:
        def __init__(self, row):
            self._row = row

        def execute(self, _sql, _params):
            pass

        def fetchone(self):
            return self._row

    # RW RE-RENDERS a source's WITH clause in its catalog (spaces around '=', expanded types) — NOT
    # verbatim like an MV — so the parser must match that stored form, not the engine's emitted DDL.
    ddl = (
        "CREATE SOURCE proj_v_src WITH (connector = 'iceberg', catalog.name = 'feast', "
        "catalog.type = 'storage', warehouse.path = 's3a://feast/wh', database.name = 'feast', "
        "table.name = 'o''brien_txn')"  # embedded quote, doubled
    )
    assert _deployed_source_table(_Cur((ddl,)), "proj_v_src") == "o'brien_txn"
    assert _deployed_source_table(_Cur(None), "missing") is None  # absent source


def test_plan_batch_reconcile_noop_when_nothing_changed():
    full, drops, creates = _plan_batch_reconcile(
        desired_tiles="SELECT tiles", desired_online={"v_online_3s": "A"},
        deployed_tiles="SELECT tiles", deployed_online={"v_online_3s": "A"},
    )
    assert (full, drops, creates) == (False, [], [])  # no rebuild, no drop, no create -> no serving blip


def test_plan_batch_reconcile_adds_new_window_without_touching_tiles():
    # add a 30d window: tiles (window-independent) unchanged -> only CREATE the new online MV.
    full, drops, creates = _plan_batch_reconcile(
        desired_tiles="T", desired_online={"v_online_3s": "A", "v_online_30s": "B"},
        deployed_tiles="T", deployed_online={"v_online_3s": "A"},
    )
    assert full is False and drops == [] and creates == [("v_online_30s", "B")]


def test_plan_batch_reconcile_drops_removed_window():
    full, drops, creates = _plan_batch_reconcile(
        desired_tiles="T", desired_online={"v_online_3s": "A"},
        deployed_tiles="T", deployed_online={"v_online_3s": "A", "v_online_30s": "B"},
    )
    assert full is False and drops == ["v_online_30s"] and creates == []


def test_plan_batch_reconcile_replaces_redefined_window():
    full, drops, creates = _plan_batch_reconcile(
        desired_tiles="T", desired_online={"v_online_3s": "A2"},
        deployed_tiles="T", deployed_online={"v_online_3s": "A1"},
    )
    assert full is False and drops == ["v_online_3s"] and creates == [("v_online_3s", "A2")]


def test_plan_batch_reconcile_full_rebuild_when_tiles_partials_changed():
    # changing an aggregation function/column changes the tiles MV -> drop every online MV + rebuild.
    full, drops, creates = _plan_batch_reconcile(
        desired_tiles="SELECT max(x) AS max_x", desired_online={"v_online_3s": "A"},
        deployed_tiles="SELECT sum(x) AS sum_x", deployed_online={"v_online_3s": "A", "v_online_30s": "B"},
    )
    assert full is True and set(drops) == {"v_online_3s", "v_online_30s"} and creates == []


def test_plan_batch_reconcile_full_rebuild_on_first_provision():
    full, drops, creates = _plan_batch_reconcile(
        desired_tiles="T", desired_online={"v_online_3s": "A"},
        deployed_tiles=None, deployed_online={},  # nothing deployed yet
    )
    assert full is True and drops == []


def test_plan_batch_reconcile_ignores_whitespace_only_differences():
    full, drops, creates = _plan_batch_reconcile(
        desired_tiles="SELECT  a,   b", desired_online={"v_online_3s": "SELECT x"},
        deployed_tiles="SELECT a, b", deployed_online={"v_online_3s": "SELECT  x"},
    )
    assert (full, drops, creates) == (False, [], [])  # normalized comparison -> no spurious rebuild


def test_batch_drop_ddl_drops_every_per_window_mv():
    # a multi-window FV provisions N online MVs -> teardown must drop ALL N (else orphaned MVs leak).
    # NON-invertible max keeps the per-(window) MV mechanics, so every window has its own online MV.
    view = _batch_view([_agg("max", 259200), _agg("max", 2592000)])
    stmts = _batch_drop_ddl("proj", view)
    assert 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_online_259200s"' in stmts
    assert 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_online_2592000s"' in stmts
    assert 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_tiles"' in stmts
    # the tiles MV is dropped AFTER every per-window online MV that reads it (dependency order)
    tiles_idx = stmts.index('DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_tiles"')
    assert tiles_idx > stmts.index('DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_online_259200s"')
    assert tiles_idx > stmts.index('DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_online_2592000s"')


# --- STREAMING tile feature view provisioning (watermarked Kafka source -> EOWC tiles MV) ---


def test_provision_streaming_tile_secondary_key_declares_the_column_on_the_source():
    # A streaming-tile view with an aggregation secondary key references that raw column in the tiles MV
    # GROUP BY, but it is neither a join key nor an aggregation input — so the CREATE SOURCE must DECLARE
    # it, else RisingWave rejects the tiles MV with "column does not exist" at provision time.
    agg = _agg("sum", window_seconds=259200)
    view = _stream_tile_view(_kafka_source(watermark=True), [agg], interval_secs=86400)
    view.tags = {**(getattr(view, "tags", None) or {}), **encode_secondary_key("ad_id")}
    source_sql, tiles_sql, *_ = _engine()._provision_streaming_tile_ddl("proj", view)
    assert '"ad_id" VARCHAR' in source_sql  # declared on the source so the tiles MV can bind it
    assert 'GROUP BY window_start, window_end, user_id, "ad_id"' in tiles_sql


def test_provision_streaming_tile_declares_filter_column_and_shares_one_tile_scan():
    # total + DEBIT counts on the same column: the DEBIT one carries a FILTER predicate over a raw column
    # (transaction_code) that is neither a join key, an aggregation input, nor the timestamp. The CREATE
    # SOURCE must DECLARE it (typed from the carried source schema) so the tiles MV can bind the FILTER, and
    # both counts must materialize side by side in ONE tiles scan.
    total = _agg("count", window_seconds=259200)
    debit = Aggregation(column="amount", function="count",
                        time_window=timedelta(seconds=259200), name="debit_count")
    debit_rn = debit.resolved_name(debit.time_window)
    view = _stream_tile_view(_kafka_source(watermark=True), [total, debit], interval_secs=86400)
    view.tags = {
        **encode_agg_filters({debit_rn: "transaction_code = 'DEBIT'"}),
        **encode_agg_filter_cols({"transaction_code": "varchar"}),
    }
    source_sql, tiles_sql, *_ = _engine()._provision_streaming_tile_ddl("proj", view)
    assert '"transaction_code" varchar' in source_sql  # declared so the tiles MV FILTER can bind it
    # window-INDEPENDENT partials: total is the bare partial, DEBIT a FILTER partial — side by side, ONE scan
    assert "count(amount) AS count_amount" in tiles_sql
    assert "count(amount) FILTER (WHERE transaction_code = 'DEBIT') AS count_amount_f" in tiles_sql


def test_provision_streaming_tile_emits_watermarked_source_eowc_tiles_and_online_rollup():
    # daily (86400s) tiles, 3-day (259200s) window: a watermarked Kafka source, an EOWC TUMBLE tiles MV,
    # and the SAME now()-anchored online rollup MV as the batch path (only the tile source differs).
    # NON-invertible max keeps the v1 per-(window) online rollup MV (cumulative path is invertible-only).
    agg = _agg("max", window_seconds=259200)
    view = _stream_tile_view(_kafka_source(watermark=True), [agg], interval_secs=86400)
    ddl = _engine()._provision_streaming_tile_ddl("proj", view)
    assert len(ddl) == 3  # source + tiles MV + online rollup MV; NO Iceberg sink (MVs read directly)
    source_sql, tiles_sql, rollup_sql = ddl
    feat = agg.resolved_name(agg.time_window)

    # watermarked Kafka source (NOT iceberg) — the streaming tile source
    assert source_sql.startswith("CREATE SOURCE")
    assert "connector='kafka'" in source_sql
    assert "WATERMARK FOR" in source_sql

    # tiles MV: EOWC TUMBLE at the interval, window_end AS tile_end (vs the batch date_trunc GROUP BY)
    assert tiles_sql.startswith("CREATE MATERIALIZED VIEW")
    assert '"proj_user_txn_tiles"' in tiles_sql
    assert "tumble(" in tiles_sql
    assert "window_end AS tile_end" in tiles_sql
    assert tiles_sql.endswith("EMIT ON WINDOW CLOSE")
    assert "max(amount) AS max_amount" in tiles_sql  # window-independent partial (not per-window)

    # online rollup MV: byte-identical shape to the batch path (reads the tiles MV, now()-anchored)
    assert rollup_sql.startswith("CREATE MATERIALIZED VIEW")
    assert '"proj_user_txn_online_259200s"' in rollup_sql
    assert "FROM proj_user_txn_tiles" in rollup_sql
    assert "now() - INTERVAL '259200' SECOND" in rollup_sql
    assert "tile_end <= now()" in rollup_sql
    assert f"max(max_amount) AS {feat}" in rollup_sql

    # NO Iceberg sink (the streaming tile path mirrors the batch tile path — MVs read directly)
    assert not any("CREATE SINK" in s for s in ddl)


def test_provision_streaming_tile_emits_one_online_mv_per_window():
    # multi-window: ONE shared EOWC tiles MV (window-independent partials) + one now() online MV PER window.
    # NON-invertible max keeps the per-(window) MV mechanics (the cumulative path is invertible-only).
    aggs = [_agg("max", 259200), _agg("max", 2592000), _agg("max", 604800)]
    view = _stream_tile_view(_kafka_source(watermark=True), aggs, interval_secs=86400)
    ddl = _engine()._provision_streaming_tile_ddl("proj", view)
    assert len(ddl) == 1 + 1 + 3  # source + tiles + 3 per-window online MVs (3d, 7d, 30d)
    joined = "\n".join(ddl)
    assert joined.count('"proj_user_txn_tiles"') == 1  # exactly one tiles MV, shared across windows
    assert sum(1 for s in ddl if "tumble(" in s) == 1  # the single EOWC tiles MV
    for w in (259200, 604800, 2592000):
        assert f'"proj_user_txn_online_{w}s"' in joined
    # each per-window MV rolls the ONE shared max_amount tile partial up to its own window
    assert "max(max_amount) AS max_amount_2592000s" in joined
    assert "max(max_amount) AS max_amount_604800s" in joined


@pytest.mark.parametrize("emit_on_window_close", [True, False])
def test_provision_streaming_tile_always_requires_a_source_watermark(emit_on_window_close):
    # EOWC is INTRINSIC to the tile model (build_streaming_tile_select always EMIT ON WINDOW CLOSE), so —
    # unlike a plain stream MV whose EOWC is opt-in via emit_on_window_close — the watermark is required
    # regardless of the engine's emit_on_window_close config. Without it the EOWC tiles never emit.
    view = _stream_tile_view(_kafka_source(watermark=False), [_agg("sum", 259200)], interval_secs=86400)
    with pytest.raises(ValueError, match="watermark"):
        _engine(emit_on_window_close=emit_on_window_close)._provision_streaming_tile_ddl("proj", view)


def test_provision_streaming_tile_rejects_pushsource():
    # isinstance check fires before any attribute access, so __new__ is enough.
    push = PushSource.__new__(PushSource)
    view = _stream_tile_view(push, [_agg("sum", 259200)], interval_secs=86400)
    with pytest.raises(ValueError, match="PushSource"):
        _engine()._provision_streaming_tile_ddl("proj", view)


def test_provision_streaming_tile_requires_a_tiling_hop_size():
    # enable_tiling without a tiling_hop_size -> is_streaming_tile is True but tile_interval is None.
    # Guard with the actionable fix instead of an opaque 'NoneType' AttributeError in the SQL builder.
    view = _stream_tile_view(_kafka_source(watermark=True), [_agg("sum", 259200)], interval_secs=86400)
    view.tiling_hop_size = None  # an enable_tiling SFV authored without the hop
    with pytest.raises(ValueError, match="tiling_hop_size"):
        _engine()._provision_streaming_tile_ddl("proj", view)


def test_is_tile_view_unions_batch_and_streaming_tile():
    # offline + serving routing keys on is_tile_view: BOTH a batch tile FV and a streaming tile view
    # take the tile path (per-window rollup MVs online + tile PIT offline); a plain stream view does not.
    assert is_tile_view(_batch_view([_agg("sum", 259200)]))  # batch tile (IcebergSource)
    assert is_tile_view(
        _stream_tile_view(_kafka_source(watermark=True), [_agg("sum", 259200)], interval_secs=86400)
    )  # streaming tile (enable_tiling)
    assert not is_tile_view(_stream_view(_kafka_source(watermark=True), [_agg("sum", 259200)]))  # plain stream


# --- STREAMING tile reconcile (verbatim-catalog comparison drives re-materialize) ---


class _TileReconcileCur:
    """Fake cursor for the tile reconcile: answers ``pg_matviews`` (the existing online MV names, for
    ``_existing_online_window_secs``) and the ``rw_catalog`` MV-definition lookup (for
    ``_deployed_mv_select``), and records executed DDL. ``deployed`` maps an MV name -> the
    ``CREATE MATERIALIZED VIEW <name> AS <select>`` definition RisingWave stores verbatim; an absent name
    => the MV does not exist. Also answers the rw_sources source-definition lookup (for
    ``_deployed_kafka_source_opts``) with ``source_def``; source_def=None => the source row is absent."""

    def __init__(self, deployed, source_def=None, source_cols=None):
        self.executed = []
        self._deployed = deployed
        self._source_def = source_def
        self._source_cols = source_cols  # information_schema.columns rows for the source (filter-col diff)
        self._name = None
        self._is_source = False
        self._cols_query = False

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self._cols_query = "information_schema.columns" in sql
        if sql.startswith("SELECT definition"):
            self._is_source = "rw_sources" in sql
            self._name = params[0]

    def fetchall(self):
        if self._cols_query:  # _deployed_source_columns: the source's (column, canonical type) rows
            return list(self._source_cols) if self._source_cols else []
        return [(name,) for name in self._deployed]  # all deployed MV names (tiles + per-window online)

    def fetchone(self):
        if self._is_source:
            return (self._source_def,) if self._source_def is not None else None
        defn = self._deployed.get(self._name)
        return (defn,) if defn is not None else None


def _deployed_from_provision_ddl(ddl):
    # Turn the engine's provision DDL into the {name: stored-definition} the catalog would hold: RW stores
    # each 'CREATE MATERIALIZED VIEW IF NOT EXISTS "<name>" AS <select>' as 'CREATE MATERIALIZED VIEW
    # <name> AS <select>' (verbatim modulo whitespace, on RisingWave v3.0.0). The MV-name ' AS ' precedes any
    # in-SELECT ' AS ', so split(" AS ", 1) lands on it.
    deployed = {}
    for stmt in ddl:
        if not stmt.startswith("CREATE MATERIALIZED VIEW"):
            continue  # skip the CREATE SOURCE
        head, select = stmt.split(" AS ", 1)
        name = head.split('"')[1]
        deployed[name] = f"CREATE MATERIALIZED VIEW {name} AS {select}"
    return deployed


def _reconcilable_streaming_tile_view():
    # NON-invertible max keeps the v1 per-(window) online rollup MV mechanics (the cumulative path is for
    # invertible aggs only), so the reconcile add/rebuild/drop checks below stay about per-window MVs.
    return _stream_tile_view(
        _kafka_source(watermark=True), [_agg("max", 259200), _agg("max", 2592000)], interval_secs=86400
    )


def test_reconcile_streaming_tile_noop_when_unchanged():
    # A kept streaming-tile view whose deployed tiles + per-window MVs match the desired SELECTs must NOT
    # be touched (no rebuild, no serving blip) — the verbatim-catalog comparison sees no change. This is
    # the load-bearing check that RW stores the EOWC TUMBLE tiles MV the way we generate it.
    eng = _engine()
    view = _reconcilable_streaming_tile_view()
    deployed = _deployed_from_provision_ddl(eng._provision_streaming_tile_ddl("proj", view))
    cur = _TileReconcileCur(deployed)
    eng._reconcile_streaming_tile_view(cur, "proj", view)
    assert not any(s.startswith(("DROP", "CREATE")) for s in cur.executed)  # only catalog SELECTs ran


def test_reconcile_streaming_tile_full_rebuild_when_tiles_changed():
    # A changed per-tile partial => the deployed tiles SELECT differs => drop the online MVs + the tiles
    # MV + the SOURCE, then re-provision the whole graph. The source is dropped (unlike the batch
    # reconcile) because a streaming source lists its agg-input columns explicitly, so a new aggregation
    # input column changes the source schema — CREATE SOURCE IF NOT EXISTS alone would keep the old cols.
    eng = _engine()
    view = _reconcilable_streaming_tile_view()
    deployed = _deployed_from_provision_ddl(eng._provision_streaming_tile_ddl("proj", view))
    deployed["proj_user_txn_tiles"] = "CREATE MATERIALIZED VIEW proj_user_txn_tiles AS SELECT stale"
    cur = _TileReconcileCur(deployed)
    eng._reconcile_streaming_tile_view(cur, "proj", view)
    assert 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_tiles"' in cur.executed
    assert 'DROP SOURCE IF EXISTS "proj_user_txn_src"' in cur.executed  # so CREATE SOURCE picks up new cols
    # the tiles MV is dropped BEFORE the source it reads (dependency order)
    assert cur.executed.index('DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_tiles"') < cur.executed.index(
        'DROP SOURCE IF EXISTS "proj_user_txn_src"'
    )
    creates = [s for s in cur.executed if s.startswith("CREATE MATERIALIZED VIEW")]
    assert len(creates) == 3  # tiles + both per-window online MVs re-created
    assert sum(1 for s in creates if "tumble(" in s) == 1  # the EOWC tiles MV is rebuilt
    assert any(s.startswith("CREATE SOURCE") for s in cur.executed)  # the source is re-created


def test_reconcile_streaming_tile_adds_only_the_new_window_mv():
    # A widened window set (tiles unchanged — partials are window-independent) creates ONLY the new
    # window's online MV; the tiles MV and the existing window MV keep running untouched.
    eng = _engine()
    view = _reconcilable_streaming_tile_view()
    deployed = _deployed_from_provision_ddl(eng._provision_streaming_tile_ddl("proj", view))
    del deployed["proj_user_txn_online_2592000s"]  # the 30d window MV isn't deployed yet
    cur = _TileReconcileCur(deployed)
    eng._reconcile_streaming_tile_view(cur, "proj", view)
    creates = [s for s in cur.executed if s.startswith("CREATE MATERIALIZED VIEW")]
    assert len(creates) == 1 and "proj_user_txn_online_2592000s" in creates[0]  # only the new window
    assert "tumble(" not in creates[0]  # NOT a tiles rebuild
    assert not any(s.startswith("DROP") for s in cur.executed)  # nothing removed


def test_reconcile_streaming_tile_noop_when_source_unchanged():
    # Tiles + per-window MVs unchanged AND the deployed source (RE-RENDERED by RW) extracts to the same opts
    # as the view's KafkaSource => no rebuild. Pins that the source readback does not over-reconcile.
    eng = _engine()
    view = _reconcilable_streaming_tile_view()  # _kafka_source: topic=txn, bootstrap=localhost:9092, wm=30
    deployed = _deployed_from_provision_ddl(eng._provision_streaming_tile_ddl("proj", view))
    cur = _TileReconcileCur(deployed, source_def=_rendered_kafka_source_def())
    eng._reconcile_streaming_tile_view(cur, "proj", view)
    assert not any(s.startswith(("DROP", "CREATE")) for s in cur.executed)  # only catalog SELECTs ran


def test_reconcile_streaming_tile_full_rebuild_when_source_changed():
    # Tiles + per-window MVs unchanged, but the source was repointed (topic) — caught from the source
    # catalog (it shows in no MV SELECT). Force a full re-materialize: drop all online MVs + the tiles MV +
    # the source, then reprovision, so the tiles stop reading the stale topic.
    eng = _engine()
    view = _reconcilable_streaming_tile_view()
    deployed = _deployed_from_provision_ddl(eng._provision_streaming_tile_ddl("proj", view))
    cur = _TileReconcileCur(deployed, source_def=_rendered_kafka_source_def(topic="OLD_TOPIC"))
    eng._reconcile_streaming_tile_view(cur, "proj", view)
    assert 'DROP SOURCE IF EXISTS "proj_user_txn_src"' in cur.executed
    assert 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_tiles"' in cur.executed
    drops_online = [
        s for s in cur.executed if s.startswith("DROP MATERIALIZED VIEW") and "_online_" in s
    ]
    assert len(drops_online) == 2  # both per-window online MVs dropped (full rebuild, not granular)
    creates = [s for s in cur.executed if s.startswith("CREATE MATERIALIZED VIEW")]
    assert len(creates) == 3  # tiles + both per-window online MVs re-created
    assert any(s.startswith("CREATE SOURCE") for s in cur.executed)  # the source is re-created


def _filtered_streaming_tile_view():
    # total + DEBIT counts (both invertible -> cumulative MV); the DEBIT filter references transaction_code,
    # a raw column declared on the CREATE SOURCE from the filter-cols carrier (typed varchar).
    debit = Aggregation(
        column="amount", function="count", time_window=timedelta(seconds=259200), name="debit_count"
    )
    view = _stream_tile_view(_kafka_source(watermark=True), [_agg("count", 259200), debit], interval_secs=86400)
    debit_rn = debit.resolved_name(debit.time_window)
    view.tags = {
        **encode_agg_filters({debit_rn: "transaction_code = 'DEBIT'"}),
        **encode_agg_filter_cols({"transaction_code": "varchar"}),
    }
    return view


def test_reconcile_streaming_tile_rebuilds_on_filter_column_type_change():
    # A type-only change to a FILTER-referenced source column shows in NO MV SELECT (the predicate references
    # the column by name, not type), so the tiles/online MVs are unchanged; it is caught from the source
    # catalog (information_schema) and forces a full rebuild so the live CREATE SOURCE picks up the new type.
    eng = _engine()
    view = _filtered_streaming_tile_view()
    deployed = _deployed_from_provision_ddl(eng._provision_streaming_tile_ddl("proj", view))
    # the deployed source declares transaction_code as a STALE type (bigint) vs the desired varchar
    stale_cols = [("transaction_code", "bigint")]
    cur = _TileReconcileCur(deployed, source_def=_rendered_kafka_source_def(), source_cols=stale_cols)
    eng._reconcile_streaming_tile_view(cur, "proj", view)
    assert 'DROP SOURCE IF EXISTS "proj_user_txn_src"' in cur.executed  # rebuilt to pick up the new type
    assert any(s.startswith("CREATE SOURCE") for s in cur.executed)


def test_reconcile_streaming_tile_noop_when_filter_column_type_matches():
    # Same filtered view; the deployed source reports transaction_code in its canonical form (varchar ->
    # "character varying"), matching the desired -> NO spurious rebuild (the placeholder-vs-carrier mismatch
    # trap is avoided by scoping the diff to the carrier-declared filter columns only).
    eng = _engine()
    view = _filtered_streaming_tile_view()
    deployed = _deployed_from_provision_ddl(eng._provision_streaming_tile_ddl("proj", view))
    matching_cols = [("transaction_code", "character varying")]
    cur = _TileReconcileCur(deployed, source_def=_rendered_kafka_source_def(), source_cols=matching_cols)
    eng._reconcile_streaming_tile_view(cur, "proj", view)
    assert not any(s.startswith(("DROP", "CREATE")) for s in cur.executed)  # only catalog SELECTs ran


def test_multi_window_streaming_tile_would_fail_the_plain_stream_path():
    # Routing-precedence rationale + intermediate-state fail-safe: a streaming-tile view is BOTH a
    # StreamFeatureView AND tile-like, so update()/teardown check is_streaming_tile FIRST. The plain
    # stream path (build_windowed_agg_select, one EOWC MV) REJECTS multi-window aggregations, so a
    # streaming-tile view mis-routed there fails LOUDLY (never silently mis-provisions) — and the tile
    # path is the only one that can provision it.
    # NON-invertible max keeps the per-(window) MV mechanics, so the multi-window view provisions one
    # online MV per window (the cumulative path is invertible-only and would collapse them to one MV).
    aggs = [_agg("max", 259200), _agg("max", 2592000)]
    view = _stream_tile_view(_kafka_source(watermark=True), aggs, interval_secs=86400)
    with pytest.raises(ValueError):  # the plain stream path can't build multi-window
        _engine()._stream_mv_select("proj", view)
    # ... but the streaming-tile path provisions one tiles MV + one online MV per window.
    ddl = _engine()._provision_streaming_tile_ddl("proj", view)
    assert sum(1 for s in ddl if "tumble(" in s) == 1
    assert len([s for s in ddl if "_online_" in s]) == 2


# --- passthrough (Attribute) provisioning + reconcile: a latest-row MV, no aggregation ---


def _passthrough_stream_view(source, feature_cols=("amount", "country"), name="user_attr"):
    # A streaming passthrough view: a StreamFeatureView-shaped view with raw feature columns and NO
    # aggregations (the Attribute flavor). The engine serves it as the latest row per entity.
    return SimpleNamespace(
        name=name,
        stream_source=source,
        aggregations=[],
        entity_columns=[SimpleNamespace(name="user_id", dtype="String")],
        features=[SimpleNamespace(name=c, dtype="Float64") for c in feature_cols],
        offline=True,
    )


def _passthrough_batch_view(feature_cols=("amount",), name="user_attr_daily"):
    # A batch passthrough view: a plain FeatureView whose batch_source is an IcebergSource with NO tile
    # aggregation spec — raw columns served as the latest row per entity.
    return SimpleNamespace(
        name=name,
        batch_source=IcebergSource(table="attr_ice", timestamp_field="event_ts"),
        entity_columns=[SimpleNamespace(name="user_id", dtype="String")],
        features=[SimpleNamespace(name=c, dtype="Float64") for c in feature_cols],
        offline=True,
    )


def test_passthrough_discriminators_select_only_no_aggregation_views():
    ps = _passthrough_stream_view(_kafka_source(watermark=False))
    pb = _passthrough_batch_view()
    assert is_passthrough_stream(ps) and is_passthrough_view(ps) and not is_passthrough_fv(ps)
    assert is_passthrough_fv(pb) and is_passthrough_view(pb) and not is_passthrough_stream(pb)
    # NOT passthrough: an aggregating stream, a streaming tile, and a batch tile all have aggregations.
    assert not is_passthrough_view(_stream_view(_kafka_source(watermark=True), [_agg("sum")]))
    assert not is_passthrough_view(
        _stream_tile_view(_kafka_source(watermark=True), [_agg("sum", 259200)], interval_secs=86400)
    )
    assert not is_passthrough_view(_batch_view([_agg("sum", 259200)]))


def test_provision_passthrough_stream_emits_source_with_raw_cols_and_latest_row_mv():
    eng = _engine()
    view = _passthrough_stream_view(_kafka_source(watermark=False))
    ddl = eng._provision_passthrough_ddl("proj", view)
    assert len(ddl) == 2  # source + latest-row MV; NO Iceberg sink on the online path
    source_sql, mv_sql = ddl
    assert source_sql.startswith("CREATE SOURCE") and "connector='kafka'" in source_sql
    assert '"amount"' in source_sql and '"country"' in source_sql  # raw feature columns declared
    assert "WATERMARK" not in source_sql  # a latest-row Group-TopN needs no watermark
    assert mv_sql.startswith("CREATE MATERIALIZED VIEW")
    assert "ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_ts DESC)" in mv_sql
    assert "tumble(" not in mv_sql and "GROUP BY" not in mv_sql  # passthrough: no window, no aggregation


def test_provision_passthrough_batch_emits_iceberg_source_and_latest_row_mv():
    eng = _engine()
    view = _passthrough_batch_view()
    ddl = eng._provision_passthrough_ddl("proj", view)
    assert len(ddl) == 2
    source_sql, mv_sql = ddl
    assert source_sql.startswith("CREATE SOURCE") and "connector='iceberg'" in source_sql
    assert mv_sql.startswith("CREATE MATERIALIZED VIEW")
    assert "ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_ts DESC)" in mv_sql


def _passthrough_source_cols(amount_type="double precision"):
    # The (name, canonical type) rows information_schema.columns reports for the deployed passthrough source
    # of _passthrough_stream_view (user_id String, amount/country Float64, event_ts ts).
    return [
        ("user_id", "character varying"),
        ("amount", amount_type),
        ("country", "double precision"),
        ("event_ts", "timestamp without time zone"),
    ]


def test_passthrough_drop_ddl_drops_sink_mv_then_source():
    # Dependents first; the SINK is dropped too because the online MV name is shared with the plain-stream
    # shape — a shape migration would otherwise orphan that shape's sink and block the MV drop.
    ddl = _passthrough_drop_ddl("proj", _passthrough_stream_view(_kafka_source(watermark=False)))
    assert ddl == [
        'DROP SINK IF EXISTS "proj_user_attr_offline"',
        'DROP MATERIALIZED VIEW IF EXISTS "proj_user_attr_online"',
        'DROP SOURCE IF EXISTS "proj_user_attr_src"',
        'DROP SOURCE IF EXISTS "proj_user_attr_history"',  # the streaming offline-history Iceberg source
    ]


def test_reconcile_passthrough_noop_when_unchanged():
    # Deployed latest-row MV matches the desired SELECT and the source opts (topic, bootstrap) match -> no
    # rebuild. The re-rendered source carries a watermark, but passthrough ignores it (Group-TopN), so it
    # must NOT trigger a spurious change.
    eng = _engine()
    view = _passthrough_stream_view(_kafka_source(watermark=False))
    desired = eng._passthrough_mv_select("proj", view)
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_attr_online AS {desired}",
        source_def=_rendered_kafka_source_def(),  # topic=txn, bootstrap=localhost:9092 (watermark ignored)
        source_cols=_passthrough_source_cols(),  # deployed column types match the desired schema
    )
    eng._reconcile_passthrough_view(cur, "proj", view)
    assert not any(s.startswith(("DROP", "CREATE")) for s in cur.executed)


def test_reconcile_passthrough_reprovisions_on_mv_change():
    eng = _engine()
    view = _passthrough_stream_view(_kafka_source(watermark=False))
    cur = _ReconcileCur(
        "CREATE MATERIALIZED VIEW proj_user_attr_online AS SELECT stale",
        source_def=_rendered_kafka_source_def(),
    )
    eng._reconcile_passthrough_view(cur, "proj", view)
    drops = [s for s in cur.executed if s.startswith("DROP")]
    creates = [s for s in cur.executed if s.startswith("CREATE")]
    # sink, latest-row MV, online source, offline-history source (the last a no-op IF EXISTS)
    assert [s.split()[1] for s in drops] == ["SINK", "MATERIALIZED", "SOURCE", "SOURCE"]
    assert [s.split()[1] for s in creates] == ["SOURCE", "MATERIALIZED"]  # source then MV


def test_reconcile_passthrough_stream_reprovisions_on_topic_change():
    # MV unchanged but the Kafka source was repointed at a different topic -> caught from the catalog.
    eng = _engine()
    view = _passthrough_stream_view(_kafka_source(watermark=False))
    desired = eng._passthrough_mv_select("proj", view)
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_attr_online AS {desired}",
        source_def=_rendered_kafka_source_def(topic="OTHER_TOPIC"),
    )
    eng._reconcile_passthrough_view(cur, "proj", view)
    assert any(s.startswith("DROP SOURCE") for s in cur.executed)
    assert any(s.startswith("CREATE SOURCE") for s in cur.executed)


def test_reconcile_passthrough_stream_reprovisions_on_feature_dtype_change():
    # A passthrough feature dtype change keeps the same column name, so the latest-row MV SELECT is unchanged
    # and the topic/bootstrap are unchanged — but the Kafka source declares the column with the OLD type.
    # Detected from information_schema (the column schema changed) -> rebuild source + MV, else the source
    # parses the field under the wrong type and serves a stale schema.
    eng = _engine()
    view = _passthrough_stream_view(_kafka_source(watermark=False))  # amount is Float64 -> double precision
    desired = eng._passthrough_mv_select("proj", view)
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_attr_online AS {desired}",
        source_def=_rendered_kafka_source_def(),  # topic/bootstrap unchanged
        source_cols=_passthrough_source_cols(amount_type="character varying"),  # deployed amount is VARCHAR
    )
    eng._reconcile_passthrough_view(cur, "proj", view)
    assert any(s.startswith("DROP SOURCE") for s in cur.executed)
    assert any(s.startswith("CREATE SOURCE") for s in cur.executed)


def test_reconcile_passthrough_batch_reprovisions_on_table_repoint():
    # A repointed Iceberg table shows in no MV SELECT (the MV reads the source by name); detect it from the
    # source catalog and re-provision.
    eng = _engine()
    view = _passthrough_batch_view()  # batch_source.table == "attr_ice"
    desired = eng._passthrough_mv_select("proj", view)
    stale_source = (
        'CREATE SOURCE "proj_user_attr_daily_src" WITH (connector = \'iceberg\', '
        "table.name = 'OLD_TABLE')"
    )
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_attr_daily_online AS {desired}",
        source_def=stale_source,
    )
    eng._reconcile_passthrough_view(cur, "proj", view)
    assert any(s.startswith("DROP SOURCE") for s in cur.executed)
    assert any(s.startswith("CREATE SOURCE") for s in cur.executed)


def _passthrough_stream_view_with_history(table="attr_hist"):
    # A streaming passthrough whose Kafka source carries an Iceberg batch_source — the historical log
    # backing the stream (the dual-source pattern), which the offline as-of read uses.
    src = _kafka_source(watermark=False)
    src.batch_source = IcebergSource(name="attr_hist", table=table, timestamp_field="event_ts")
    return _passthrough_stream_view(src)


def test_passthrough_pit_query_is_asof_latest_per_entity():
    # Offline read: per entity row, the latest raw row at-or-before its label ts (within ttl), over the raw
    # history — a LEFT JOIN + ROW_NUMBER rn=1, NOT an aggregation. This is the as-of cut that makes offline
    # == the latest-row online MV.
    ci = ColumnInfo(
        join_keys=["user_id"], feature_cols=["amount"], ts_col="event_ts",
        created_ts_col=None, field_mapping=None,
    )
    sql = build_passthrough_pit_query(
        "SELECT 1", ["user_id", "label_ts"], "label_ts",
        history_relation="hist", column_info=ci, ttl_seconds=3600,
    )
    assert "LEFT JOIN hist h ON" in sql
    assert 'h."event_ts" <= e."label_ts"' in sql  # as-of: at-or-before the label
    assert '>= e."label_ts" - INTERVAL \'3600\' SECOND' in sql  # ttl lower bound, INCLUSIVE
    assert "ROW_NUMBER() OVER (PARTITION BY" in sql and sql.rstrip().endswith("rn = 1")
    assert "GROUP BY" not in sql  # latest-row pick, not an aggregation


def test_passthrough_pit_query_rejects_feature_colliding_with_an_entity_column():
    # A feature named like a non-join-key entity_df column (e.g. the label-ts column) is ambiguous: the
    # as-of read cannot return both the feature and the entity column under one name. Fail clearly instead
    # of silently shadowing the feature with the entity column (and dropping its full_feature_names alias).
    ci = ColumnInfo(
        join_keys=["user_id"], feature_cols=["event_ts", "amount"], ts_col="event_ts",
        created_ts_col=None, field_mapping=None,
    )
    with pytest.raises(ValueError, match="collide with an entity-dataframe column"):
        build_passthrough_pit_query(
            "SELECT 1", ["user_id", "event_ts"], "event_ts", history_relation="hist", column_info=ci,
        )
    # a non-colliding feature set is unaffected
    ci2 = ColumnInfo(
        join_keys=["user_id"], feature_cols=["amount"], ts_col="event_ts",
        created_ts_col=None, field_mapping=None,
    )
    sql = build_passthrough_pit_query(
        "SELECT 1", ["user_id", "event_ts"], "event_ts", history_relation="hist", column_info=ci2,
    )
    assert 'h."amount"' in sql


def test_offline_passthrough_streaming_without_iceberg_history_is_rejected_clearly():
    # A streaming passthrough whose Kafka source has no Iceberg batch_source has no offline history (the
    # latest-row MV holds only the current row), so training fails CLEARLY rather than silently reading the
    # inert placeholder offline source.
    view = _passthrough_stream_view(_kafka_source(watermark=False))  # no batch_source
    entity_df = pd.DataFrame({"user_id": ["A"], "event_timestamp": [pd.Timestamp("2026-01-01")]})
    with pytest.raises(NotImplementedError, match="Iceberg batch source|online only"):
        RisingWaveOfflineStore.get_historical_features(
            config=None, feature_views=[view], feature_refs=[], entity_df=entity_df,
            registry=None, project="proj",
        )


def test_provision_passthrough_stream_with_iceberg_history_adds_history_source():
    # When the stream's Kafka source declares an Iceberg batch_source, provisioning ALSO creates an Iceberg
    # source over it (the offline history) — Kafka source + latest-row MV + history Iceberg source.
    eng = _engine()
    view = _passthrough_stream_view_with_history()
    ddl = eng._provision_passthrough_ddl("proj", view)
    assert len(ddl) == 3
    assert ddl[0].startswith("CREATE SOURCE") and "connector='kafka'" in ddl[0]
    assert ddl[1].startswith("CREATE MATERIALIZED VIEW")
    assert ddl[2].startswith('CREATE SOURCE IF NOT EXISTS "proj_user_attr_history"')
    assert "connector='iceberg'" in ddl[2]


def test_reconcile_passthrough_stream_reprovisions_on_history_table_repoint():
    # A repointed offline-history Iceberg table shows in no MV SELECT and no Kafka opt; detect it from the
    # history source catalog and rebuild (so offline training reads the new table).
    eng = _engine()
    view = _passthrough_stream_view_with_history(table="attr_hist")  # desired history table
    desired = eng._passthrough_mv_select("proj", view)
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_attr_online AS {desired}",
        source_def=_rendered_kafka_source_def(),  # topic/bootstrap unchanged
        source_cols=_passthrough_source_cols(),  # column types unchanged
        history_def='CREATE SOURCE "proj_user_attr_history" WITH (connector = \'iceberg\', '
        "table.name = 'OLD_TABLE')",  # deployed history points at a different table
    )
    eng._reconcile_passthrough_view(cur, "proj", view)
    assert any(s.startswith("CREATE SOURCE") and "connector='iceberg'" in s for s in cur.executed)
    assert 'DROP SOURCE IF EXISTS "proj_user_attr_history"' in cur.executed


def test_reconcile_passthrough_stream_provisions_missing_history_source():
    # An Iceberg batch_source ADDED to a previously online-only streaming passthrough: the history source
    # does not exist yet (deployed_history is None), so reconcile must rebuild and create it — else offline
    # training would read a non-existent relation.
    eng = _engine()
    view = _passthrough_stream_view_with_history(table="attr_hist")
    desired = eng._passthrough_mv_select("proj", view)
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_attr_online AS {desired}",
        source_def=_rendered_kafka_source_def(),  # topic/bootstrap unchanged
        source_cols=_passthrough_source_cols(),  # column types unchanged
        history_def=None,  # the offline-history Iceberg source is not yet provisioned
    )
    eng._reconcile_passthrough_view(cur, "proj", view)
    assert any(
        s.startswith('CREATE SOURCE IF NOT EXISTS "proj_user_attr_history"') for s in cur.executed
    )


# --- passthrough offline over a PostgreSQL batch_source (read directly over pgwire, not provisioned) ---


def _passthrough_stream_view_with_pg_history(table="attr_hist_pg", query=None):
    # A streaming passthrough whose Kafka source carries a PostgreSQL batch_source. RisingWave reads a
    # Postgres relation directly over pgwire, so the offline as-of read queries it in place — no provisioned
    # Iceberg history source. A projection is set because the offline read resolves the view name.
    src = _kafka_source(watermark=False)
    src.batch_source = PostgreSQLSource(
        name="attr_hist_pg", table=table, query=query, timestamp_field="event_ts"
    )
    view = _passthrough_stream_view(src)
    view.projection = SimpleNamespace(name_to_use=lambda: view.name)
    return view


def test_offline_passthrough_streaming_reads_postgres_table_batch_source_directly():
    # A streaming passthrough whose Kafka source declares a PostgreSQL batch_source: the as-of read queries
    # that PG table in place as the history relation — NOT a provisioned {base}_history Iceberg source.
    from unittest.mock import patch

    view = _passthrough_stream_view_with_pg_history(table="attr_hist_pg")
    entity_df = pd.DataFrame({"user_id": ["A"], "event_timestamp": [pd.Timestamp("2026-01-01")]})
    registry = MagicMock()
    registry.list_on_demand_feature_views.return_value = []
    target = "feast.infra.compute_engines.risingwave.offline_store.PostgreSQLRetrievalJob"
    with patch(target) as job:
        RisingWaveOfflineStore.get_historical_features(
            config=MagicMock(), feature_views=[view], feature_refs=["user_attr:amount"],
            entity_df=entity_df, registry=registry, project="proj",
        )
    sql = job.call_args.kwargs["query"]
    assert "LEFT JOIN attr_hist_pg h ON" in sql  # the PG table queried directly as the history relation
    assert "proj_user_attr_history" not in sql  # no provisioned Iceberg history source is referenced
    assert 'h."event_ts" <= e."event_timestamp"' in sql  # as-of cut on the PG source's timestamp_field
    assert "ROW_NUMBER() OVER (PARTITION BY" in sql and sql.rstrip().endswith("rn = 1")
    assert 'h."amount"' in sql  # the requested feature projected from the PG history side


def test_offline_passthrough_streaming_reads_postgres_query_batch_source_as_subquery():
    # A query-based PostgreSQL batch_source: get_table_query_string() parenthesizes the query, which the PIT
    # builder aliases as the history relation (LEFT JOIN (subquery) h) — valid over pgwire.
    from unittest.mock import patch

    view = _passthrough_stream_view_with_pg_history(table=None, query="SELECT * FROM raw_attr")
    entity_df = pd.DataFrame({"user_id": ["A"], "event_timestamp": [pd.Timestamp("2026-01-01")]})
    registry = MagicMock()
    registry.list_on_demand_feature_views.return_value = []
    target = "feast.infra.compute_engines.risingwave.offline_store.PostgreSQLRetrievalJob"
    with patch(target) as job:
        RisingWaveOfflineStore.get_historical_features(
            config=MagicMock(), feature_views=[view], feature_refs=["user_attr:amount"],
            entity_df=entity_df, registry=registry, project="proj",
        )
    sql = job.call_args.kwargs["query"]
    assert "LEFT JOIN (SELECT * FROM raw_attr) h ON" in sql


def test_provision_passthrough_stream_with_postgres_history_adds_no_history_source():
    # A PostgreSQL batch_source is read directly over pgwire at training time, so provisioning emits only
    # the Kafka source + latest-row MV — NO {base}_history source (there is nothing to provision).
    eng = _engine()
    view = _passthrough_stream_view_with_pg_history(table="attr_hist_pg")
    ddl = eng._provision_passthrough_ddl("proj", view)
    assert len(ddl) == 2
    assert ddl[0].startswith("CREATE SOURCE") and "connector='kafka'" in ddl[0]
    assert ddl[1].startswith("CREATE MATERIALIZED VIEW")
    assert not any("_history" in s for s in ddl)


def test_reconcile_passthrough_stream_with_postgres_history_is_noop_when_unchanged():
    # A PostgreSQL batch_source provisions no history source, so the history-table-repoint check (which keys
    # on an IcebergSource batch_source) naturally skips it — a kept, unchanged PG-history passthrough must
    # NOT spuriously rebuild.
    eng = _engine()
    view = _passthrough_stream_view_with_pg_history(table="attr_hist_pg")
    desired = eng._passthrough_mv_select("proj", view)
    cur = _ReconcileCur(
        f"CREATE MATERIALIZED VIEW proj_user_attr_online AS {desired}",
        source_def=_rendered_kafka_source_def(),  # topic/bootstrap unchanged
        source_cols=_passthrough_source_cols(),  # column types unchanged
    )
    eng._reconcile_passthrough_view(cur, "proj", view)
    assert not any(s.startswith(("DROP", "CREATE")) for s in cur.executed)


# --- offline materialize routing: tile views no-op (offline reads the live tiles MV directly) ---


def test_materialize_one_tile_view_offline_is_noop(monkeypatch):
    # A tile view (batch OR streaming) trains offline by reading the live tiles MV directly, so there is no
    # durable offline table to backfill. materialize(offline=True) must be a no-op — NOT run the plain
    # windowed-agg backfill, which would skew offline from the tile rollup served online (and a batch tile
    # view has no stream_source, so the backfill path would even raise). Proven by: the backfill builder is
    # never constructed (we make it explode) yet the job SUCCEEDS.
    import feast.infra.compute_engines.risingwave.engine as eng_mod

    def _explode(*args, **kwargs):
        raise RuntimeError("entered the windowed-agg backfill path")

    monkeypatch.setattr(eng_mod, "RisingWaveFeatureBuilder", _explode)
    eng = _engine()
    start, end = datetime(2026, 1, 1), datetime(2026, 1, 2)
    for view in (
        _batch_view([_agg("sum", 259200)]),  # batch tile (IcebergSource), offline=True
        _stream_tile_view(  # streaming tile (enable_tiling), offline=True via _stream_view default
            _kafka_source(watermark=True), [_agg("sum", 259200)], interval_secs=86400
        ),
    ):
        job = eng._materialize_one(MagicMock(), MaterializationTask("proj", view, start, end))
        assert job.status() == MaterializationJobStatus.SUCCEEDED
        assert job.error() is None


def test_materialize_one_plain_stream_view_still_enters_backfill(monkeypatch):
    # Routing-boundary guard: the tile no-op must NOT swallow a plain (non-tile) stream view — it still
    # enters the windowed-agg backfill path. We make the builder explode and assert the error surfaces.
    import feast.infra.compute_engines.risingwave.engine as eng_mod

    sentinel = RuntimeError("entered-backfill")

    def _explode(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(eng_mod, "RisingWaveFeatureBuilder", _explode)
    eng = _engine()
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])  # offline=True default, NOT a tile
    job = eng._materialize_one(
        MagicMock(), MaterializationTask("proj", view, datetime(2026, 1, 1), datetime(2026, 1, 2))
    )
    assert job.status() == MaterializationJobStatus.ERROR
    assert job.error() is sentinel


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


# --- feature_refs subset + full_feature_names on the custom PIT read paths ---
# The two custom RisingWave read paths (passthrough as-of, tile rollup) must obey the standard Feast
# offline contract: project ONLY the requested features, and prefix feature outputs as
# "{view}__{feature}" when full_feature_names is set (entity/join-key columns are never prefixed).


def _odfv_registry():
    # Both _get_requested_feature_views_to_features_dict (positional) and
    # OnDemandFeatureView.get_requested_odfvs (allow_cache kw) call list_on_demand_feature_views.
    return SimpleNamespace(
        list_on_demand_feature_views=lambda project, allow_cache=False: []
    )


def _with_projection(view):
    # Real FeatureViews carry a projection; the SimpleNamespace fixtures don't. name_to_use() is the
    # name the offline contract prefixes with, defaulting to the view name (no alias here).
    view.projection = SimpleNamespace(name_to_use=lambda: view.name)
    return view


def _captured_pit_query(view, feature_refs, full_feature_names):
    from unittest.mock import patch

    entity_df = pd.DataFrame(
        {"user_id": ["u1"], "event_timestamp": [pd.Timestamp("2026-06-04")]}
    )
    target = (
        "feast.infra.compute_engines.risingwave.offline_store.PostgreSQLRetrievalJob"
    )
    with patch(target) as job:
        RisingWaveOfflineStore.get_historical_features(
            config=MagicMock(),
            feature_views=[view],
            feature_refs=feature_refs,
            entity_df=entity_df,
            registry=_odfv_registry(),
            project="proj",
            full_feature_names=full_feature_names,
        )
    return job.call_args.kwargs


# --- passthrough as-of read ---


def test_passthrough_pit_projects_only_the_requested_feature_subset():
    view = _with_projection(_passthrough_batch_view(feature_cols=("amount", "country")))
    sql = _captured_pit_query(
        view, ["user_attr_daily:amount"], full_feature_names=False
    )["query"]
    assert '"amount"' in sql  # the requested feature is projected
    assert "country" not in sql  # the unrequested feature is NOT pulled


def test_passthrough_pit_full_feature_names_prefixes_only_features_not_entities():
    view = _with_projection(_passthrough_batch_view(feature_cols=("amount", "country")))
    kwargs = _captured_pit_query(
        view,
        ["user_attr_daily:amount", "user_attr_daily:country"],
        full_feature_names=True,
    )
    sql = kwargs["query"]
    assert kwargs["full_feature_names"] is True  # threaded to the retrieval job too
    assert '"amount" AS "user_attr_daily__amount"' in sql
    assert '"country" AS "user_attr_daily__country"' in sql
    # entity / label-timestamp columns are never prefixed
    assert "user_attr_daily__user_id" not in sql
    assert "user_attr_daily__event_timestamp" not in sql


def test_passthrough_pit_bare_feature_names_when_full_feature_names_false():
    view = _with_projection(_passthrough_batch_view(feature_cols=("amount", "country")))
    sql = _captured_pit_query(
        view,
        ["user_attr_daily:amount", "user_attr_daily:country"],
        full_feature_names=False,
    )["query"]
    assert '"amount"' in sql and '"country"' in sql
    assert "user_attr_daily__" not in sql  # no view prefix on any column


def test_passthrough_pit_builder_full_feature_names_prefixes_only_feature_columns():
    # Builder-level pin of the naming contract (independent of the offline-store wiring).
    ci = ColumnInfo(
        join_keys=["user_id"], feature_cols=["amount", "country"], ts_col="event_ts",
        created_ts_col=None, field_mapping=None,
    )
    sql = build_passthrough_pit_query(
        "SELECT 1", ["user_id", "event_timestamp"], "event_timestamp",
        history_relation="hist", column_info=ci,
        full_feature_names=True, view_name="user_attr",
    )
    assert '"amount" AS "user_attr__amount"' in sql
    assert '"country" AS "user_attr__country"' in sql
    assert "user_attr__user_id" not in sql  # entity key bare
    assert "user_attr__event_timestamp" not in sql  # label ts bare


# --- tile rollup read ---


def test_offline_tile_pit_rolls_up_only_the_requested_window_subset():
    view = _with_projection(_batch_view([_agg("sum", 259200), _agg("sum", 2592000)]))
    sql = _captured_pit_query(
        view, ["user_txn_daily:sum_amount_259200s"], full_feature_names=False
    )["query"]
    assert "sum_amount_259200s" in sql  # the requested 3d window
    assert "sum_amount_2592000s" not in sql  # the unrequested 30d window is not rolled up


def test_offline_tile_pit_full_feature_names_prefixes_only_features_not_entities():
    view = _with_projection(_batch_view([_agg("sum", 259200), _agg("sum", 2592000)]))
    kwargs = _captured_pit_query(
        view,
        ["user_txn_daily:sum_amount_259200s", "user_txn_daily:sum_amount_2592000s"],
        full_feature_names=True,
    )
    sql = kwargs["query"]
    assert kwargs["full_feature_names"] is True
    assert 'AS "user_txn_daily__sum_amount_259200s"' in sql
    assert 'AS "user_txn_daily__sum_amount_2592000s"' in sql
    assert "user_txn_daily__user_id" not in sql  # entity key bare
    assert "user_txn_daily__event_timestamp" not in sql  # label ts bare


def test_offline_tile_pit_bare_feature_names_when_full_feature_names_false():
    view = _with_projection(_batch_view([_agg("sum", 259200)]))
    sql = _captured_pit_query(
        view, ["user_txn_daily:sum_amount_259200s"], full_feature_names=False
    )["query"]
    assert "AS sum_amount_259200s" in sql  # bare per-window name
    assert "user_txn_daily__" not in sql  # no view prefix


def test_offline_tile_pit_builder_full_feature_names_prefixes_only_feature_columns():
    # Builder-level pin: the rollup output alias carries the "{view}__{feature}" prefix; entity columns
    # (projected as e."...") do not.
    sql = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, TIMESTAMP '2026-06-04' AS event_timestamp",
        ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="t", column_info=_column_info(),
        aggregations=[_agg("sum", 259200)], aggregation_interval=timedelta(days=1),
        full_feature_names=True, view_name="user_txn_daily",
    )
    assert 'AS "user_txn_daily__sum_amount_259200s"' in sql  # feature prefixed
    assert 'e."user_id"' in sql and "user_txn_daily__user_id" not in sql  # entity key bare
    assert 'e."event_timestamp"' in sql  # label ts bare


# --- multi-view composition: LEFT JOIN each view's per-view PIT over one shared entity spine ---
# A FeatureService mixing tile + passthrough views (or several of either) trains in ONE
# get_historical_features call: each view's per-view point-in-time read becomes a CTE over the shared
# entity spine, and the CTEs are LEFT JOINed back on the entity columns (the row identity) into one frame.


def _captured_multi_view_query(views, feature_refs, full_feature_names):
    from unittest.mock import patch

    entity_df = pd.DataFrame(
        {"user_id": ["u1"], "event_timestamp": [pd.Timestamp("2026-06-04")]}
    )
    target = (
        "feast.infra.compute_engines.risingwave.offline_store.PostgreSQLRetrievalJob"
    )
    with patch(target) as job:
        RisingWaveOfflineStore.get_historical_features(
            config=MagicMock(),
            feature_views=views,
            feature_refs=feature_refs,
            entity_df=entity_df,
            registry=_odfv_registry(),
            project="proj",
            full_feature_names=full_feature_names,
        )
    return job.call_args.kwargs["query"]


def test_multi_view_left_joins_tile_and_passthrough_over_one_spine():
    tile = _with_projection(_batch_view([_agg("sum", 259200)]))  # user_txn_daily
    passthrough = _with_projection(_passthrough_batch_view(("country",)))  # user_attr_daily
    sql = _captured_multi_view_query(
        [tile, passthrough],
        ["user_txn_daily:sum_amount_259200s", "user_attr_daily:country"],
        full_feature_names=False,
    )
    # one composed query: a CTE per view, LEFT JOINed over the shared entity spine
    assert sql.startswith("WITH ")
    assert '"_feast_view_0" AS (' in sql and '"_feast_view_1" AS (' in sql
    assert ' LEFT JOIN "_feast_view_1" ON ' in sql
    # the row identity is the entity columns (joined across views), NOT the features
    assert '"_feast_view_0"."user_id" = "_feast_view_1"."user_id"' in sql
    assert '"_feast_view_0"."event_timestamp" = "_feast_view_1"."event_timestamp"' in sql
    # each view's per-view PIT is present as a CTE...
    assert "date_trunc('day'" in sql  # tile rollup CTE
    assert "ROW_NUMBER() OVER (PARTITION BY" in sql  # passthrough as-of CTE
    assert "LEFT JOIN proj_user_attr_daily_src h" in sql  # passthrough reads its batch source
    # ...and each view's feature column is projected into the single output frame
    assert '"_feast_view_0"."sum_amount_259200s"' in sql
    assert '"_feast_view_1"."country"' in sql


def test_multi_view_full_feature_names_projects_prefixed_feature_columns():
    # full_feature_names prefixes each feature output as "{view}__{feature}" — distinct per view, so two
    # views' features cannot collide; the composed frame projects those prefixed names.
    tile = _with_projection(_batch_view([_agg("sum", 259200)]))
    passthrough = _with_projection(_passthrough_batch_view(("country",)))
    sql = _captured_multi_view_query(
        [tile, passthrough],
        ["user_txn_daily:sum_amount_259200s", "user_attr_daily:country"],
        full_feature_names=True,
    )
    assert 'AS "user_txn_daily__sum_amount_259200s"' in sql  # tile CTE prefixes its feature
    assert '"country" AS "user_attr_daily__country"' in sql  # passthrough CTE prefixes its feature
    assert '"_feast_view_0"."user_txn_daily__sum_amount_259200s"' in sql  # projected from the frame
    assert '"_feast_view_1"."user_attr_daily__country"' in sql
    # entity columns are joined bare (never prefixed), so the spine identity is unambiguous
    assert '"_feast_view_0"."user_id" = "_feast_view_1"."user_id"' in sql


def test_single_view_query_is_unchanged_by_multi_view_support():
    # A single requested view must NOT be wrapped in the multi-view CTE join — the bare per-view PIT is
    # returned byte-for-byte, so multi-view support is a no-op on the one-view path.
    from feast.infra.compute_engines.risingwave.names import tiles_name

    tile = _with_projection(_batch_view([_agg("sum", 259200)]))
    sql = _captured_multi_view_query(
        [tile], ["user_txn_daily:sum_amount_259200s"], full_feature_names=False
    )
    assert "_feast_view_" not in sql  # no multi-view CTE wrapper
    assert sql.startswith("SELECT ")
    expected = build_offline_tile_pit_query(
        _entity_df_to_sql(
            pd.DataFrame(
                {"user_id": ["u1"], "event_timestamp": [pd.Timestamp("2026-06-04")]}
            )
        ),
        ["user_id", "event_timestamp"],
        "event_timestamp",
        tiles_relation=tiles_name("proj", "user_txn_daily"),
        column_info=ColumnInfo(
            join_keys=["user_id"],
            feature_cols=["sum_amount_259200s"],
            ts_col="event_timestamp",
            created_ts_col=None,
            field_mapping=None,
        ),
        aggregations=list(tile.batch_source.aggregations),
        aggregation_interval=timedelta(days=1),
        full_feature_names=False,
        view_name="user_txn_daily",
    )
    assert sql == expected  # byte-identical to the standalone single-view builder output


def test_multi_view_without_full_feature_names_rejects_colliding_feature_names():
    # Two views projecting the same bare feature name would put a duplicate column in the joined frame;
    # reject clearly. (full_feature_names disambiguates by prefixing — see the test above.)
    a = _with_projection(_passthrough_batch_view(("amount",), name="attr_a"))
    b = _with_projection(_passthrough_batch_view(("amount",), name="attr_b"))
    with pytest.raises(NotImplementedError, match="colliding feature names"):
        _captured_multi_view_query(
            [a, b], ["attr_a:amount", "attr_b:amount"], full_feature_names=False
        )


def test_get_historical_features_rejects_custom_plus_plain_view_mix():
    # A plain (parent-served) view has no shared entity spine with the custom reads, so mixing it with a
    # tile/passthrough view in one call is rejected rather than silently dropping it.
    tile = _with_projection(_batch_view([_agg("sum", 259200)]))
    plain = SimpleNamespace(name="plain_view")  # neither tile nor passthrough
    entity_df = pd.DataFrame(
        {"user_id": ["u1"], "event_timestamp": [pd.Timestamp("2026-06-04")]}
    )
    with pytest.raises(NotImplementedError, match="plain parent-served views"):
        RisingWaveOfflineStore.get_historical_features(
            config=MagicMock(),
            feature_views=[tile, plain],
            feature_refs=["user_txn_daily:sum_amount_259200s"],
            entity_df=entity_df,
            registry=_odfv_registry(),
            project="proj",
        )


# --- v2 cumulative-subtraction serving: ONE running-total tile MV serves every invertible window ---
# (sum/count/mean/var/stddev) by read-time 2-point asof subtraction, instead of one now()-anchored MV
# per window. Non-invertible aggregations (min/max/count_distinct/sequence) keep the v1 per-window MVs.


def _cum_ci():
    # The cumulative builders take a ColumnInfo (join keys + event-time column); reuse the file's shape.
    return ColumnInfo(
        join_keys=["user_id"],
        feature_cols=["sum_amount_259200s"],
        ts_col="event_ts",
        created_ts_col=None,
        field_mapping=None,
    )


def test_cumulative_tile_select_runs_partials_OVER_tile_end():
    # The cumulative-tile MV is ONE window-agnostic running total per partial COLUMN over tile_end — the
    # source of every invertible window by later subtraction. mean@7d reuses the sum+count partials, so a
    # sum@3d + mean@7d view stores cum_ntiles, cum_sum_amount, cum_count_amount (the deduped invertible set).
    ci = _cum_ci()
    sql = build_cumulative_tile_select(ci, [_agg("sum", 259200), _agg("mean", 604800)], "tiles")
    assert "count(*) OVER (PARTITION BY user_id ORDER BY tile_end) AS cum_ntiles" in sql
    assert "sum(sum_amount) OVER (PARTITION BY user_id ORDER BY tile_end) AS cum_sum_amount" in sql
    assert "sum(count_amount) OVER (PARTITION BY user_id ORDER BY tile_end) AS cum_count_amount" in sql
    assert sql.endswith("FROM tiles")  # reads the tiles MV (source-agnostic), no now()


def test_cumulative_read_subtracts_two_asof_points_for_a_trailing_window():
    # A TRAILING window = cum_at_end - cum_at_(end - window): two asof LATERAL reads (latest tile <= each
    # bound) subtracted, with the cum_ntiles empty-window guard mapping a no-tile window to NULL (offline
    # parity). end is the request-time floor; the lower bound is end - window.
    ci = _cum_ci()
    sql = build_cumulative_read_query(
        "(SPINE)",
        ["user_id", "event_timestamp"],
        "event_timestamp",
        cumulative_relation="cum",
        column_info=ci,
        aggregations=[_agg("sum", 259200)],
        aggregation_interval=timedelta(days=1),
    )
    assert sql.count("LEFT JOIN LATERAL") == 2  # one asof point per window edge (upper + lower)
    assert 'c.tile_end <= date_trunc(\'day\', e."event_timestamp")' in sql  # upper bound (window end)
    assert (
        'c.tile_end <= date_trunc(\'day\', e."event_timestamp") - INTERVAL \'259200\' SECOND' in sql
    )  # lower bound (window end - window_size)
    assert "CASE WHEN (COALESCE(a0.cum_ntiles, 0) - COALESCE(a1.cum_ntiles, 0)) = 0 THEN NULL" in sql
    assert "(COALESCE(a0.cum_sum_amount, 0) - COALESCE(a1.cum_sum_amount, 0))" in sql  # the subtraction


def test_cumulative_read_lifetime_has_no_lower_bound():
    # A LIFETIME invertible agg is the cumulative-to-end value = cum_at_end, with NO lower bound and NO
    # subtraction: a single asof point. (A floor would add an upper-floored asof; an unfloored lifetime has
    # only the one.)
    ci = _cum_ci()
    life = Aggregation(column="amount", function="sum", time_window=None)
    name = life.resolved_name(life.time_window)
    sql = build_cumulative_read_query(
        "(SPINE)",
        ["user_id", "event_timestamp"],
        "event_timestamp",
        cumulative_relation="cum",
        column_info=ci,
        aggregations=[life],
        aggregation_interval=timedelta(days=1),
        lifetimes={name: None},
    )
    assert sql.count("LEFT JOIN LATERAL") == 1  # only ONE asof point (the cumulative-to-end value)
    assert "INTERVAL" not in sql  # no lower-bound shift
    assert " - COALESCE(" not in sql  # no 2-point subtraction
    assert "CASE WHEN COALESCE(a0.cum_ntiles, 0) = 0 THEN NULL ELSE COALESCE(a0.cum_sum_amount, 0) END" in sql


def test_cumulative_read_rejects_a_series():
    # A window-series is NOT served by the cumulative MV: assembling L windows as L stacked asof LATERALs
    # is an O(L)-deep correlated decorrelation the optimizer cannot plan at series scale. The series goes
    # through the single-scan build_offline_tile_pit_query instead; the cumulative read rejects a series.
    ci = _cum_ci()
    ser = Aggregation(column="amount", function="sum", time_window=None, name="daily_sum_3")
    with pytest.raises(ValueError, match="does not serve a window-series"):
        build_cumulative_read_query(
            "(SPINE)", ["user_id", "event_timestamp"], "event_timestamp",
            cumulative_relation="cum", column_info=ci, aggregations=[ser],
            aggregation_interval=timedelta(days=1), series={"daily_sum_3": [86400, 86400, 3]},
        )


def test_desired_online_mvs_splits_invertible_to_cumulative_and_noninvertible_to_window():
    # The v2 split lives in _desired_online_mvs: an invertible agg (sum) -> the single cumulative MV, a
    # non-invertible agg (max) -> its own per-(window) now()-anchored MV. So a view with both gets the
    # cumulative MV AND a per-window MV, but NO per-window MV for the invertible sum.
    ci = _cum_ci()
    mvs = _desired_online_mvs(
        "proj",
        "user_txn_daily",
        ci,
        [_agg("sum", 259200), _agg("max", 259200)],
        "proj_user_txn_daily_tiles",
        aggregation_interval=timedelta(days=1),
        agg_params={},
        secondary_key=None,
        offsets={},
        lifetimes={},
        series={},
    )
    assert "proj_user_txn_daily_online_cum" in mvs  # ONE cumulative MV for the invertible sum
    assert "proj_user_txn_daily_online_259200s" in mvs  # per-window MV for the non-invertible max
    # the invertible sum has NO per-window MV (it is derived from the cumulative MV at read time)
    assert "cum_sum_amount" in mvs["proj_user_txn_daily_online_cum"]
    assert "max(max_amount) AS max_amount_259200s" in mvs["proj_user_txn_daily_online_259200s"]
    assert "sum(sum_amount)" not in mvs["proj_user_txn_daily_online_259200s"]


def test_desired_online_mvs_secondary_key_view_stays_v1():
    # A SECONDARY-KEY view is excluded from the cumulative path (the cumulative MV carries no per-key Map
    # dimension), so even an invertible sum keeps the v1 per-(window) now()-anchored MV — NO cumulative MV.
    ci = _cum_ci()
    mvs = _desired_online_mvs(
        "proj",
        "user_txn_daily",
        ci,
        [_agg("sum", 259200)],
        "proj_user_txn_daily_tiles",
        aggregation_interval=timedelta(days=1),
        agg_params={},
        secondary_key="ad_id",
        offsets={},
        lifetimes={},
        series={},
    )
    assert "proj_user_txn_daily_online_cum" not in mvs  # cumulative path disabled for secondary-key views
    assert "proj_user_txn_daily_online_259200s" in mvs  # the per-window MV is used instead


# --- window-series SNAPSHOT MV: a step==interval series materializes as last-L (tile_end, value) pairs ---


def _ser(function, name, column="amount"):
    return Aggregation(column=column, function=function, time_window=None, name=name)


def test_series_snapshot_select_step_equals_interval_emits_last_l_pairs():
    # A step==interval series (each element == one tile) materializes per entity as the last-`depth` tiles'
    # ends plus each series' per-tile value, both as array_agg over a per-entity row_number()<=depth TopN.
    # depth is the longest series (so a shorter series shares the same ends array, positioned by the reader).
    sql = build_series_snapshot_select(
        _column_info(), [_ser("sum", "daily_sum_3"), _ser("max", "daily_max_4")], "tiles_rel",
        aggregation_interval=timedelta(days=1), agg_params={},
        series={"daily_sum_3": [86400, 86400, 3], "daily_max_4": [86400, 86400, 4]},
    )
    assert f"array_agg(tile_end ORDER BY tile_end DESC) AS {SERIES_SNAPSHOT_ENDS_COL}" in sql
    assert "sum_amount AS daily_sum_3" in sql  # per-tile finalize (identity for sum) in the inner TopN
    assert "max_amount AS daily_max_4" in sql
    # each value array is trimmed to its OWN series length via a FILTER on the shared TopN
    assert "array_agg(daily_sum_3 ORDER BY tile_end DESC) FILTER (WHERE __rn <= 3) AS daily_sum_3" in sql
    assert "array_agg(daily_max_4 ORDER BY tile_end DESC) FILTER (WHERE __rn <= 4) AS daily_max_4" in sql
    assert "array_agg(tile_end ORDER BY tile_end DESC) AS __series_tile_ends" in sql  # ends stays at max depth
    assert "row_number() OVER (PARTITION BY user_id ORDER BY tile_end DESC) AS __rn" in sql
    assert "WHERE __rn <= 4" in sql  # depth = max(3, 4)
    assert sql.rstrip().endswith("GROUP BY user_id")


def test_series_snapshot_select_finalizes_mean_per_tile():
    # mean's per-tile value is the tile's sum/count — the SAME algebra the single-scan recombines, with the
    # cross-tile merge collapsed to identity (one tile per element).
    sql = build_series_snapshot_select(
        _column_info(), [_ser("mean", "daily_mean_2")], "tiles_rel",
        aggregation_interval=timedelta(days=1), agg_params={},
        series={"daily_mean_2": [86400, 86400, 2]},
    )
    assert "sum_amount / NULLIF(count_amount, 0) AS daily_mean_2" in sql
    assert "WHERE __rn <= 2" in sql


def test_series_snapshot_select_none_for_ineligible_series():
    # A coarser step (k>1, frontier-relative buckets), an overlapping window, and an array-valued aggregate
    # are NOT snapshotted — they keep the read-time single-scan, so the builder returns None.
    ci, day = _column_info(), timedelta(days=1)
    assert build_series_snapshot_select(  # coarse step: step 2d > interval 1d
        ci, [_ser("sum", "x")], "t", aggregation_interval=day, agg_params={},
        series={"x": [172800, 172800, 2]}) is None
    assert build_series_snapshot_select(  # overlapping: window 2d > step 1d
        ci, [_ser("sum", "x")], "t", aggregation_interval=day, agg_params={},
        series={"x": [172800, 86400, 2]}) is None
    assert build_series_snapshot_select(  # array-valued: count_distinct
        ci, [_ser("count_distinct", "x")], "t", aggregation_interval=day, agg_params={},
        series={"x": [86400, 86400, 3]}) is None
    assert build_series_snapshot_select(  # no series at all
        ci, [_agg("sum", 86400)], "t", aggregation_interval=day, agg_params={}, series={}) is None


def test_snapshot_series_aggs_keeps_only_step_equals_interval_scalar():
    aggs = [_ser("sum", "s"), _ser("count_distinct", "cd"), _ser("max", "coarse"), _ser("sum", "overlap")]
    eligible = snapshot_series_aggs(
        aggs, {"s": [86400, 86400, 3], "cd": [86400, 86400, 3],
               "coarse": [172800, 172800, 2], "overlap": [172800, 86400, 2]}, 86400)
    assert [(a.resolved_name(a.time_window), length) for a, length in eligible] == [("s", 3)]


def test_desired_online_mvs_emits_series_snapshot_for_step_interval_series():
    # _desired_online_mvs gains ONE per-view snapshot MV when a step==interval series is present (named
    # ..._online_series), and emits none for a coarse-step series (which stays on the single-scan).
    ci = _cum_ci()
    mvs = _desired_online_mvs(
        "proj", "v", ci, [_ser("max", "daily_max_5")], "proj_v_tiles",
        aggregation_interval=timedelta(days=1), agg_params={}, secondary_key=None,
        offsets={}, lifetimes={}, series={"daily_max_5": [86400, 86400, 5]},
    )
    assert online_series_mv_name("proj", "v") in mvs  # == "proj_v_online_series"
    assert "array_agg(daily_max_5 ORDER BY tile_end DESC) FILTER (WHERE __rn <= 5) AS daily_max_5" in mvs[online_series_mv_name("proj", "v")]

    coarse = _desired_online_mvs(
        "proj", "v", ci, [_ser("max", "biday_max_5")], "proj_v_tiles",
        aggregation_interval=timedelta(days=1), agg_params={}, secondary_key=None,
        offsets={}, lifetimes={}, series={"biday_max_5": [172800, 172800, 5]},
    )
    assert online_series_mv_name("proj", "v") not in coarse  # coarse step: no snapshot MV


def test_batch_drop_ddl_drops_series_snapshot_before_tiles():
    # the snapshot MV reads the tiles MV, so teardown must drop it BEFORE the tiles MV (dependency order).
    view = _batch_view([_ser("max", "daily_max_5")])
    view.tags = encode_agg_series({"daily_max_5": [86400, 86400, 5]})
    ddl = _batch_drop_ddl("proj", view)
    series_drop = f'DROP MATERIALIZED VIEW IF EXISTS "{online_series_mv_name("proj", "user_txn_daily")}"'
    tiles_drop = 'DROP MATERIALIZED VIEW IF EXISTS "proj_user_txn_daily_tiles"'
    assert series_drop in ddl and tiles_drop in ddl
    assert ddl.index(series_drop) < ddl.index(tiles_drop)


def test_existing_online_mv_names_sweep_matches_series_snapshot():
    # the reconcile sweep must SEE a deployed snapshot MV, else it would re-CREATE-IF-NOT-EXISTS forever and
    # never drop a removed one. The widened regex matches ..._online_series alongside the other forms.
    class _Cur:
        def execute(self, sql):
            self.rows = [("proj_v_online_series",), ("proj_v_online_cum",), ("proj_other_online_series",)]

        def fetchall(self):
            return self.rows

    names = _existing_online_mv_names(_Cur(), "proj", "v")
    assert "proj_v_online_series" in names
    assert "proj_other_online_series" not in names  # anchored to THIS view's base name


def test_desired_online_mvs_excludes_series_snapshot_for_secondary_key_view():
    # A secondary-key view's series is a per-key Map of arrays offline; the snapshot collapses to the join
    # keys, so it must NOT be materialized — the series stays on the read-time single-scan (same guard the
    # cumulative path uses). No online_series MV is emitted.
    ci = _cum_ci()
    mvs = _desired_online_mvs(
        "proj", "v", ci, [_ser("max", "daily_max_5")], "proj_v_tiles",
        aggregation_interval=timedelta(days=1), agg_params={}, secondary_key="ad_id",
        offsets={}, lifetimes={}, series={"daily_max_5": [86400, 86400, 5]},
    )
    assert online_series_mv_name("proj", "v") not in mvs


@pytest.mark.parametrize("fn", ["sum", "count", "min", "max", "mean",
                                "var_pop", "var_samp", "stddev_pop", "stddev_samp"])
def test_tile_value_expr_equals_recombine_over_one_tile(fn):
    # PARITY PIN (online == offline): the snapshot's per-tile value MUST equal the offline single-scan
    # recombine evaluated over exactly ONE tile. Over one tile the cross-tile merge is identity
    # (sum(p)==p, min(p)==p, max(p)==p), so stripping those wrappers from _recombine_expr must reproduce
    # _tile_value_expr verbatim — for EVERY snapshotted function (sum/count/min/max/mean/var/stddev). If a
    # finalize formula ever changes in only one place, this fails.
    import re

    agg = _ser(fn, "feat")
    recombine = _recombine_expr(agg)                      # e.g. sum(sum_amount) / NULLIF(sum(count_amount), 0)
    one_tile = re.sub(r"\b(?:sum|min|max)\((\w+)\)", r"\1", recombine)  # collapse the single-tile merge
    assert one_tile == _tile_value_expr(agg)


# --- transient-DDL retry: a RisingWave cluster-state error on CREATE/DROP is retried, a real error is not ---


class _FakeCursor:
    def __init__(self, fail_times, exc):
        self.calls = 0
        self._fail_times = fail_times
        self._exc = exc

    def execute(self, sql):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc


def test_execute_ddl_retries_a_transient_cluster_error_then_succeeds():
    from feast.infra.compute_engines.risingwave.engine import _execute_ddl

    cur = _FakeCursor(fail_times=2, exc=RuntimeError("Scheduler error: streaming vnode mapping not found"))
    _execute_ddl(cur, 'CREATE MATERIALIZED VIEW "x" AS SELECT 1', backoff_ms=0)
    assert cur.calls == 3  # two transient failures retried, third attempt succeeds


def test_execute_ddl_reraises_a_permanent_error_without_retrying():
    from feast.infra.compute_engines.risingwave.engine import _execute_ddl

    cur = _FakeCursor(fail_times=99, exc=ValueError("syntax error near 'SELCT'"))
    with pytest.raises(ValueError):
        _execute_ddl(cur, "SELCT 1", backoff_ms=0)
    assert cur.calls == 1  # a permanent (non-transient) error is raised on the first attempt, no retry


def test_partial_column_name_matches_legacy_names():
    # the Partial atom is the single namer; for filter=None it MUST reproduce the legacy bare names
    # byte-identically (so the tile-plan refactor changes structure, not emitted SQL).
    from feast.infra.compute_engines.risingwave.tiling import Partial

    assert Partial("sum", "amount").column_name() == "sum_amount"
    assert Partial("count", "amount").column_name() == "count_amount"
    assert Partial("min", "amount").column_name() == "min_amount"
    assert Partial("max", "amount").column_name() == "max_amount"
    assert Partial("sumsq", "amount").column_name() == "sumsq_amount"
    assert Partial("distinct", "amount").column_name() == "distinct_amount"
    assert Partial("last", "amount", n=5).column_name() == "last_amount_5"
    # a filtered partial appends a stable hash suffix, distinct from the unfiltered name
    filtered = Partial("count", "amount", filter="transaction_code = 'DEBIT'")
    assert filtered.column_name().startswith("count_amount_f")
    assert filtered.column_name() != "count_amount"
    assert filtered.column_name() == Partial("count", "amount", filter="transaction_code = 'DEBIT'").column_name()


def test_partials_for_pairs_unchanged():
    # _partials_for is now a shim over Partial; the (name, sql) pairs must be exactly the legacy ones.
    import datetime as dt

    from feast.aggregation import Aggregation
    from feast.infra.compute_engines.risingwave.tiling import _partials_for

    def a(fn):
        return Aggregation(column="amount", function=fn, time_window=dt.timedelta(days=1))

    assert _partials_for(a("sum")) == [("sum_amount", "sum(amount)")]
    assert _partials_for(a("count")) == [("count_amount", "count(amount)")]
    assert _partials_for(a("min")) == [("min_amount", "min(amount)")]
    assert _partials_for(a("mean")) == [("sum_amount", "sum(amount)"), ("count_amount", "count(amount)")]
    assert _partials_for(a("var_pop")) == [
        ("sum_amount", "sum(amount)"),
        ("count_amount", "count(amount)"),
        ("sumsq_amount", "sum(amount * amount)"),
    ]
    assert _partials_for(a("count_distinct")) == [
        ("distinct_amount", "array_agg(DISTINCT amount) FILTER (WHERE amount IS NOT NULL)")
    ]


def test_view_partials_raises_on_name_collision(monkeypatch):
    # the assert-equal dedup: if two aggregations rendered the SAME partial name with DIFFERENT SQL
    # (what a filtered partial would do without its predicate-hash suffix), _view_partials must raise,
    # not silently drop one.
    import datetime as dt

    import pytest

    from feast.aggregation import Aggregation
    import feast.infra.compute_engines.risingwave.tiling as tiling

    monkeypatch.setattr(tiling, "_partials_for", lambda a, *args, **kw: [("count_amount", f"count({a.function})")])
    aggs = [
        Aggregation(column="amount", function="count", time_window=dt.timedelta(days=1)),
        Aggregation(column="amount", function="sum", time_window=dt.timedelta(days=1)),
    ]
    with pytest.raises(ValueError, match="collision"):
        tiling._view_partials(aggs)
