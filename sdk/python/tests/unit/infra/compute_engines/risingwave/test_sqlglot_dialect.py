"""RisingWaveStreaming SQLGlot dialect — faithful round-trip of the streaming-SELECT constructs.

The dialect is the codegen substrate the TilePlan node ``.select()`` leaves can migrate onto (f-strings ->
SQLGlot AST). These tests pin that it PARSES and round-trips the RisingWave streaming constructs the stock
``risingwave`` dialect drops — ``now()`` (not folded to CURRENT_TIMESTAMP), the ``INTERVAL '<n>' <unit>`` form,
and ``EMIT ON WINDOW CLOSE`` (which the stock dialect cannot even parse). The guarantee is SEMANTIC +
idempotent (re-rendering the output is stable), not byte-identical to the f-string builders — SQLGlot renders
known aggregates / date_trunc in a canonical (upper) spelling.
"""

import datetime as dt

import pytest
import sqlglot

from feast.aggregation import Aggregation
from feast.infra.compute_engines.dag.context import ColumnInfo
from feast.infra.compute_engines.risingwave.sql_builders import (
    build_batch_tile_select,
    build_cumulative_read_query,
    build_cumulative_tile_select,
    build_latest_row_select,
    build_lifetime_rollup_select,
    build_offline_tile_pit_query,
    build_online_rollup_select,
    build_series_snapshot_select,
    build_streaming_tile_select,
    build_tile_rollup_select,
    build_windowed_agg_select,
)
from feast.infra.compute_engines.risingwave.sqlglot_dialect import RisingWaveStreaming

_CI = ColumnInfo(join_keys=["user_id"], feature_cols=["amount"], ts_col="event_ts",
                 created_ts_col=None, field_mapping=None)
_DAY, _HOUR = dt.timedelta(days=1), dt.timedelta(seconds=3600)
_AGGS = [Aggregation(column="amount", function="sum", time_window=_DAY),
         Aggregation(column="amount", function="count", time_window=_DAY)]
_MEANVAR = [Aggregation(column="amount", function="mean", time_window=_DAY),
            Aggregation(column="amount", function="var_pop", time_window=_DAY)]
_MAX = Aggregation(column="amount", function="max", time_window=_DAY)  # non-invertible -> rollup MVs
_CDISTINCT = Aggregation(column="amount", function="count_distinct", time_window=_DAY)
_SEQ = Aggregation(column="amount", function="first", time_window=_DAY, name="seq5")  # -> array_agg[1:n]
_PCT = Aggregation(column="amount", function="approx_percentile", time_window=_DAY, name="p95")  # -> WITHIN GROUP
_SPINE = "SELECT 1 AS user_id, now() AS event_timestamp"


def _render(tree):
    # normalize_functions=False keeps anonymous-function spellings; known aggregates still canonicalize.
    return tree.sql(dialect=RisingWaveStreaming, normalize_functions=False)


def _roundtrip(sql):
    return _render(sqlglot.parse_one(sql, read=RisingWaveStreaming))


# --- the specific constructs (DB-free, upstream-shaped) ---

def test_emit_on_window_close_parses_and_round_trips():
    sql = ("SELECT user_id, count(*) AS n FROM tumble(src, event_ts, INTERVAL '60' SECOND) "
           "GROUP BY user_id EMIT ON WINDOW CLOSE")
    out = _roundtrip(sql)
    assert out.endswith("EMIT ON WINDOW CLOSE")
    assert _roundtrip(out) == out  # idempotent canonical form


def test_now_is_preserved_and_uses_the_risingwave_interval_form():
    out = _roundtrip("SELECT now() - INTERVAL '604800' SECOND AS lower_bound")
    assert "now()" in out and "CURRENT_TIMESTAMP" not in out  # not folded to the per-txn-fixed value
    assert "INTERVAL '604800' SECOND" in out  # RW form, not Postgres' INTERVAL '604800 SECOND'


def test_tumble_table_function_survives_round_trip():
    out = _roundtrip("SELECT * FROM tumble(src, event_ts, INTERVAL '60' SECOND)")
    assert "tumble" in out.lower() and "INTERVAL '60' SECOND" in out


def test_stock_risingwave_dialect_lacks_both():
    # Documents the gap the enrichment closes: the stock dialect cannot parse EMIT ON WINDOW CLOSE and folds
    # now() to CURRENT_TIMESTAMP.
    with pytest.raises(Exception):
        sqlglot.parse_one("SELECT a FROM t GROUP BY a EMIT ON WINDOW CLOSE", read="risingwave")
    assert "CURRENT_TIMESTAMP" in sqlglot.parse_one("SELECT now()", read="risingwave").sql(dialect="risingwave")


# --- the dialect round-trips the engine's ACTUAL streaming SQL (semantic + idempotent) ---

# Every f-string builder in sql_builders.py, across its aggregate variants — so the dialect's coverage of the
# whole engine SQL surface is pinned (a SQLGlot bump that breaks any construct's round-trip is caught here).
_ENGINE_SQL = {
    "batch_tile": build_batch_tile_select(_CI, _AGGS + _MEANVAR, "src", aggregation_interval=_DAY),
    "batch_tile_count_distinct": build_batch_tile_select(_CI, [_CDISTINCT], "src", aggregation_interval=_DAY),
    "streaming_tile_eowc_tumble": build_streaming_tile_select(_CI, _AGGS, "src", aggregation_interval=_HOUR),
    "cumulative_tile": build_cumulative_tile_select(_CI, _AGGS + _MEANVAR, "tiles"),
    "online_rollup_now": build_online_rollup_select(_CI, [_MAX], "tiles", aggregation_interval=_DAY),
    "tile_rollup_asof": build_tile_rollup_select(_CI, [_MAX], "tiles", aggregation_interval=_DAY, as_of_sql="now()"),
    "lifetime_rollup": build_lifetime_rollup_select(_CI, [_MAX], "tiles", lifetime_start_secs=None),
    "cumulative_read": build_cumulative_read_query(
        _SPINE, ["user_id", "event_timestamp"], "event_timestamp", cumulative_relation="cum",
        column_info=_CI, aggregations=_AGGS, aggregation_interval=_DAY, lifetimes={}),
    "offline_pit_trailing": build_offline_tile_pit_query(
        _SPINE, ["user_id", "event_timestamp"], "event_timestamp", tiles_relation="tiles",
        column_info=_CI, aggregations=[_AGGS[0], _MAX], aggregation_interval=_DAY),
    "offline_series_pit": build_offline_tile_pit_query(
        _SPINE, ["user_id", "event_timestamp"], "event_timestamp", tiles_relation="tiles",
        column_info=_CI, aggregations=[_AGGS[0]], aggregation_interval=_DAY,
        series={_AGGS[0].resolved_name(_AGGS[0].time_window): [86400, 86400, 5]}),
    "series_snapshot": build_series_snapshot_select(
        _CI, [_AGGS[0]], "tiles", aggregation_interval=_DAY,
        series={_AGGS[0].resolved_name(_AGGS[0].time_window): [86400, 86400, 5]}),
    "windowed_agg_v1": build_windowed_agg_select(
        _CI, _AGGS, "src", source_is_retractable=False, emit_on_close=True),
    "windowed_agg_sequence_slice": build_windowed_agg_select(
        _CI, [_SEQ], "src", source_is_retractable=False, emit_on_close=False, agg_params={"seq5": [5.0]}),
    "windowed_agg_approx_percentile": build_windowed_agg_select(
        _CI, [_PCT], "src", source_is_retractable=False, emit_on_close=False, agg_params={"p95": [0.95]}),
    "latest_row": build_latest_row_select(_CI, "src"),
}


@pytest.mark.parametrize("label", sorted(_ENGINE_SQL))
def test_engine_streaming_sql_round_trips_semantically(label):
    sql = _ENGINE_SQL[label]
    out = _roundtrip(sql)  # parses (no PARSE FAIL) under the enriched dialect
    assert _roundtrip(out) == out, "canonical form must be idempotent"
    # the RisingWave-specific constructs the stock dialect drops survive the round-trip
    if "EMIT ON WINDOW CLOSE" in sql:
        assert "EMIT ON WINDOW CLOSE" in out
    if "now()" in sql:
        assert "now()" in out and "CURRENT_TIMESTAMP" not in out
    if "tumble(" in sql.lower():
        assert "tumble" in out.lower()
    if "INTERVAL '" in sql:
        import re
        assert re.search(r"INTERVAL '\d+' \w+", out), "RisingWave interval form preserved"
