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
    build_offline_tile_pit_query,
    build_online_rollup_select,
    build_streaming_tile_select,
)
from feast.infra.compute_engines.risingwave.sqlglot_dialect import RisingWaveStreaming

_CI = ColumnInfo(join_keys=["user_id"], feature_cols=["amount"], ts_col="event_ts",
                 created_ts_col=None, field_mapping=None)
_AGGS = [Aggregation(column="amount", function="sum", time_window=dt.timedelta(days=1)),
         Aggregation(column="amount", function="count", time_window=dt.timedelta(days=1))]
_DAY, _HOUR = dt.timedelta(days=1), dt.timedelta(seconds=3600)


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

_ENGINE_SQL = {
    "streaming_tile": build_streaming_tile_select(_CI, _AGGS, "src", aggregation_interval=_HOUR),
    "online_rollup": build_online_rollup_select(_CI, [_AGGS[0]], "tiles", aggregation_interval=_DAY),
    "batch_tile": build_batch_tile_select(_CI, _AGGS, "src", aggregation_interval=_DAY),
    "offline_series_pit": build_offline_tile_pit_query(
        "SELECT 1 AS user_id, now() AS event_timestamp", ["user_id", "event_timestamp"], "event_timestamp",
        tiles_relation="tiles", column_info=_CI, aggregations=[_AGGS[0]], aggregation_interval=_DAY,
        series={_AGGS[0].resolved_name(_AGGS[0].time_window): [86400, 86400, 5]}),
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
