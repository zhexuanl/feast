"""TilePlan golden-equivalence + structure tests (DB-free).

The load-bearing guard for the tile-plan refactor: ``TilePlan.from_inputs(...).online_mvs()`` must equal
``_desired_online_mvs(...)`` CHARACTER-FOR-CHARACTER (same MV names, same SELECTs, same order) for the same
inputs across the view matrix — so routing provisioning / reconcile / drop / serving through the plan changes
structure, not emitted SQL. ``tiles_ddl()`` likewise equals ``build_*_tile_select``.
"""

import datetime as dt

import pytest

from feast.aggregation import Aggregation
from feast.infra.compute_engines.dag.context import ColumnInfo
from feast.infra.compute_engines.risingwave.ddl import _desired_online_mvs
from feast.infra.compute_engines.risingwave.sql_builders import (
    build_batch_tile_select,
    build_streaming_tile_select,
)
from feast.infra.compute_engines.risingwave.tile_plan import TilePlan

CI = ColumnInfo(join_keys=["user_id"], feature_cols=["amount"], ts_col="event_ts",
                created_ts_col=None, field_mapping=None)
PROJECT, VIEW, TILES, INTERVAL = "proj", "v", "v_tiles", dt.timedelta(days=1)


def _agg(fn, days=None, name=None):
    return Aggregation(column="amount", function=fn, time_window=dt.timedelta(days=days) if days else None, name=name)


def _rn(a):
    return a.resolved_name(a.time_window)


# (label, aggregations, carrier overrides) — the view matrix.
_LT = _agg("max", None, "max_lt")
_OFF = _agg("max", 1, "max_off")
_SER = _agg("sum", None, "sum_ser")
_SCENARIOS = [
    ("invertible trailing (sum/count/mean -> cumulative)", [_agg("sum", 1), _agg("count", 1), _agg("mean", 1)], {}),
    ("non-invertible (min/max/count_distinct -> per-window MVs)", [_agg("min", 1), _agg("max", 1), _agg("count_distinct", 1)], {}),
    ("mixed invertible + non-invertible", [_agg("sum", 1), _agg("max", 1)], {}),
    ("multi-window non-invertible (two window MVs)", [_agg("max", 1), _agg("max", 7)], {}),
    ("var/stddev composite (-> cumulative)", [_agg("var_pop", 7), _agg("stddev_samp", 30)], {}),
    ("lifetime non-invertible (-> lifetime MV)", [_LT], {"lifetimes": {_rn(_LT): None}}),
    ("offset non-invertible (-> window MV at offset)", [_OFF], {"offsets": {_rn(_OFF): -86400}}),
    ("series snapshot (step==interval scalar)", [_SER], {"series": {_rn(_SER): [86400, 86400, 5]}}),
    ("secondary-key (no cumulative; all per-window MVs)", [_agg("sum", 1), _agg("count", 1)], {"secondary_key": "ad_id"}),
]


@pytest.mark.parametrize("label,aggs,carriers", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_online_mvs_equals_desired_online_mvs(label, aggs, carriers):
    kw = dict(aggregation_interval=INTERVAL, agg_params=None, secondary_key=carriers.get("secondary_key"),
              offsets=carriers.get("offsets", {}), lifetimes=carriers.get("lifetimes", {}),
              series=carriers.get("series", {}))
    legacy = _desired_online_mvs(PROJECT, VIEW, CI, aggs, TILES, **kw)
    plan = TilePlan.from_inputs(PROJECT, VIEW, CI, aggs, TILES, **kw).online_mvs()
    # byte-identical: same names, same SELECT strings, SAME insertion order
    assert list(plan.items()) == list(legacy.items()), label


def test_filtered_aggregation_shares_one_tile_scan():
    # total + DEBIT counts on the same column share ONE tiles MV: the total is the bare partial, the filtered
    # count is a FILTER(WHERE ...) partial whose name carries a predicate hash (so they don't collide). The
    # recombine references the hashed name, and the offline read references the IDENTICAL name (online==offline).
    from feast.infra.compute_engines.risingwave.sql_builders import build_offline_tile_pit_query

    total = _agg("count", 1, "total_count")
    debit = _agg("count", 1, "debit_count")
    pred = "transaction_code = 'DEBIT'"
    filters = {_rn(debit): pred}
    plan = TilePlan.from_inputs(PROJECT, VIEW, CI, [total, debit], TILES, aggregation_interval=INTERVAL,
                                filters=filters, flavor="batch", source_relation="src")
    _, tiles_sql = plan.tiles_ddl()
    assert "count(amount) AS count_amount" in tiles_sql            # total: bare partial
    assert f"count(amount) FILTER (WHERE {pred}) AS count_amount_f" in tiles_sql  # DEBIT: filtered, hashed
    # both counts are invertible -> ONE cumulative MV carrying BOTH running totals (distinct columns)
    cum_sql = next(s for n, s in plan.online_mvs().items() if "_cum" in n)
    assert "AS cum_count_amount," in cum_sql and "AS cum_count_amount_f" in cum_sql
    # the offline read references the SAME filtered partial name -> online == offline by construction
    off = build_offline_tile_pit_query(
        "SELECT 'u1' AS user_id, now() AS event_timestamp", ["user_id", "event_timestamp"],
        "event_timestamp", tiles_relation=TILES, column_info=CI, aggregations=[total, debit],
        aggregation_interval=INTERVAL, filters=filters,
    )
    assert "count_amount_f" in off and "AS debit_count" in off and "AS total_count" in off


def test_desired_online_mvs_threads_filters_into_cumulative():
    # regression: _desired_online_mvs is the provisioning/reconcile entry point and must thread `filters`
    # into the plan, not just TilePlan.from_inputs. If it drops them, a filtered view provisions a
    # cumulative MV WITHOUT the filtered partial while the read references it -> "column not found" at serve.
    total = _agg("count", 1, "total_count")
    debit = _agg("count", 1, "debit_count")
    filters = {_rn(debit): "transaction_code = 'DEBIT'"}
    mvs = _desired_online_mvs(PROJECT, VIEW, CI, [total, debit], TILES, aggregation_interval=INTERVAL,
                              agg_params=None, secondary_key=None, offsets={}, lifetimes={}, series={},
                              filters=filters)
    cum_sql = next(s for n, s in mvs.items() if "_cum" in n)
    assert "AS cum_count_amount," in cum_sql and "AS cum_count_amount_f" in cum_sql


@pytest.mark.parametrize("flavor,builder", [("batch", build_batch_tile_select), ("streaming", build_streaming_tile_select)])
def test_tiles_ddl_equals_builder(flavor, builder):
    aggs = [_agg("sum", 1), _agg("mean", 1)]
    plan = TilePlan.from_inputs(PROJECT, VIEW, CI, aggs, TILES, aggregation_interval=INTERVAL,
                                flavor=flavor, source_relation="src")
    name, sql = plan.tiles_ddl()
    assert name == TILES
    assert sql == builder(CI, aggs, "src", aggregation_interval=INTERVAL)
