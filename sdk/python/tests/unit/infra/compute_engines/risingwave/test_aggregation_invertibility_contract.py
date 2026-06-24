"""The aggregation invertibility contract — the load-bearing classification that decides how an aggregation
is served from tiles: INVERTIBLE (an Abelian group / deletable — sum, count, mean, variance, stddev) is
served by 2-point subtraction over ONE cumulative MV; NON-INVERTIBLE (a monoid with no inverse — min, max,
count_distinct, approx_*, first/last sequences) keeps a per-(window, offset) / lifetime roll-up MV.

This mirrors Chronon's Operation deletable-vs-monoid split (api.thrift Operation; SimpleAggregator.delete /
isDeletable). The classification is declared in TWO places — ``tiling._INVERTIBLE_TILE_FN`` (the positive set:
which functions admit cumulative subtraction) and ``sql_builders.MONOID_FUNCTIONS`` (the non-deletable set) —
so these tests pin that the two cannot DRIFT and that no supported function is ever left UNCLASSIFIED. A new
aggregation function added to ``SUPPORTED_AGG_FUNCTIONS`` without classifying its invertibility fails here.
"""

import datetime as dt

from feast.aggregation import Aggregation
from feast.infra.compute_engines.risingwave.sql_builders import (
    MONOID_FUNCTIONS,
    SUPPORTED_AGG_FUNCTIONS,
)
from feast.infra.compute_engines.risingwave.tiling import (
    _INVERTIBLE_TILE_FN,
    _TILE_SUPPORTED_FN,
    is_invertible_agg,
)

# The expected classification, pinned literally so a change is deliberate + reviewed (it changes which
# aggregations are served by cumulative subtraction vs a per-window roll-up — a correctness-critical routing).
_EXPECTED_INVERTIBLE = {"sum", "count", "mean", "var_pop", "var_samp", "stddev_pop", "stddev_samp"}
_EXPECTED_MONOID = {"min", "max", "count_distinct", "approx_count_distinct", "approx_percentile",
                    "first", "last", "first_distinct", "last_distinct"}


def test_invertible_and_monoid_partition_supported():
    # Every supported function is classified EXACTLY once: invertible and monoid are disjoint, and together
    # they cover the whole supported set — no function is both, none is left out.
    assert _INVERTIBLE_TILE_FN.isdisjoint(MONOID_FUNCTIONS)
    assert set(_INVERTIBLE_TILE_FN) | set(MONOID_FUNCTIONS) == set(SUPPORTED_AGG_FUNCTIONS)


def test_classification_membership_is_pinned():
    assert set(_INVERTIBLE_TILE_FN) == _EXPECTED_INVERTIBLE
    assert set(MONOID_FUNCTIONS) == _EXPECTED_MONOID


def test_is_invertible_agg_is_exactly_the_non_monoid_supported():
    # The cross-file consistency invariant: for every supported function, is_invertible_agg() agrees with the
    # complement of MONOID_FUNCTIONS. If the two declarations drift, this catches it function-by-function.
    for fn in SUPPORTED_AGG_FUNCTIONS:
        agg = Aggregation(column="x", function=fn, time_window=dt.timedelta(days=1))
        assert is_invertible_agg(agg) is (fn not in MONOID_FUNCTIONS), fn


def test_every_tile_supported_function_routes_to_exactly_one_path():
    # Within the tile model, every tile-supported function must route to cumulative (invertible) XOR roll-up
    # (non-invertible) — there is no third path, so an unclassified tile-supported function is a bug.
    for fn in _TILE_SUPPORTED_FN:
        agg = Aggregation(column="x", function=fn, time_window=dt.timedelta(days=1))
        invertible = is_invertible_agg(agg)
        assert invertible == (fn not in MONOID_FUNCTIONS), fn
