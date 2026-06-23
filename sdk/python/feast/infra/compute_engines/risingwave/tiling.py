"""The partial-aggregate IR algebra for the RisingWave engine's tile model.

The tile model materializes per-(entity, tile) PARTIALS that recombine across the tiles in
a window. This module is the single home of that algebra: the tile-family classification
(which functions decompose into which mergeable partials), the per-tile partial columns an
aggregation needs, and the retrieval-time recombine EXPRESSIONS (interval, cumulative
2-point subtraction, and window-series) — WITHOUT the SQL that bolts those expressions onto
a SELECT (that lives in ``sql_builders``). Depends only on ``feast.Aggregation`` and the
leaf carriers module; the SQL builders depend on it.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from feast.aggregation import Aggregation

# Sequence (Array-valued) aggregates -> the ORDER BY direction of their array_agg: last* keep the
# n MOST-RECENT (event-time DESC), first* the n EARLIEST (ASC). The _distinct variants wrap
# array_distinct (which preserves first occurrence, so most-recent/earliest distinct is kept).
_SEQUENCE_ORDER = {
    "last": "DESC",
    "last_distinct": "DESC",
    "first": "ASC",
    "first_distinct": "ASC",
}

# The tile model materializes per-(entity, tile) PARTIALS that recombine additively across the tiles
# in a window. WINDOW-INDEPENDENT (one tile set reused across every time-window): a partial is
# keyed by (function-family, column), NOT by window — ``sum_amount``, ``count_amount``, ``min_amount``,
# ``max_amount``, ``sumsq_amount``. So a ``sum(amount)`` over 3d and another over 30d SHARE the one
# ``sum_amount`` tile partial, and ``mean(amount)`` reuses the same ``sum_amount`` + ``count_amount``.
# Two families — the partial-aggregate IR decomposition. This is the SAME algebra Feast's tiling defines
# in ``feast.aggregation.tiling.base.get_ir_metadata_for_aggregation`` (the canonical reference): ADDITIVE
# == Feast's "algebraic", COMPOSITE == Feast's "avg"/"holistic". We keep the SQL rendering here (Feast's
# orchestrator is pure-pandas — not SQL-pushdownable), but a conformance test pins these families to
# Feast's IR metadata so the shared algebra cannot drift (see test_tile_partials_conform_to_feast_tiling).
#   ADDITIVE — one partial == the aggregate; recombine: sum/sum/min/max (count rolls up by SUMMING
#     per-tile counts).
#   COMPOSITE — the aggregate is NOT additive, but decomposes into additive partials that DO merge
#     and a recombine formula (mean = {sum, count}; variance via {sum, sumsq, count},
#     var = (Σx² − (Σx)²/n)/n) — matching Feast's "avg" IR {sum,count} and "holistic" IR {sum,count,sum_sq}.
# Each aggregation's OUTPUT column is still its per-window ``resolved_name`` (e.g. ``sum_amount_259200s``);
# only the stored tile partials are window-independent. count_distinct/sequence are OUR extension BEYOND
# Feast's tiling (it rejects count_distinct and omits sequence); approx has no mergeable sketch — rejected.
_ADDITIVE_TILE_FN = frozenset({"sum", "count", "min", "max"})
_COMPOSITE_TILE_FN = frozenset({"mean", "var_pop", "var_samp", "stddev_pop", "stddev_samp"})
# SET — exact count_distinct: the per-tile partial is the tile's DISTINCT SET (an array), and the
# rollup unions the sets (concat the per-tile arrays, dedup, count). Set union is a commutative
# monoid with an additive partial, so it tiles correctly — unlike approx_count_distinct, whose HLL
# sketch RisingWave does not expose as a mergeable value. CAVEAT: a high-cardinality column makes the
# per-tile distinct array (hence the tiles MV state) large; this is the inherent cost of EXACT distinct.
_SET_TILE_FN = frozenset({"count_distinct"})
# SEQUENCE — bounded last/first(n): the per-tile partial is the tile's OWN top-n ordered array
# (bounded to n), and the rollup concatenates the per-tile arrays in tile_end order (array_flatten),
# re-slicing to n. Each tile's top-n contains every value in the global top-n for that tile's slot, so
# the union's top-n is the window's top-n. n is per-aggregation (it shapes the partial), so it threads
# from agg_params into the partial AND its name.
_SEQUENCE_TILE_FN = frozenset(_SEQUENCE_ORDER)
_TILE_SUPPORTED_FN = (
    _ADDITIVE_TILE_FN | _COMPOSITE_TILE_FN | _SET_TILE_FN | _SEQUENCE_TILE_FN
)

# INVERTIBLE — the aggregations whose windowed value can be served by 2-POINT SUBTRACTION over a
# CUMULATIVE (running-total) tile MV: windowed = cum_at_T - cum_at_(T - window). Only functions whose
# tile partials are additive AND have a subtractive inverse qualify: sum/count, and the composite
# mean/var/stddev (built from the invertible sum/count/sumsq partials). min/max are additive-tile but
# NOT invertible (you cannot un-max a value out of a running max); count_distinct (set union) and
# sequence (top-n array) have no subtractive inverse either. Non-invertible aggregations keep the
# interval-tile range read. This is the principled boundary of the unified v2 serving model — it is a
# mathematical fact (existence of an inverse), not an implementation choice, so it is stable.
_INVERTIBLE_TILE_FN = _COMPOSITE_TILE_FN | frozenset({"sum", "count"})


def is_invertible_agg(agg: Aggregation) -> bool:
    """True if ``agg`` can be served by cumulative 2-point subtraction (see ``_INVERTIBLE_TILE_FN``)."""
    return agg.function in _INVERTIBLE_TILE_FN


def _sequence_n(agg: Aggregation, agg_params: Optional[Dict[str, List[float]]]) -> int:
    """The n (count of values to keep) for a sequence aggregate, from the out-of-band agg_params keyed
    by resolved_name. Raises if absent — a sequence aggregate cannot tile (or render) without its n."""
    key = agg.resolved_name(agg.time_window)
    params = (agg_params or {}).get(key)
    if not params:
        raise ValueError(
            f"{agg.function} on {agg.column!r} needs an n parameter (the number of values to keep); "
            f"none was provided in agg_params (keyed by resolved_name {key!r})."
        )
    return int(params[0])


def _filter_hash(canonical_predicate: str) -> str:
    """A deterministic 8-hex digest of an ALREADY-canonicalized FILTER predicate, used as a filtered
    partial's column-name suffix. Must be stable run-to-run: the reconcile catalog keys on the rendered
    SELECT, so a non-deterministic name would needlessly re-materialize the MV. The predicate is
    canonicalized upstream (by DataFusion, via the ``feast_rw_agg_filter`` carrier); we never re-parse it."""
    return hashlib.sha1(canonical_predicate.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class Partial:
    """A WINDOW-INDEPENDENT per-tile partial column — the atom of the tile model. ``function`` is the
    PARTIAL-level family token (``sum``/``count``/``min``/``max``/``sumsq``/``distinct``, or a sequence fn),
    ``column`` the source column, ``n`` the sequence top-n (shapes the slice + the name), ``filter`` an
    optional canonical ``FILTER`` predicate (over static source columns).

    ``column_name`` is the ONE namer every writer (the tiles MV) AND every reader (the recombines) use, so a
    partial's stored name and every reference to it cannot drift — the invariant that makes the filtered
    variant safe. With ``filter=None`` it returns the legacy bare name byte-identically (so existing views
    emit unchanged SQL); a filter appends ``_f<hash>`` so total/DEBIT/QR partials on one column stay distinct.
    """

    function: str
    column: str
    n: Optional[int] = None
    filter: Optional[str] = None

    def column_name(self) -> str:
        if self.function in _SEQUENCE_ORDER:
            base = f"{self.function}_{self.column}_{self.n}"
        elif self.function == "distinct":
            base = f"distinct_{self.column}"
        else:  # sum / count / min / max / sumsq
            base = f"{self.function}_{self.column}"
        return base if self.filter is None else f"{base}_f{_filter_hash(self.filter)}"

    def _filter_clause(self, intrinsic: str = "") -> str:
        # the user predicate (when set) is ANDed with any INTRINSIC partial filter (count_distinct's
        # NOT NULL), so a filtered count_distinct would compose correctly; both absent => no FILTER clause.
        preds = [p for p in (intrinsic, self.filter) if p]
        return f" FILTER (WHERE {' AND '.join(preds)})" if preds else ""

    def materialize_sql(self, ts_col: Optional[str] = None) -> str:
        """The per-tile aggregate that builds this partial column (the value, no alias)."""
        col, fn = self.column, self.function
        if fn in _SEQUENCE_ORDER:
            # the tile's OWN top-n ordered array (bounded); n is part of the partial.
            ordered = f"array_agg({col} ORDER BY {ts_col} {_SEQUENCE_ORDER[fn]})"
            if fn.endswith("_distinct"):
                ordered = f"array_distinct({ordered})"
            return f"({ordered})[1:{self.n}]{self._filter_clause()}"
        if fn == "distinct":
            # the tile's DISTINCT SET; NULLs filtered so the union+count matches count(distinct <col>).
            return f"array_agg(DISTINCT {col}){self._filter_clause(f'{col} IS NOT NULL')}"
        if fn == "sumsq":
            return f"sum({col} * {col}){self._filter_clause()}"
        return f"{fn}({col}){self._filter_clause()}"  # sum / count / min / max

    @staticmethod
    def from_aggregation(
        agg: Aggregation,
        ts_col: Optional[str] = None,
        agg_params: Optional[Dict[str, List[float]]] = None,
        filter: Optional[str] = None,
    ) -> List["Partial"]:
        """The window-independent partials one aggregation needs (the partial-aggregate IR decomposition):
        additive functions need ONE partial; composite (mean/var/stddev) need the additive sub-partials that
        merge across tiles; count_distinct stores the tile's distinct set; sequence stores the bounded
        top-n. An optional ``filter`` rides every partial of a filtered aggregation."""
        fn, col = agg.function, agg.column
        if fn == "sum":
            return [Partial("sum", col, filter=filter)]
        if fn == "count":
            return [Partial("count", col, filter=filter)]
        if fn in {"min", "max"}:
            return [Partial(fn, col, filter=filter)]
        if fn in _SEQUENCE_ORDER:
            return [Partial(fn, col, n=_sequence_n(agg, agg_params), filter=filter)]
        if fn == "count_distinct":
            return [Partial("distinct", col, filter=filter)]
        partials = [Partial("sum", col, filter=filter), Partial("count", col, filter=filter)]
        if fn in {"var_pop", "var_samp", "stddev_pop", "stddev_samp"}:
            partials.append(Partial("sumsq", col, filter=filter))
        return partials


def _partials_for(
    agg: Aggregation,
    ts_col: Optional[str] = None,
    agg_params: Optional[Dict[str, List[float]]] = None,
    agg_filter: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """The WINDOW-INDEPENDENT per-tile partial columns (name, SQL aggregate) one aggregation needs — a thin
    shim over ``Partial.from_aggregation`` (the single namer/renderer). Named by (function-family, column)
    so multiple windows / functions on the same column share a partial; a filtered aggregation's partials
    carry ``agg_filter`` (a canonical FILTER predicate), which hashes into the name so they stay distinct."""
    return [
        (p.column_name(), p.materialize_sql(ts_col))
        for p in Partial.from_aggregation(agg, ts_col, agg_params, filter=agg_filter)
    ]


def _view_partials(
    aggregations: List[Aggregation],
    ts_col: Optional[str] = None,
    agg_params: Optional[Dict[str, List[float]]] = None,
    filters: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """The deduped union of every aggregation's window-independent partials = the tiles MV's partial
    columns. Dedup is by partial NAME, and is ASSERT-EQUAL: two aggregations may share a partial only if
    they render the IDENTICAL aggregate SQL for that name. A same-name/different-SQL clash raises rather
    than silently dropping one — the collision a filtered partial would cause if its name did not carry the
    predicate hash (``Partial.column_name``). Insertion order is preserved (the tiles MV column order)."""
    out: dict = {}
    for a in aggregations:
        agg_filter = (filters or {}).get(a.resolved_name(a.time_window))
        for name, sql in _partials_for(a, ts_col, agg_params, agg_filter=agg_filter):
            if name in out and out[name] != sql:
                raise ValueError(
                    f"tile partial name collision: {name!r} maps to two different aggregates "
                    f"({out[name]!r} vs {sql!r}) — a filtered partial must carry its predicate hash."
                )
            out[name] = sql
    return list(out.items())


def _tile_recombine(
    agg: Aggregation,
    *,
    prefix: str = "",
    partial_filter: Optional[str] = None,
    output_prefix: str = "",
    agg_params: Optional[Dict[str, List[float]]] = None,
    agg_filter: Optional[str] = None,
) -> str:
    """The retrieval-time recombine for one aggregation: an expression over the window-independent
    tile partials aliased to the FINAL per-window ``resolved_name``. ``prefix`` qualifies the partial
    columns for a joined relation (``"t."`` in the offline PIT range-join). ``partial_filter`` is a SQL
    predicate that narrows the tiles to THIS aggregation's window (``CASE WHEN <filter> THEN p END``) —
    used when one query rolls up several windows over a shared join (the multi-window offline PIT); when
    None the surrounding ``WHERE`` already bounds the window (the per-window online/floored rollups).
    ``output_prefix`` qualifies the OUTPUT alias as ``"{output_prefix}__{resolved_name}"`` for an offline
    read with full_feature_names; the online/materialize rollups pass none and keep the bare name."""
    out = agg.resolved_name(agg.time_window)
    if output_prefix:
        out = f'"{output_prefix}__{out}"'
    expr = _recombine_expr(
        agg, prefix=prefix, partial_filter=partial_filter, agg_params=agg_params, agg_filter=agg_filter
    )
    return f"{expr} AS {out}"


def _composite_finalize(fn: str, *, sm: str, cnt: str, sumsq: str) -> str:
    """The mean / variance / stddev finalize over already-built sum, count, and sum-of-squares operand
    expressions — the ONE home of the composite recombine algebra. The three call sites differ ONLY in how
    they build the operands: ``_recombine_expr`` wraps each tile partial in an aggregate, ``_cumulative_
    recombine_expr`` passes a hi-lo 2-point delta, and ``_tile_value_expr`` passes the bare single-tile
    partial. Keeping the formula here is what guarantees online == offline — the interval recombine, the
    cumulative subtraction, and the materialized series snapshot cannot drift apart. ``sumsq`` is unused for
    mean (pass "").

    variance/stddev: (Σx² − (Σx)²/n) / n (population) or / (n−1) (sample); stddev = sqrt(var).
    GREATEST(..., 0) clamps the centered sum-of-squared-deviations to non-negative: the single-pass
    computational form is catastrophic-cancellation-prone, so over large-magnitude values summed in
    RisingWave's nondeterministic parallel order the residual can round slightly NEGATIVE — an impossible
    variance, and (since RW's sqrt ERRORS on negative input) a hard query failure for stddev. RisingWave's
    OWN native var/stddev plan wraps the identical expression in Greatest(_, 0), so we match it."""
    if fn == "mean":
        return f"{sm} / NULLIF({cnt}, 0)"
    centered = f"GREATEST({sumsq} - {sm} * {sm} / NULLIF({cnt}, 0), 0)"
    denom = f"NULLIF({cnt} - 1, 0)" if fn.endswith("_samp") else f"NULLIF({cnt}, 0)"
    var = f"{centered} / {denom}"
    return f"sqrt({var})" if fn.startswith("stddev") else var


def _recombine_expr(
    agg: Aggregation,
    *,
    prefix: str = "",
    partial_filter: Optional[str] = None,
    agg_params: Optional[Dict[str, List[float]]] = None,
    agg_filter: Optional[str] = None,
) -> str:
    """The bare retrieval-time recombine EXPRESSION for one aggregation over the window-independent tile
    partials — WITHOUT the output alias. The single home of the per-function recombine: ``_tile_recombine``
    aliases it to the per-window ``resolved_name``, and ``_series_recombine`` builds an ``ARRAY[...]`` of
    these (one per series step), so the online == offline recombine is never re-implemented per caller.
    See ``_tile_recombine`` for the ``prefix`` / ``partial_filter`` semantics."""
    col, fn = agg.column, agg.function

    def merged(kind: str, op: str = "sum") -> str:
        p = f"{prefix}{Partial(kind, col, filter=agg_filter).column_name()}"
        inner = f"CASE WHEN {partial_filter} THEN {p} END" if partial_filter else p
        return f"{op}({inner})"

    if fn == "sum":
        return merged("sum")
    if fn == "count":  # count recombines by SUMMING per-tile counts
        return merged("count")
    if fn in {"min", "max"}:
        return merged(fn, fn)
    if fn == "count_distinct":
        # Union the per-tile distinct sets and count: concat the tile arrays (array_flatten skips the
        # NULL inner arrays a multi-window CASE produces), dedup, count. NULLIF(_, 0) maps an empty
        # window (no tiles, or only NULL values) to NULL, so the offline LEFT-JOIN result matches the
        # online MV — where such an entity is simply absent.
        sets = merged("distinct", "array_agg")
        return f"NULLIF(cardinality(array_distinct(array_flatten({sets}))), 0)"
    if fn in _SEQUENCE_ORDER:
        # Concat the per-tile top-n arrays in tile_end order (array_flatten preserves it AND skips the
        # NULL inner arrays a multi-window CASE produces), then slice to n. Each tile array is already
        # ordered within the tile, so the flattened order is the global event-time order; the _distinct
        # variants dedup again across tiles (a value may repeat in two tiles).
        n = _sequence_n(agg, agg_params)
        p = f"{prefix}{Partial(fn, col, n=n, filter=agg_filter).column_name()}"
        inner = f"CASE WHEN {partial_filter} THEN {p} END" if partial_filter else p
        flat = f"array_flatten(array_agg({inner} ORDER BY {prefix}tile_end {_SEQUENCE_ORDER[fn]}))"
        if fn.endswith("_distinct"):
            flat = f"array_distinct({flat})"
        return f"({flat})[1:{n}]"
    # mean/var/stddev: one finalize home (_composite_finalize) so this interval recombine, the cumulative
    # 2-point subtraction, and the materialized series snapshot cannot drift apart. mean needs sum/count
    # only; var/stddev also need sumsq (a mean-only view has no sumsq partial, so do not build one).
    sumsq = "" if fn == "mean" else merged("sumsq")
    return _composite_finalize(fn, sm=merged("sum"), cnt=merged("count"), sumsq=sumsq)


def _cumulative_recombine_expr(
    agg: Aggregation,
    *,
    hi: str = "hi",
    lo: Optional[str] = "lo",
    agg_filter: Optional[str] = None,
) -> str:
    """The windowed recombine for an INVERTIBLE aggregation by 2-POINT SUBTRACTION over the cumulative-tile
    MV, from two cumulative rows: ``hi`` = cumulative at the latest tile_end <= window END, ``lo`` =
    cumulative at the latest tile_end <= window END - window_size (``lo=None`` for a lifetime /
    cumulative-to-end read — no lower bound). The windowed IR is Δ = hi.cum_X - lo.cum_X, recombined
    EXACTLY as ``_recombine_expr`` recombines the per-tile partials — because Δ(running total) over
    (end-W, end] equals the sum of the per-tile partials in (end-W, end], value-for-value. Emits NULL when
    the window holds no tiles (Δ cum_ntiles == 0), matching the offline PIT's empty-window NULL (the
    decisive parity case). Go/Python serving readers MUST mirror these expressions verbatim so online ==
    offline. Non-invertible aggregations are rejected (they keep the interval read)."""
    if not is_invertible_agg(agg):
        raise ValueError(
            f"{agg.function} on {agg.column!r} is not invertible; serve it from the interval tiles, "
            f"not by cumulative subtraction."
        )
    col, fn = agg.column, agg.function

    def delta(kind: str) -> str:
        # COALESCE each cumulative to 0: a boundary asof that finds NO tile (the entity's history begins
        # inside the window, or there are no tiles at all) yields a NULL cumulative = 0 contribution. The
        # cumulative column is ``cum_`` + the partial's column_name(), so a filtered partial's hashed name
        # flows through here too (the single-namer invariant).
        name = Partial(kind, col, filter=agg_filter).column_name()
        hi_c = f"COALESCE({hi}.cum_{name}, 0)"
        return hi_c if lo is None else f"({hi_c} - COALESCE({lo}.cum_{name}, 0))"

    hi_n = f"COALESCE({hi}.cum_ntiles, 0)"
    dntiles = hi_n if lo is None else f"({hi_n} - COALESCE({lo}.cum_ntiles, 0))"

    if fn == "sum":
        value = delta("sum")
    elif fn == "count":
        value = delta("count")
    else:  # mean / var_pop / var_samp / stddev_pop / stddev_samp — the shared finalize home, over deltas
        sumsq = "" if fn == "mean" else delta("sumsq")
        value = _composite_finalize(fn, sm=delta("sum"), cnt=delta("count"), sumsq=sumsq)
    # Empty window (no tiles in range) -> NULL, matching the offline PIT — the common case (an entity with
    # no recent events). One narrow divergence remains for SUM: a window that HAS tiles but whose every
    # aggregation-input value is NULL gives Δsum == 0 here, while the offline sum over only-NULL partials is
    # NULL. It takes a window whose sole events all carry a NULL input value — rare, but real; count/mean/var
    # agree in that case. Exact parity there would need a running value-count for sum-only columns to gate on;
    # left as-is because the case is rare and 0 is a defensible sum of no (present) values.
    return f"CASE WHEN {dntiles} = 0 THEN NULL ELSE {value} END"


def cumulative_tile_recombine(
    agg: Aggregation,
    *,
    hi: str = "hi",
    lo: Optional[str] = "lo",
    output_prefix: str = "",
    agg_filter: Optional[str] = None,
) -> str:
    """``_cumulative_recombine_expr`` aliased to the aggregation's per-window ``resolved_name`` (or
    ``{output_prefix}__{resolved_name}`` for a full-feature-names offline read) — the cumulative twin of
    ``_tile_recombine``."""
    out = agg.resolved_name(agg.time_window)
    if output_prefix:
        out = f'"{output_prefix}__{out}"'
    return f"{_cumulative_recombine_expr(agg, hi=hi, lo=lo, agg_filter=agg_filter)} AS {out}"


def _series_recombine(
    agg: Aggregation,
    *,
    end_expr: str,
    window_secs: int,
    step_secs: int,
    length: int,
    prefix: str = "",
    output_prefix: str = "",
    agg_params: Optional[Dict[str, List[float]]] = None,
    agg_filter: Optional[str] = None,
) -> str:
    """The retrieval-time recombine for a window-SERIES: an ``ARRAY`` of ``length`` per-window recombines
    over the ONE shared tile set, one element per step, ordered OLDEST window FIRST (the earliest->latest
    array contract). Each element is the same per-window recombine ``_tile_recombine`` emits, narrowed by a
    ``CASE`` to that step's window ``(end - W - i*step, end - i*step]`` off ``end_expr`` — i.e. the L-fold
    copy of the offset ``CASE``. An empty step (no tiles in range) recombines to NULL (a sum over no rows,
    or the NULLIF on count_distinct/mean), so its array element is NULL — matching the online assembled
    array element-for-element, and established feature stores' empty-window = None."""
    out = agg.resolved_name(agg.time_window)
    if output_prefix:
        out = f'"{output_prefix}__{out}"'
    elements = []
    for steps_back in range(length - 1, -1, -1):  # oldest window first (largest shift into the past first)
        lo = window_secs + steps_back * step_secs
        hi = steps_back * step_secs
        pf = f"{prefix}tile_end > {end_expr} - INTERVAL '{lo}' SECOND"
        # The newest step (hi == 0) needs no upper CASE bound — the join's `<= end` already caps it; every
        # older step's upper edge sits below `end`, so it gets its own explicit upper bound.
        if hi:
            pf += f" AND {prefix}tile_end <= {end_expr} - INTERVAL '{hi}' SECOND"
        elements.append(_recombine_expr(
            agg, prefix=prefix, partial_filter=pf, agg_params=agg_params, agg_filter=agg_filter
        ))
    return f"ARRAY[{', '.join(elements)}] AS {out}"


# The scalar finalizable functions a per-entity series SNAPSHOT can carry: sum/count/min/max (additive)
# and mean/var/stddev (composite). count_distinct/sequence are array-valued (the per-tile value is itself
# an array), so they are not snapshotted and keep the read-time single-scan.
_SNAPSHOT_SERIES_FN = _ADDITIVE_TILE_FN | _COMPOSITE_TILE_FN


def _tile_value_expr(
    agg: Aggregation,
    *,
    prefix: str = "",
    agg_params: Optional[Dict[str, List[float]]] = None,
    agg_filter: Optional[str] = None,
) -> str:
    """The FINALIZED feature value of ONE tile, computed from that tile's stored partials WITHOUT an
    aggregate wrapper — for a window-series whose step equals the tile interval, where each step is exactly
    one tile. It is ``_recombine_expr`` with the cross-tile merge collapsed to identity (a single tile has
    nothing to sum across), so a materialized last-L snapshot carries the SAME value the offline single-scan
    recombines per step. Only scalar finalizable functions are covered (see ``_SNAPSHOT_SERIES_FN``);
    count_distinct/sequence are array-valued and not snapshotted."""
    col, fn = agg.column, agg.function
    if fn in {"sum", "count", "min", "max"}:
        return f"{prefix}{Partial(fn, col, filter=agg_filter).column_name()}"
    # mean/var/stddev from this ONE tile's partials, via the shared finalize home — the merge collapsed to
    # identity (no cross-tile aggregate), so it equals _recombine_expr over a single tile by construction.
    sumsq = "" if fn == "mean" else f"{prefix}{Partial('sumsq', col, filter=agg_filter).column_name()}"
    return _composite_finalize(
        fn,
        sm=f"{prefix}{Partial('sum', col, filter=agg_filter).column_name()}",
        cnt=f"{prefix}{Partial('count', col, filter=agg_filter).column_name()}",
        sumsq=sumsq,
    )


def snapshot_series_aggs(
    aggregations: List[Aggregation],
    series: Optional[Dict[str, Sequence[int]]],
    interval_secs: int,
) -> List[Tuple[Aggregation, int]]:
    """The ``(aggregation, length)`` pairs a per-entity last-L snapshot MV can serve: a window-series whose
    window == step == the tile interval (so each element is exactly ONE tile) AND a scalar finalizable
    function. Excluded — and kept on the read-time single-scan: a coarser step (step > interval, whose
    step-buckets are anchored to the request frontier and so cannot be materialized frontier-agnostically),
    an overlapping window (window > step), and array-valued aggregates (count_distinct/sequence)."""
    out: List[Tuple[Aggregation, int]] = []
    for a in aggregations:
        geom = (series or {}).get(a.resolved_name(a.time_window))
        if not geom:
            continue
        w, s, length = int(geom[0]), int(geom[1]), int(geom[2])
        if w == s == interval_secs and a.function in _SNAPSHOT_SERIES_FN:
            out.append((a, length))
    return out


def _assert_tile_supported(aggregations: List[Aggregation]) -> None:
    # The tile model supports any aggregation that recombines from a mergeable per-tile partial:
    # sum/count/min/max directly, mean/var/stddev via composite partials (Chronon's IR), and exact
    # count_distinct via a per-tile distinct SET unioned at rollup. approx_count_distinct (HLL) and
    # approx_percentile have no mergeable partial RisingWave exposes across tiles — rejected.
    unsupported = sorted({a.function for a in aggregations} - _TILE_SUPPORTED_FN)
    if unsupported:
        raise ValueError(
            f"Batch tile aggregations {unsupported} are not supported: the tile model rolls up "
            f"mergeable per-tile partials, so {sorted(_TILE_SUPPORTED_FN)} work, but "
            f"approx_count_distinct/approx_percentile have no mergeable sketch across tiles."
        )


def _cumulative_partials(
    aggregations: List[Aggregation],
    ts_col: Optional[str] = None,
    agg_params: Optional[Dict[str, List[float]]] = None,
    filters: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """The running-total columns of the cumulative-tile MV: for each INVERTIBLE aggregation's
    window-independent partial (``sum_``/``count_``/``sumsq_``), a ``cum_{partial}`` whose value is a
    running total of that partial COLUMN over ``tile_end``. Reuses ``_view_partials`` (so the dedup +
    partial naming match the tiles MV exactly), filtered to invertible aggregations; non-invertible
    aggregations contribute nothing (they keep the interval read). The SQL references the tile partial
    column NAME (e.g. ``sum_amount``) — the cumulative MV reads the tiles MV, not the raw source."""
    invertible = [a for a in aggregations if is_invertible_agg(a)]
    # cum_ntiles = running count of TILE ROWS, the universal emptiness guard. A window's Δ(cum_ntiles)
    # is the number of tiles in it; Δ == 0 means NO tiles in (end-window, end], which is exactly when the
    # offline PIT (sum over zero in-window tiles) yields NULL. The serving recombine emits NULL when
    # Δ(cum_ntiles) == 0 so cumulative-subtraction matches offline on the empty window — the decisive case
    # (entity has no recent events). It needs no aggregation-specific partial, so it is always available
    # even for a pure-sum view (which has no count partial).
    cols: List[Tuple[str, str]] = [("cum_ntiles", "count(*)")]
    cols += [
        (f"cum_{name}", f"sum({name})")
        for (name, _materialize_sql) in _view_partials(invertible, ts_col, agg_params, filters)
    ]
    return cols
