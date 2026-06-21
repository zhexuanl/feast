"""RisingWave DAG nodes and the shared SQL builders that hold this engine's
correctness invariants.

Design choice: nodes are **pure SQL builders** — they never open a database
connection. Each node composes a RisingWave SQL relation string (a CTE/subquery)
and returns a ``DAGValue(data=<relation>, format=DAGFormat.RISINGWAVE,
metadata={"columns": [...]})``, flowing the column list forward through metadata
(mirroring Flink's ``_get_columns`` / ``_sql_value``). The single composed query is
executed only at the edges — the ``RisingWaveDAGRetrievalJob`` (a terminal SELECT
over pgwire) or the ``RWOutputNode`` materialize INSERT (run by the engine). This
keeps the correctness logic unit-testable without a live RisingWave, and matches a
SQL-pushdown engine.

Every SQL fragment traces to a RisingWave end-to-end example validated against a live
instance. Anything not yet validated end-to-end is marked ``UNVERIFIED`` and listed under
the unvalidated surfaces in ``README.md``.
"""

from typing import List, Optional, Tuple

import pandas as pd

from feast.aggregation import Aggregation
from feast.infra.compute_engines.risingwave.names import (
    offline_staging_name,
    source_name,
)
from feast.infra.compute_engines.dag.context import ColumnInfo, ExecutionContext
from feast.infra.compute_engines.dag.model import DAGFormat
from feast.infra.compute_engines.dag.node import DAGNode
from feast.infra.compute_engines.dag.value import DAGValue
from feast.infra.compute_engines.utils import (
    ENTITY_ROW_ID,
    ENTITY_TS_ALIAS,
    find_entity_timestamp_column,
    infer_entity_timestamp_column,
)

# Aggregation functions this engine supports, named as the user writes them (the Feast
# Aggregation.function string). Validated at apply time (build_windowed_agg_select) so an
# unsupported function fails fast with a clear error instead of reaching RisingWave as raw
# SQL and only failing at parse time. Grounded in RisingWave's streaming aggregate support:
#   - sum / count / mean (-> avg) / min / max: core streaming aggregates.
#   - stddev_pop / stddev_samp / var_pop / var_samp: RisingWave's optimizer rewrites these
#     into streaming-safe sum(x) / sum(x*x) / count primitives (logical_agg.rs:660-690), so
#     they run inside an EOWC materialized view.
#   - count_distinct (count(distinct ...)) + approx_count_distinct (HyperLogLog): streaming,
#     but monoids without an inverse (see MONOID_FUNCTIONS). NOTE: RisingWave has a known
#     crash-RECOVERY state bug for updatable approx_count_distinct — harmless for our
#     append-only EOWC model (the source never retracts), but flagged.
# Deliberately EXCLUDED — rejected at apply with a reason, not silently:
#   - first(n) / last(n) / first_distinct / last_distinct (sequence features): RisingWave has
#     no bare first()/last() aggregate; they need ordered-set / Array outputs, which are not yet supported.
#   - approx_percentile: parameterized (takes the percentile) — needs a parameter field on
#     feast.Aggregation, which has none today, so it is not yet supported.
#   - aggregation_secondary_key: produces a per-secondary-key breakdown = an Array/Map output
#     that the scalar engine and the ServingSpec wire do not carry yet, so it is not yet supported.
SUPPORTED_AGG_FUNCTIONS = frozenset(
    {
        "sum",
        "count",
        "mean",
        "min",
        "max",
        "count_distinct",
        "approx_count_distinct",
        "stddev_pop",
        "stddev_samp",
        "var_pop",
        "var_samp",
    }
)

# The non-retractable members of SUPPORTED_AGG_FUNCTIONS: monoids with no inverse, so
# RisingWave cannot incrementally *retract* them over an upsert/retractable source without a
# full per-window recompute. Mirrors Chronon's deletable (Abelian group) vs non-deletable
# (monoid) split. sum/count/mean and stddev/variance are Abelian-group
# (or decompose into sum/count), so they are retractable-safe and are NOT listed here.
MONOID_FUNCTIONS = frozenset({"min", "max", "count_distinct", "approx_count_distinct"})

# Feast Aggregation.function -> RisingWave SQL function (only names that differ).
_SQL_FUNCTION = {"mean": "avg"}

DEDUP_ROW_NUMBER = "_feast_row_number"


def _agg_expr(agg: Aggregation) -> str:
    # Output column == resolved_name(time_window) so the online MV and the offline
    # sink emit byte-identical column names — no online/offline column-name skew
    # (aggregation/__init__.py:106-118).
    out = agg.resolved_name(agg.time_window)
    if agg.function == "count_distinct":
        return f"count(distinct {agg.column}) AS {out}"
    fn = _SQL_FUNCTION.get(agg.function, agg.function)
    return f"{fn}({agg.column}) AS {out}"


def _window_relation(
    aggregations: List[Aggregation], ts_col: str, relation: str
) -> str:
    # RisingWave TUMBLE/HOP is a table function over the whole relation, so every
    # aggregation in one MV must share a single window. For HOP the 3rd arg is the
    # slide and the 4th arg is the size.
    windows = {(a.time_window, a.slide_interval) for a in aggregations}
    if len(windows) != 1:
        raise ValueError(
            "All aggregations in one RisingWave feature view must share a single "
            f"(time_window, slide_interval); got {windows}. Split differing windows "
            "into separate feature views (one materialized view per window)."
        )
    time_window, slide = next(iter(windows))
    if time_window is None:
        return relation
    size = int(time_window.total_seconds())
    if slide == time_window:
        return f"tumble({relation}, {ts_col}, INTERVAL '{size}' SECOND)"
    return (
        f"hop({relation}, {ts_col}, INTERVAL '{int(slide.total_seconds())}' SECOND, "
        f"INTERVAL '{size}' SECOND)"
    )


def build_windowed_agg_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    relation: str,
    *,
    source_is_retractable: bool,
    emit_on_close: bool,
) -> str:
    """Windowed-aggregation SELECT shared by BOTH the online MV (``engine.update``)
    and the offline backfill, so the two definitions cannot drift apart.

    ``emit_on_close`` appends ``EMIT ON WINDOW CLOSE``,
    required for online/offline consistency; it is only valid when the source has a
    WATERMARK and is append-only — that precondition is enforced by the caller
    (``engine.update``), not here.

    Supports BOTH windowed (TUMBLE/HOP) and non-windowed (plain GROUP BY)
    aggregations: unlike Flink (flink/nodes.py:536-542), RisingWave does NOT reject
    time windows — they are this engine's value-add.
    """
    if not aggregations:
        raise ValueError("build_windowed_agg_select requires at least one aggregation")

    # Apply-time allow-list: reject any unsupported function here, with a clear message,
    # instead of letting it reach RisingWave as raw SQL and fail at parse time.
    unsupported = sorted({a.function for a in aggregations} - SUPPORTED_AGG_FUNCTIONS)
    if unsupported:
        raise ValueError(
            f"Unsupported aggregation function(s) {unsupported}. The RisingWave engine "
            f"supports {sorted(SUPPORTED_AGG_FUNCTIONS)}. Sequence aggregations "
            f"(first/last), approx_percentile (parameterized), and aggregation_secondary_key "
            f"(Array output) are not yet supported."
        )

    if source_is_retractable:
        monoids = sorted({a.function for a in aggregations} & MONOID_FUNCTIONS)
        if monoids:
            raise ValueError(
                f"Aggregations {monoids} are monoids and cannot be retracted over a "
                "retractable/upsert source without a per-window recompute "
                "(monoids have no inverse, so they cannot be incrementally retracted). Use an "
                "append-only source, or only Abelian-group ops (sum/count/mean)."
            )

    keys = ", ".join(column_info.join_keys_columns)
    exprs = ", ".join(_agg_expr(a) for a in aggregations)
    windowed = aggregations[0].time_window is not None
    src = _window_relation(aggregations, column_info.timestamp_column, relation)

    if windowed:
        # window_END is the row's event timestamp: a window [t, t+w) is only knowable
        # at t+w, so an as-of (<=) join never sees a window before it closes.
        # Timestamping by window_start would leak the full
        # aggregate to label times that fall inside the still-open window.
        select = (
            f"SELECT {keys}, {exprs}, window_start, window_end "
            f"FROM {src} GROUP BY window_start, window_end, {keys}"
        )
    else:
        select = f"SELECT {keys}, {exprs} FROM {src} GROUP BY {keys}"

    if emit_on_close:
        select += " EMIT ON WINDOW CLOSE"
    return select


def build_latest_row_select(column_info: ColumnInfo, relation: str) -> str:
    """Latest-row-per-entity SELECT for a passthrough (non-aggregated) feature view's online MV: project
    the entity keys + raw feature columns + event timestamp, keeping only the newest row per entity. Shared
    by provisioning and reconcile so the two definitions cannot drift.

    A passthrough column is a raw value carried through unchanged (no aggregation, no window), so online it
    is the latest value per entity, last-write-wins by event time — RisingWave maintains
    ``ROW_NUMBER() OVER (PARTITION BY <keys> ORDER BY <ts> DESC[, <created_ts> DESC]) ... WHERE rn = 1`` as
    an incrementally-updated Group-TopN (over an append-only source it keeps only the current top row per
    key), so the MV holds exactly one row per entity. It is served by the SAME point-lookup as an
    aggregation MV (one row per entity + a timestamp column), so the online read shape does not change.

    The ORDER BY breaks ties on the created timestamp when the source defines one — the SAME order the
    offline read uses (latest event timestamp, then latest created timestamp), so two rows sharing an
    entity's newest event timestamp resolve to the SAME row online and offline (no train/serve skew on
    ties). Offline training reads the raw history with an as-of cut (the latest row at-or-before each label
    timestamp, same ordering), not this MV, since the MV holds only the current latest row, not the
    history."""
    keys = ", ".join(column_info.join_keys_columns)
    ts = column_info.timestamp_column
    created_ts = column_info.created_timestamp_column
    order_by = ", ".join(f"{col} DESC" for col in (ts, created_ts) if col)
    # Project each column once: a feature column may coincide with an entity key or the timestamp (a
    # passthrough schema can name a feature the same as a key/ts), and a duplicated output column would make
    # CREATE MATERIALIZED VIEW fail on an ambiguous column.
    projection_cols: List[str] = []
    seen: set = set()
    for col in [*column_info.join_keys_columns, *column_info.feature_cols, ts]:
        if col not in seen:
            projection_cols.append(col)
            seen.add(col)
    projection = ", ".join(projection_cols)
    return (
        f"SELECT {projection} FROM (SELECT {projection}, "
        f"ROW_NUMBER() OVER (PARTITION BY {keys} ORDER BY {order_by}) AS {DEDUP_ROW_NUMBER} "
        f"FROM {relation}) AS _ranked WHERE {DEDUP_ROW_NUMBER} = 1"
    )


def build_passthrough_pit_query(
    entity_df_sql: str,
    entity_columns: List[str],
    label_ts_column: str,
    *,
    history_relation: str,
    column_info: ColumnInfo,
    ttl_seconds: Optional[int] = None,
    full_feature_names: bool = False,
    view_name: Optional[str] = None,
) -> str:
    """Offline point-in-time training read for a passthrough feature view: for EACH entity row, the latest
    raw feature row at-or-before that row's label timestamp — the as-of cut that makes offline == the
    latest-row online MV serves. Reads the raw history relation (the view's batch source), LEFT JOIN so an
    entity row with no match still appears (NULL features), ROW_NUMBER per entity row ordered by event (then
    created) timestamp DESC, keeping rn = 1. ``ttl_seconds`` bounds the lookback (the value is valid only
    within ttl of the label; older => NULL); unset => no lower bound. Identical entity rows collapse to one
    output row (PARTITION BY the entity columns), matching the tile PIT's GROUP BY behavior.

    ``column_info.feature_cols`` is already the requested feature subset (the caller restricts it to the
    features in feature_refs). ``full_feature_names`` aliases every feature output as
    ``"{view_name}__{feature}"`` (the standard Feast offline contract); entity columns are never prefixed."""
    keys = column_info.join_keys_columns
    # A passthrough feature must not be named like an entity-dataframe column (the timestamp/label column, a
    # join key, or another spine column): the as-of read would have to return both the feature (from the
    # history) and the entity column under ONE name. Reject it clearly rather than silently shadowing the
    # feature with the entity column (and, under full_feature_names, dropping its "{view}__{feature}" alias).
    collisions = [f for f in column_info.feature_cols if f in entity_columns]
    if collisions:
        raise ValueError(
            f"passthrough feature column(s) {collisions} collide with an entity-dataframe column; the "
            f"point-in-time read cannot return both the feature and the entity column under one name. "
            f"Rename the feature(s), or remove the column(s) from the entity dataframe."
        )
    feature_cols = column_info.feature_cols
    ts = column_info.timestamp_column
    created = column_info.created_timestamp_column
    e_cols = ", ".join(f'e."{c}"' for c in entity_columns)
    h_feats = ", ".join(f'h."{c}"' for c in feature_cols)
    join_on = " AND ".join(f'h."{k}" = e."{k}"' for k in keys)
    asof = f'h."{ts}" <= e."{label_ts_column}"'
    if ttl_seconds is not None:
        # Inclusive lower bound (a row exactly ttl before the label is still valid), matching Feast's PIT
        # template and the engine's other PIT filters.
        asof += f' AND h."{ts}" >= e."{label_ts_column}" - INTERVAL \'{ttl_seconds}\' SECOND'
    order_by = ", ".join(f'h."{c}" DESC' for c in (ts, created) if c)
    partition = ", ".join(f'e."{c}"' for c in entity_columns)
    # Entity columns project bare; feature columns take the "{view}__{feature}" alias under
    # full_feature_names. The inner subquery still emits bare names, so the alias lives only on the outer
    # projection (mirrors Feast's PIT template, which prefixes feature columns but never the entity keys).
    prefix = view_name if full_feature_names else None
    out_entity = [f'"{c}"' for c in entity_columns]
    out_feats = [
        f'"{c}" AS "{prefix}__{c}"' if prefix else f'"{c}"' for c in feature_cols
    ]
    out_cols = ", ".join([*out_entity, *out_feats])
    return (
        f"SELECT {out_cols} FROM (SELECT {e_cols}, {h_feats}, "
        f"ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY {order_by}) AS rn "
        f"FROM ({entity_df_sql}) e LEFT JOIN {history_relation} h ON {join_on} AND {asof}) AS _pit "
        f"WHERE rn = 1"
    )


# --- Batch tile aggregation (partial-aggregate tile model) --------------------------------
# A BATCH feature view materializes PARTIAL aggregates at the aggregation_interval (tiles), then
# rolls them up to the requested window AT RETRIEVAL, anchored to the request/label time. This is
# distinct from the streaming TUMBLE path above: tiles are a plain batch GROUP BY over a batch
# relation (e.g. an Iceberg source) and one fixed tile set serves any window size, sliding with the
# request time. Validated end-to-end on RisingWave v3.0.0.

# The tile model materializes per-(entity, tile) PARTIALS that recombine additively across the tiles
# in a window. WINDOW-INDEPENDENT (one tile set reused across every time-window): a partial is
# keyed by (function-family, column), NOT by window — ``sum_amount``, ``count_amount``, ``min_amount``,
# ``max_amount``, ``sumsq_amount``. So a ``sum(amount)`` over 3d and another over 30d SHARE the one
# ``sum_amount`` tile partial, and ``mean(amount)`` reuses the same ``sum_amount`` + ``count_amount``.
# Two families:
#   ADDITIVE — one partial == the aggregate; recombine: sum/sum/min/max (count rolls up by SUMMING
#     per-tile counts).
#   COMPOSITE — the aggregate is NOT additive, but decomposes into additive partials that DO merge
#     and a recombine formula (Chronon's IR: Average = {sum, count}; Variance via {sum, sumsq, count},
#     var = (Σx² − (Σx)²/n)/n).
# Each aggregation's OUTPUT column is still its per-window ``resolved_name`` (e.g. ``sum_amount_259200s``);
# only the stored tile partials are window-independent. count_distinct/approx have no safe additive
# sketch merge — still rejected.
_ADDITIVE_TILE_FN = frozenset({"sum", "count", "min", "max"})
_COMPOSITE_TILE_FN = frozenset({"mean", "var_pop", "var_samp", "stddev_pop", "stddev_samp"})
_TILE_SUPPORTED_FN = _ADDITIVE_TILE_FN | _COMPOSITE_TILE_FN


def _partials_for(agg: Aggregation) -> List[Tuple[str, str]]:
    """The WINDOW-INDEPENDENT per-tile partial columns (name, SQL aggregate) one aggregation needs.
    Named by (function-family, column) so multiple windows / functions on the same column share a
    partial. Additive functions need ONE partial; composite (mean/var/stddev) need the additive
    sub-partials that merge across tiles."""
    col, fn = agg.column, agg.function
    if fn == "sum":
        return [(f"sum_{col}", f"sum({col})")]
    if fn == "count":
        return [(f"count_{col}", f"count({col})")]
    if fn in {"min", "max"}:
        return [(f"{fn}_{col}", f"{fn}({col})")]
    partials = [(f"sum_{col}", f"sum({col})"), (f"count_{col}", f"count({col})")]
    if fn in {"var_pop", "var_samp", "stddev_pop", "stddev_samp"}:
        partials.append((f"sumsq_{col}", f"sum({col} * {col})"))
    return partials


def _view_partials(aggregations: List[Aggregation]) -> List[Tuple[str, str]]:
    """The deduped union of every aggregation's window-independent partials = the tiles MV's partial
    columns. ``setdefault`` keeps one entry per partial name (the materialize-SQL is identical for a
    given (family, column), so dedup is safe)."""
    out: dict = {}
    for a in aggregations:
        for name, sql in _partials_for(a):
            out.setdefault(name, sql)
    return list(out.items())


def _tile_recombine(
    agg: Aggregation,
    *,
    prefix: str = "",
    partial_filter: Optional[str] = None,
    output_prefix: str = "",
) -> str:
    """The retrieval-time recombine for one aggregation: an expression over the window-independent
    tile partials aliased to the FINAL per-window ``resolved_name``. ``prefix`` qualifies the partial
    columns for a joined relation (``"t."`` in the offline PIT range-join). ``partial_filter`` is a SQL
    predicate that narrows the tiles to THIS aggregation's window (``CASE WHEN <filter> THEN p END``) —
    used when one query rolls up several windows over a shared join (the multi-window offline PIT); when
    None the surrounding ``WHERE`` already bounds the window (the per-window online/floored rollups).
    ``output_prefix`` qualifies the OUTPUT alias as ``"{output_prefix}__{resolved_name}"`` for an offline
    read with full_feature_names; the online/materialize rollups pass none and keep the bare name."""
    col, fn = agg.column, agg.function
    out = agg.resolved_name(agg.time_window)
    if output_prefix:
        out = f'"{output_prefix}__{out}"'

    def merged(kind: str, op: str = "sum") -> str:
        p = f"{prefix}{kind}_{col}"
        inner = f"CASE WHEN {partial_filter} THEN {p} END" if partial_filter else p
        return f"{op}({inner})"

    if fn == "sum":
        return f"{merged('sum')} AS {out}"
    if fn == "count":  # count recombines by SUMMING per-tile counts
        return f"{merged('count')} AS {out}"
    if fn in {"min", "max"}:
        return f"{merged(fn, fn)} AS {out}"
    sm, cnt = merged("sum"), merged("count")
    if fn == "mean":
        return f"{sm} / NULLIF({cnt}, 0) AS {out}"
    # variance/stddev: (Σx² − (Σx)²/n) / n  (population) or / (n−1) (sample); stddev = sqrt(var).
    # GREATEST(..., 0) clamps the centered sum-of-squared-deviations to non-negative: the single-pass
    # computational form is catastrophic-cancellation-prone, so over large-magnitude values summed in
    # RisingWave's nondeterministic parallel order the residual can round slightly NEGATIVE — an
    # impossible variance, and (since RW's sqrt ERRORS on negative input) a hard query failure for
    # stddev. RisingWave's OWN native var/stddev plan wraps the identical expression in Greatest(_, 0)
    # (over_window_function plan output), so we match it.
    centered = f"GREATEST({merged('sumsq')} - {sm} * {sm} / NULLIF({cnt}, 0), 0)"
    denom = f"NULLIF({cnt} - 1, 0)" if fn.endswith("_samp") else f"NULLIF({cnt}, 0)"
    var = f"{centered} / {denom}"
    return f"sqrt({var}) AS {out}" if fn.startswith("stddev") else f"{var} AS {out}"

# aggregation_interval (the tile size) -> RisingWave date_trunc unit. Standard units only for now
# (date_trunc only supports these units); arbitrary intervals (e.g. 15min) need epoch-bucketing.
_TILE_INTERVAL_UNIT = {3600: "hour", 86400: "day", 604800: "week"}


def _tile_unit(aggregation_interval) -> str:
    unit = _TILE_INTERVAL_UNIT.get(int(aggregation_interval.total_seconds()))
    if unit is None:
        raise ValueError(
            f"aggregation_interval {aggregation_interval} is not supported yet: the batch tile "
            f"builder buckets with date_trunc, so it must be 1 "
            f"{'/'.join(sorted(_TILE_INTERVAL_UNIT.values()))}. Arbitrary intervals need "
            f"epoch-bucketing, which is not yet supported."
        )
    return unit


def _assert_tile_supported(aggregations: List[Aggregation]) -> None:
    # The tile model supports any aggregation that recombines from additive partials: sum/count/min/max
    # directly, and mean/var/stddev via composite partials (Chronon's IR). count_distinct/approx have
    # no safe additive sketch merge across tiles (they are monoids with no inverse) — rejected.
    unsupported = sorted({a.function for a in aggregations} - _TILE_SUPPORTED_FN)
    if unsupported:
        raise ValueError(
            f"Batch tile aggregations {unsupported} are not supported: the tile model rolls up "
            f"additive per-tile partials, so {sorted(_TILE_SUPPORTED_FN)} work, but "
            f"count_distinct/approx_count_distinct have no safe sketch merge across tiles."
        )


def _single_window_secs(aggregations: List[Aggregation]) -> int:
    windows = {a.time_window for a in aggregations}
    if len(windows) != 1 or None in windows:
        raise ValueError(
            f"tile rollup needs a single non-null time_window; got {windows} "
            f"(this builder does not support multiple windows from one tile set)."
        )
    return int(next(iter(windows)).total_seconds())


def _assert_window_multiple_of_interval(window_secs: int, aggregation_interval) -> None:
    # The window is a COUNT of tiles, so it must be a whole number of aggregation_intervals. This is
    # also what makes the online now()-anchored rollup equal the offline floor-anchored rollup:
    # for interval-boundary tiles, (now - W, now] selects the SAME tiles as
    # (floor(now, interval) - W, floor(now, interval)] only when W is a multiple of the interval.
    interval_secs = int(aggregation_interval.total_seconds())
    # A zero/negative window is not None (so the None guards miss it) yet 0 % interval == 0, so it would
    # slip through to emit an always-empty (end, end] range -> every feature silently NULL. Reject it.
    if window_secs <= 0:
        raise ValueError(
            f"time_window must be a positive whole multiple of aggregation_interval; got {window_secs}s."
        )
    if window_secs % interval_secs != 0:
        raise ValueError(
            f"time_window ({window_secs}s) must be a whole multiple of aggregation_interval "
            f"({interval_secs}s) for the tile model (the window is a count of tiles)."
        )


def _validate_window_rollup(aggregations: List[Aggregation], aggregation_interval) -> int:
    """Single-window precondition for the per-window rollup builders (offline floored, online now()):
    tile-supported aggs only, a single window, and window a whole multiple of the interval. Returns
    window_secs. Centralized so the online and offline rollups CANNOT validate differently.
    Online is per-window (the engine provisions one now()-anchored MV per distinct window), so each of
    those MVs is built from a single-window aggregation subset."""
    _assert_tile_supported(aggregations)
    _assert_distinct_output_names(aggregations)
    window_secs = _single_window_secs(aggregations)
    _assert_window_multiple_of_interval(window_secs, aggregation_interval)
    return window_secs


def _assert_distinct_output_names(aggregations: List[Aggregation]) -> None:
    # Each aggregation projects one rollup column aliased to its resolved_name. resolved_name returns an
    # explicit ``name`` verbatim (ignoring the window), so two aggregations sharing a name — or two
    # identical aggregations — collide on one alias, emitting ``... AS feat, ... AS feat`` (a duplicate
    # output column RisingWave rejects, or a silently-arbitrary pick for a by-name reader). The partial
    # columns are deduped (``_view_partials``); the OUTPUT columns must be guaranteed distinct too.
    names = [a.resolved_name(a.time_window) for a in aggregations]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"tile aggregations resolve to duplicate output column name(s) {dupes}; give each "
            f"aggregation a distinct name (resolved_name uses an explicit name verbatim, ignoring the "
            f"window, so same-name aggregations on different windows still collide)."
        )


def _validate_windows(aggregations: List[Aggregation], aggregation_interval) -> List[int]:
    """Multi-window precondition for the offline PIT builder: tile-supported aggs only, distinct output
    names, and EVERY aggregation's (non-null) window a whole multiple of the interval. Returns the
    DISTINCT window seconds ascending. The aggregations may carry different windows over the ONE shared
    tile set (tiles reused across time-windows), so unlike ``_validate_window_rollup`` this does
    NOT require a single window."""
    _assert_tile_supported(aggregations)
    _assert_distinct_output_names(aggregations)
    # group_aggregations_by_window owns the non-null-window check + the distinct-ascending window set
    # (one source of truth shared with the engine/apply provisioning path); here we only layer on the
    # multiple-of-interval precondition the rollup needs.
    windows = [secs for secs, _ in group_aggregations_by_window(aggregations)]
    for secs in windows:
        _assert_window_multiple_of_interval(secs, aggregation_interval)
    return windows


def group_aggregations_by_window(
    aggregations: List[Aggregation],
) -> List[Tuple[int, List[Aggregation]]]:
    """Group aggregations by their (distinct, non-null) window seconds, ascending. Each group becomes
    ONE online rollup MV (the engine provisions one now()-anchored MV per window — RisingWave can't put
    now() inside a CASE, so windows can't share an MV) AND one OnlineView serving shard (apply). The
    engine and apply MUST group identically so the per-window MV names match — hence this single shared
    helper. Pure (no interval): callers that need the multiple-of-interval precondition run
    ``_validate_windows`` first."""
    groups: dict = {}
    for a in aggregations:
        if a.time_window is None:
            raise ValueError(
                "tile rollup needs a non-null time_window on every aggregation; "
                f"got None on {a.function}({a.column})."
            )
        groups.setdefault(int(a.time_window.total_seconds()), []).append(a)
    return [(secs, groups[secs]) for secs in sorted(groups)]


def _tile_rollup_exprs(aggregations: List[Aggregation], prefix: str = "") -> str:
    """The per-aggregation recombine projection for the SINGLE-window rollup builders (the surrounding
    WHERE bounds the window, so no per-agg ``partial_filter``). Shared by online + floored rollups so
    they recombine per-tile partials IDENTICALLY (no-drift — one source of truth, via
    ``_tile_recombine``). ``prefix`` qualifies the partial columns for a joined relation."""
    return ", ".join(_tile_recombine(a, prefix=prefix) for a in aggregations)


def _tile_partials_projection(column_info: ColumnInfo, aggregations: List[Aggregation]) -> str:
    """The deduped window-independent partial columns as a ``{expr} AS {name}`` projection — the ONE
    source of the tile partial set, shared by the batch and streaming tile builders so they cannot drift.
    Includes the partial-name vs join-key clash guard (a bare ``{family}_{col}`` partial that equals an
    entity column would make the tiles MV have two identically-named columns, which RisingWave rejects)."""
    view_partials = _view_partials(aggregations)
    key_clash = sorted(set(column_info.join_keys_columns) & {name for (name, _) in view_partials})
    if key_clash:
        raise ValueError(
            f"entity/join-key column(s) {key_clash} collide with tile partial column name(s); rename "
            f"the entity column(s) (tile partials are named '<function>_<column>', e.g. sum_amount)."
        )
    return ", ".join(f"{expr} AS {name}" for (name, expr) in view_partials)


def build_batch_tile_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    relation: str,
    *,
    aggregation_interval,
) -> str:
    """Tile materialization for a BATCH feature view: one PARTIAL aggregate per (entity, tile),
    where a tile spans ``[tile_start, tile_start + aggregation_interval)`` and is stamped by
    ``tile_end`` (its event-time upper boundary). A plain batch ``GROUP BY date_trunc`` over a batch
    relation (e.g. a RisingWave Iceberg source) — NOT the streaming TUMBLE path. The additive
    partial is the aggregate itself, named by the final feature's ``resolved_name`` so tile ->
    rollup -> serve carry one column name. Rolled up at retrieval by ``build_tile_rollup_select``."""
    if not aggregations:
        raise ValueError("build_batch_tile_select requires at least one aggregation")
    _assert_tile_supported(aggregations)
    unit = _tile_unit(aggregation_interval)
    keys = ", ".join(column_info.join_keys_columns)
    bucket = f"date_trunc('{unit}', {column_info.timestamp_column})"
    partials = _tile_partials_projection(column_info, aggregations)
    return (
        f"SELECT {keys}, {bucket} + INTERVAL '1 {unit}' AS tile_end, {partials} "
        f"FROM {relation} GROUP BY {keys}, {bucket}"
    )


# Streaming tile intervals are restricted to HOUR/DAY: the streaming tiles MV buckets with RisingWave's
# epoch-anchored TUMBLE, but the offline PIT rollup floors with date_trunc — epoch-tumble lands on the
# SAME boundary as date_trunc only for hour/day (a 1-week TUMBLE anchors to epoch-Thursday, date_trunc
# 'week' to ISO-Monday, so weekly tiles would mis-grid online vs offline). Week/arbitrary intervals need
# an epoch-aligned offline floor first, which is not yet supported.
_STREAMING_TILE_INTERVAL_SECS = frozenset({3600, 86400})


def build_streaming_tile_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    relation: str,
    *,
    aggregation_interval,
) -> str:
    """Tile materialization for a STREAMING feature view — the streaming twin of ``build_batch_tile_select``.
    Emits the SAME per-(entity, tile_end) window-independent partials, but materialized by an EOWC TUMBLE at
    ``aggregation_interval`` over a watermarked append-only source (vs batch's ``date_trunc`` GROUP BY over an
    Iceberg relation). ``tile_end`` is the tumble ``window_end``; ``EMIT ON WINDOW CLOSE`` emits a tile ONCE,
    only when its interval closes past the watermark, so a late event is dropped once at the tile boundary
    (train/serve parity at the tile level: online and offline both read the same EOWC tiles). Everything downstream — the per-window now()-anchored
    rollup MVs and the offline PIT — reads the identical tile contract (``tile_end`` + bare partials),
    source-agnostic, so those builders need zero edits.

    The watermark + append-only precondition on ``relation`` is the PROVISIONING layer's responsibility (the
    engine asserts a ``watermark_delay_threshold`` before emitting EOWC), not this pure builder's."""
    if not aggregations:
        raise ValueError("build_streaming_tile_select requires at least one aggregation")
    _assert_tile_supported(aggregations)
    secs = int(aggregation_interval.total_seconds())
    if secs not in _STREAMING_TILE_INTERVAL_SECS:
        raise ValueError(
            f"streaming tile aggregation_interval must be 1 hour (3600s) or 1 day (86400s); got {secs}s. "
            f"The TUMBLE grid is epoch-anchored and must match the offline date_trunc floor — week/arbitrary "
            f"intervals need an epoch-aligned offline floor, which is not yet supported."
        )
    keys = ", ".join(column_info.join_keys_columns)
    ts = column_info.timestamp_column
    partials = _tile_partials_projection(column_info, aggregations)
    return (
        f"SELECT {keys}, window_end AS tile_end, {partials} "
        f"FROM tumble({relation}, {ts}, INTERVAL '{secs}' SECOND) "
        f"GROUP BY window_start, window_end, {keys} EMIT ON WINDOW CLOSE"
    )


def build_tile_rollup_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    tile_relation: str,
    *,
    aggregation_interval,
    as_of_sql: str,
) -> str:
    """Roll up tiles to the requested window, ANCHORED TO THE REQUEST/LABEL time (a request-anchored
    sliding window over a fixed tile set). Recombine each aggregation's per-tile
    partial with its rollup combiner (sum/min/max). The window is ``(end - time_window, end]`` where
    ``end = date_trunc(aggregation_interval, as_of)`` = the most-recent aggregation_interval boundary
    at or before the request/label time. ``as_of_sql`` is a SQL expression: a bind placeholder for
    online serving, or the entity-row timestamp column for offline PIT. ``tile_end`` carries the
    event-time PIT boundary, so there is no future leakage."""
    if not aggregations:
        raise ValueError("build_tile_rollup_select requires at least one aggregation")
    window_secs = _validate_window_rollup(aggregations, aggregation_interval)
    unit = _tile_unit(aggregation_interval)
    keys = ", ".join(column_info.join_keys_columns)
    rollups = _tile_rollup_exprs(aggregations)
    end = f"date_trunc('{unit}', {as_of_sql})"
    return (
        f"SELECT {keys}, {rollups} FROM {tile_relation} "
        f"WHERE tile_end > {end} - INTERVAL '{window_secs}' SECOND AND tile_end <= {end} "
        f"GROUP BY {keys}"
    )


def build_online_rollup_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    tile_relation: str,
    *,
    aggregation_interval,
) -> str:
    """Online rollup MV over the tiles: a CONTINUOUS RisingWave materialized view that maintains the
    request-anchored window rollup for ``as_of = now()`` (wall-clock), so the ONLINE READ stays an
    unchanged point-lookup (one row per entity = the current rollup).

    Uses a plain two-sided ``now()`` window ``tile_end > now() - W AND tile_end <= now()`` rather than
    ``build_tile_rollup_select``'s ``date_trunc(now())`` form. Validated end-to-end on RisingWave v3.0.0: a two-sided
    ``date_trunc(now())`` range in a CREATE MATERIALIZED VIEW is REJECTED ("Failed to run the query"),
    but the plain two-sided ``now()`` form is accepted AND maintained correctly as the wall-clock
    advances (tiles evicted past the lower bound and admitted as ``now()`` crosses the upper bound).
    The two forms are EQUIVALENT here because tiles live only at ``aggregation_interval`` boundaries and the
    window is a whole number of intervals (``_assert_window_multiple_of_interval``): no tile ever falls
    in the intra-interval gap between ``now()`` and ``floor(now(), interval)``. So online (now-anchored)
    == offline (floor-anchored) for the same as_of. ``max(tile_end) AS window_end`` is the PIT stamp the
    point-lookup orders by (one row per entity, so LIMIT 1 is that row)."""
    if not aggregations:
        raise ValueError("build_online_rollup_select requires at least one aggregation")
    window_secs = _validate_window_rollup(aggregations, aggregation_interval)
    keys = ", ".join(column_info.join_keys_columns)
    rollups = _tile_rollup_exprs(aggregations)
    return (
        f"SELECT {keys}, {rollups}, max(tile_end) AS window_end FROM {tile_relation} "
        f"WHERE tile_end > now() - INTERVAL '{window_secs}' SECOND AND tile_end <= now() "
        f"GROUP BY {keys}"
    )


def build_offline_tile_pit_query(
    entity_df_sql: str,
    entity_columns: List[str],
    label_ts_column: str,
    *,
    tiles_relation: str,
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    aggregation_interval,
    full_feature_names: bool = False,
    view_name: Optional[str] = None,
) -> str:
    """Offline point-in-time training rollup for a tile BatchFeatureView. For EACH entity row, rolls
    the tiles up in the request-anchored window ``(end - W, end]`` where ``end = floor(label_ts,
    aggregation_interval)`` — anchored to THAT ROW's label timestamp (NOT a global now()).

    ``aggregations`` is already the requested feature subset (the caller filters it to the features in
    feature_refs), so each rolled-up column maps to a requested feature. ``full_feature_names`` aliases
    every feature output as ``"{view_name}__{feature}"`` (the standard Feast offline contract); entity
    columns are never prefixed. Without it the bare per-window ``resolved_name`` is emitted.

    This CANNOT reuse Feast's standard latest-row PIT template: that picks the latest tile <= label,
    which anchors the window at the latest tile WITH DATA, not at floor(label) — they diverge when the
    most recent intervals have no events (validated end-to-end on RisingWave v3.0.0: 180/280/230 vs a wrong 180/280/280). So we
    range-JOIN the inlined entity rows to the tiles and GROUP BY the entity row. LEFT JOIN so a row
    with no tiles in range still appears (NULL feature).

    MULTI-WINDOW: the aggregations may carry DIFFERENT windows over the ONE shared tile set (tiles
    reused across time-windows). The join reads tiles up to the MAX window once; each aggregation
    recombines only the tiles inside ITS window via a per-agg ``CASE`` on ``tile_end`` — all windows in
    one query, one pass over the tiles.

    TTL note: the feature view's ``ttl`` is intentionally NOT applied as a second lower bound. For an
    aggregation feature view the ``time_window`` IS the lookback bound (windowed-aggregation semantics); a
    ttl shorter than the window would silently shrink the aggregation below what the user requested, and
    a longer ttl is a no-op. So the window is the single, authoritative bound."""
    if not aggregations:
        raise ValueError("build_offline_tile_pit_query requires at least one aggregation")
    max_window = max(_validate_windows(aggregations, aggregation_interval))
    unit = _tile_unit(aggregation_interval)
    keys = column_info.join_keys_columns
    e_cols = ", ".join(f'e."{c}"' for c in entity_columns)
    join_on = " AND ".join(f't."{k}" = e."{k}"' for k in keys)
    end = f'date_trunc(\'{unit}\', e."{label_ts_column}")'
    output_prefix = (view_name or "") if full_feature_names else ""
    rollups = ", ".join(
        _tile_recombine(
            a,
            prefix="t.",
            partial_filter=(
                f"t.tile_end > {end} - INTERVAL '{int(a.time_window.total_seconds())}' SECOND"
            ),
            output_prefix=output_prefix,
        )
        for a in aggregations
    )
    return (
        f"SELECT {e_cols}, {rollups} FROM ({entity_df_sql}) e "
        f"LEFT JOIN {tiles_relation} t ON {join_on} "
        f"AND t.tile_end > {end} - INTERVAL '{max_window}' SECOND AND t.tile_end <= {end} "
        f"GROUP BY {e_cols}"
    )


def _quote(identifier: str) -> str:
    return f'"{identifier}"'


class _RWNode(DAGNode):
    """Base for RisingWave nodes that carry a SQL relation forward.

    The relation is stored under ``DAGValue.data`` as the SQL string; the column
    list is carried in ``metadata["columns"]`` so downstream nodes can reference
    columns without re-parsing the SQL (mirrors Flink's metadata-carried columns).
    """

    def __init__(self, name, view, column_info: ColumnInfo, inputs=None):
        super().__init__(name, inputs=inputs)
        self.view = view
        self.column_info = column_info

    def _input_value(self, context: ExecutionContext) -> DAGValue:
        value = self.get_single_input_value(context)
        value.assert_format(DAGFormat.RISINGWAVE)
        return value

    def _input_relation(self, context: ExecutionContext) -> str:
        return self._input_value(context).data

    @staticmethod
    def _columns(value: DAGValue) -> List[str]:
        columns = value.metadata.get("columns") if value.metadata else None
        if columns:
            return list(columns)
        raise ValueError(
            "Could not infer columns for RisingWave DAG value from metadata."
        )

    def _value(
        self, relation: str, columns: List[str], *, metadata: Optional[dict] = None
    ) -> DAGValue:
        return DAGValue(
            data=relation,
            format=DAGFormat.RISINGWAVE,
            metadata={**(metadata or {}), "columns": list(columns)},
        )


def _agg_output_columns(column_info: ColumnInfo, aggregations: List[Aggregation]) -> List[str]:
    columns = list(column_info.join_keys_columns)
    columns.extend(a.resolved_name(a.time_window) for a in aggregations)
    if aggregations and aggregations[0].time_window is not None:
        columns.extend(["window_start", "window_end"])
    return columns


class RWSourceNode(_RWNode):
    """Source read.

    For retrieval the relation is the RisingWave object holding governed history (the
    Iceberg-history read-back source, or the online MV); for a materialize backfill it
    is the same provisioned relation, narrowed to the bounded window by the filter
    node. Either way the engine provisions ``{project}_{view}_src`` in
    ``engine.update()``; we reference it and carry the source schema columns.
    """

    def execute(self, context: ExecutionContext) -> DAGValue:
        relation = _quote(source_name(context.project, self.view.name))
        columns = list(self.column_info.join_keys_columns)
        ts = self.column_info.timestamp_column
        if ts and ts not in columns:
            columns.append(ts)
        for feature_col in self.column_info.feature_cols:
            if feature_col not in columns:
                columns.append(feature_col)
        return self._value(
            relation,
            columns,
            metadata={
                "source": "feature_view_source",
                "timestamp_field": ts,
            },
        )


class RWJoinNode(_RWNode):
    """Entity-spine LEFT JOIN for historical retrieval.

    Mirrors ``FlinkJoinNode``: the entity_df (a pandas DataFrame OR a SQL string) is
    the LEFT side; feature rows are joined on the view join keys, and the entity
    event-timestamp column is aliased to ``ENTITY_TS_ALIAS`` so the downstream filter
    node can apply the point-in-time cut. ``ENTITY_ROW_ID`` is attached so dedup picks
    exactly one feature row per entity row.
    """

    def execute(self, context: ExecutionContext) -> DAGValue:
        feature_value = self._input_value(context)
        feature_relation = feature_value.data
        feature_columns = self._columns(feature_value)
        join_keys = self.column_info.join_keys_columns

        if context.entity_df is None:
            raise RuntimeError(
                f"RWJoinNode '{self.name}' requires an entity_df on the execution "
                "context for historical retrieval."
            )

        entity_relation, entity_columns, entity_ts_col = self._entity_relation(
            context.entity_df, join_keys
        )

        # Feature columns that are not join keys / entity columns flow through.
        feature_only = [
            col
            for col in feature_columns
            if col not in join_keys and col not in entity_columns
        ]
        on_clause = " AND ".join(
            f"e.{_quote(key)} = f.{_quote(key)}" for key in join_keys
        )
        select_entity = ", ".join(f"e.{_quote(col)}" for col in entity_columns)
        select_features = ", ".join(f"f.{_quote(col)}" for col in feature_only)
        projection = ", ".join(p for p in (select_entity, select_features) if p)
        select = (
            f"SELECT {projection} FROM ({entity_relation}) AS e "
            f"LEFT JOIN ({feature_relation}) AS f ON {on_clause}"
        )
        output_columns = entity_columns + feature_only
        # Carry the feature side's effective event-timestamp column forward (window_end
        # when the feature relation was already aggregated upstream of this join, on the
        # PIT-aggregated retrieval path). The downstream filter/dedup nodes read it to
        # apply the as-of cut on window_end rather than the (now-absent) raw ts.
        feature_event_ts = (feature_value.metadata or {}).get("event_timestamp_column")
        return self._value(
            f"({select})",
            output_columns,
            metadata={
                "joined_on": join_keys,
                "join_type": "left",
                "entity_timestamp_column": entity_ts_col,
                "event_timestamp_column": feature_event_ts,
            },
        )

    def _entity_relation(self, entity_df, join_keys: List[str]):
        """Build the entity-spine relation + its columns + its timestamp column.

        Handles ``entity_df`` as a pandas DataFrame OR a SQL string, using the shared
        ``find_entity_timestamp_column`` / ``infer_entity_timestamp_column`` helpers
        so the event-timestamp column resolves to ``ENTITY_TS_ALIAS``.
        """
        if isinstance(entity_df, pd.DataFrame):
            entity_columns = list(entity_df.columns)
            entity_schema = dict(zip(entity_df.columns, entity_df.dtypes))
            entity_ts_col = infer_entity_timestamp_column(entity_schema)
            # NOT YET IMPLEMENTED: a pandas entity_df must be staged into RisingWave (e.g. a
            # temporary table / VALUES list) before it can be joined over pgwire. We
            # reference a conventional staging relation here; the upload itself is not yet
            # implemented (mirrors Flink's pandas_to_flink_table staging).
            relation = f"SELECT * FROM {_quote(ENTITY_ROW_ID + '_spine')}"
            select_exprs = [
                f"{_quote(col)} AS {_quote(ENTITY_TS_ALIAS)}"
                if col == entity_ts_col
                else _quote(col)
                for col in entity_columns
            ]
            select_exprs.append(
                f"ROW_NUMBER() OVER () - 1 AS {_quote(ENTITY_ROW_ID)}"
            )
            relation = f"SELECT {', '.join(select_exprs)} FROM ({relation}) AS spine"
            output_columns = [
                ENTITY_TS_ALIAS if col == entity_ts_col else col
                for col in entity_columns
            ]
            output_columns.append(ENTITY_ROW_ID)
            return relation, output_columns, entity_ts_col

        if isinstance(entity_df, str):
            entity_sql = entity_df.strip()
            if not entity_sql:
                raise ValueError("SQL entity_df for RisingWave must be non-empty.")
            # Wrap the user SQL in a subquery; we cannot statically know its columns,
            # so SELECT *, find the timestamp column by name and re-alias it.
            entity_ts_col = find_entity_timestamp_column(
                [ENTITY_TS_ALIAS]
            ) or ENTITY_TS_ALIAS
            # The user SQL must already expose an ``event_timestamp`` (or the alias);
            # the filter node references ENTITY_TS_ALIAS, so re-alias to it.
            relation = (
                f"SELECT spine.*, spine.{_quote('event_timestamp')} AS "
                f"{_quote(ENTITY_TS_ALIAS)}, ROW_NUMBER() OVER () - 1 AS "
                f"{_quote(ENTITY_ROW_ID)} FROM ({entity_sql}) AS spine"
            )
            # Columns of a SQL spine are not statically known; carry only what the
            # filter/dedup nodes reference. Feature columns are added by the join.
            output_columns = [ENTITY_TS_ALIAS, ENTITY_ROW_ID] + join_keys
            return relation, output_columns, ENTITY_TS_ALIAS

        raise TypeError(
            "RisingWave entity_df must be a pandas DataFrame, a SQL string, or None."
        )


class RWFilterNode(_RWNode):
    """Filter: the point-in-time cut + the TTL lower bound + an optional view filter.

    Mirrors ``FlinkFilterNode``: when an entity_df is present (its ``ENTITY_TS_ALIAS``
    is in scope), apply the inclusive ``timestamp_column <= ENTITY_TS_ALIAS`` PIT cut
    (postgres.py:962) plus, if a ttl is set, ``timestamp_column >= ENTITY_TS_ALIAS -
    INTERVAL ttl``. Without an entity_df (the backfill path), apply the TTL window
    relative to ``now()``. An optional ``view.filter`` expression is ANDed on.

    The PIT/TTL cut is applied on the *effective* event-timestamp column: ``window_end``
    when the input relation was already windowed-aggregated (carried via metadata by
    ``RWAggregationNode``), otherwise the raw event ts. Cutting an aggregated relation
    on its raw ts would be impossible (the column is gone) and cutting the raw stream on
    ``ts <= ENTITY_TS_ALIAS`` *before* a window closes would admit partial, future-dated
    windows — so on the aggregated-PIT path the builder runs this node AFTER aggregation
    with ``include_pit_cut=True`` (cut on window_end), and a separate pre-aggregation
    node with ``include_pit_cut=False`` for the raw-row ``view.filter`` only.
    """

    def __init__(
        self,
        name,
        view,
        column_info,
        *,
        filter_expr=None,
        ttl=None,
        inputs=None,
        include_pit_cut: bool = True,
    ):
        super().__init__(name, view, column_info, inputs=inputs)
        self.filter_expr = filter_expr
        self.ttl = ttl
        # When False, the PIT (``ts <= ENTITY_TS_ALIAS``) + TTL predicates are skipped
        # entirely. Used for the pre-aggregation filter on the aggregated-PIT path,
        # where the as-of cut must instead be applied on window_end AFTER aggregation
        # (a raw-ts cut there would leak partial/future-dated windows).
        self.include_pit_cut = include_pit_cut

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        relation = input_value.data
        columns = self._columns(input_value)
        # On an already-aggregated relation the raw event-timestamp column is gone and
        # window_end is the effective event timestamp (carried by RWAggregationNode). The
        # PIT cut MUST be applied on window_end, not the raw ts: cutting the raw ts before
        # the window closes leaks partial/future-dated windows. Fall back to the raw ts
        # column for the non-aggregated path.
        metadata = input_value.metadata or {}
        ts = metadata.get("event_timestamp_column") or self.column_info.timestamp_column
        conditions: List[str] = []

        if self.include_pit_cut and ts and ENTITY_TS_ALIAS in columns and ts in columns:
            # Inclusive <= per Feast's offline PIT join (postgres.py:962); a window
            # row stamped by window_end is only admitted once it has closed.
            conditions.append(f"{_quote(ts)} <= {_quote(ENTITY_TS_ALIAS)}")
            if self.ttl:
                secs = int(self.ttl.total_seconds())
                conditions.append(
                    f"{_quote(ts)} >= {_quote(ENTITY_TS_ALIAS)} - "
                    f"INTERVAL '{secs}' SECOND"
                )
        elif self.include_pit_cut and self.ttl and ts and ts in columns:
            secs = int(self.ttl.total_seconds())
            conditions.append(f"{_quote(ts)} >= now() - INTERVAL '{secs}' SECOND")

        if self.filter_expr:
            conditions.append(f"({self.filter_expr})")

        if not conditions:
            return input_value

        where = " AND ".join(conditions)
        # TUMBLE/HOP's 1st arg cannot be a sub-SELECT but CAN be a CTE/view
        # (window_table_function.rs:66), so we wrap as a subquery relation.
        select = f"SELECT * FROM ({relation}) AS _f WHERE {where}"
        return self._value(
            f"({select})",
            columns,
            # Preserve event_timestamp_column so a downstream dedup on the aggregated
            # path still knows to order by window_end (it is unchanged by the filter).
            metadata={**metadata, "filter_applied": True},
        )


class RWAggregationNode(_RWNode):
    """The value-add node: honors ``time_window`` (TUMBLE/HOP), which the Spark/Flink/
    Ray engines reject (aggregation/__init__.py:132-134). Also supports non-windowed
    GROUP BY (``time_window is None``)."""

    def __init__(
        self, name, view, column_info, *, source_is_retractable, emit_on_close, inputs=None
    ):
        super().__init__(name, view, column_info, inputs=inputs)
        self.source_is_retractable = source_is_retractable
        self.emit_on_close = emit_on_close

    def execute(self, context: ExecutionContext) -> DAGValue:
        aggregations = list(self.view.aggregations)
        input_value = self._input_value(context)
        select = build_windowed_agg_select(
            self.column_info,
            aggregations,
            input_value.data,
            source_is_retractable=self.source_is_retractable,
            emit_on_close=self.emit_on_close,
        )
        columns = _agg_output_columns(self.column_info, aggregations)
        # After windowed aggregation the raw event-timestamp column is gone; window_end
        # is the row's effective event timestamp (a window [t, t+w) is only knowable at
        # t+w). Carry it forward so the downstream PIT filter cuts on window_end — never
        # on the raw ts (which would admit a still-open/partial window with a
        # future-dated stamp). Non-windowed GROUP BY has no event timestamp.
        windowed = bool(aggregations) and aggregations[0].time_window is not None
        event_ts_col = "window_end" if windowed else None
        # Preserve the entity-spine columns (ENTITY_TS_ALIAS / ENTITY_ROW_ID) when the
        # join ran upstream of aggregation; they are NOT in the GROUP BY, so they only
        # exist if the builder placed the join AFTER aggregation (the PIT path).
        return self._value(
            f"({select})",
            columns,
            metadata={
                **(input_value.metadata or {}),
                "aggregated": True,
                "select": select,
                "event_timestamp_column": event_ts_col,
            },
        )


class RWDedupNode(_RWNode):
    """Latest-row-per-key for the only_latest / historical path. Mirrors
    ``FlinkDedupNode`` and the Postgres offline PIT dedup (postgres.py:1002-1012):
    ``ROW_NUMBER() OVER (PARTITION BY <keys> ORDER BY ts DESC[, created_ts DESC]) =
    1``. Partitions by ``ENTITY_ROW_ID`` when present (one feature row per entity
    spine row), else by the join keys.

    Orders by the effective event-timestamp column: ``window_end`` on the aggregated
    path (carried via metadata by ``RWAggregationNode`` — so a HOP/TUMBLE view collapses
    its many closed windows per label row to the single as-of-latest one), or the raw
    event ts (+created ts) otherwise. If neither is present it RAISES rather than
    silently picking an arbitrary row, which would break the latest-per-entity
    guarantee."""

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        relation = input_value.data
        columns = self._columns(input_value)

        dedup_keys = (
            [ENTITY_ROW_ID]
            if ENTITY_ROW_ID in columns
            else list(self.column_info.join_keys_columns)
        )
        dedup_keys = [key for key in dedup_keys if key in columns]
        if not dedup_keys:
            return input_value

        # On an aggregated relation the raw ts column has been dropped and window_end is
        # the effective event timestamp (carried by RWAggregationNode); order by it so
        # "latest per entity" means the latest CLOSED window. Fall back to the raw ts
        # (+created_ts) for the non-aggregated path.
        metadata = input_value.metadata or {}
        event_ts_col = metadata.get("event_timestamp_column")
        if event_ts_col:
            order_cols = [event_ts_col] if event_ts_col in columns else []
        else:
            ts = self.column_info.timestamp_column
            created_ts = self.column_info.created_timestamp_column
            order_cols = [c for c in (ts, created_ts) if c and c in columns]
        if not order_cols:
            # No time-ordering column means "latest per entity" cannot be defined;
            # an arbitrary pick would silently return a non-as-of row. Refuse instead
            # of degrading to dedup_keys[0] ASC.
            raise ValueError(
                f"[Dedup: {self.name}] Cannot select the latest row per "
                f"{dedup_keys}: no event-timestamp column is present in {sorted(columns)}. "
                "An ordered timestamp (raw event ts, or window_end for an aggregated "
                "view) is required to define the point-in-time-latest row."
            )
        order_exprs = [f"{_quote(c)} DESC" for c in order_cols]

        partition = ", ".join(_quote(k) for k in dedup_keys)
        order_by = ", ".join(order_exprs)
        projection = ", ".join(_quote(c) for c in columns)
        select = (
            f"SELECT {projection} FROM (SELECT *, ROW_NUMBER() OVER ("
            f"PARTITION BY {partition} ORDER BY {order_by}) AS "
            f"{_quote(DEDUP_ROW_NUMBER)} FROM ({relation}) AS _d) AS _r "
            f"WHERE {_quote(DEDUP_ROW_NUMBER)} = 1"
        )
        return self._value(
            f"({select})",
            columns,
            metadata={**(input_value.metadata or {}), "deduped": True},
        )


class RWTransformNode(_RWNode):
    """RisingWave SQL/UDF transformation.

    A transformation is only honored when it is expressible as a RisingWave SQL
    string (``view.feature_transformation`` exposing a ``.sql`` / ``.expr``). Anything
    that requires running arbitrary python out-of-engine raises NotImplementedError —
    honest, mirroring Flink's JSON-validation NotImplementedError (flink/nodes.py:723).
    """

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        relation = input_value.data
        columns = self._columns(input_value)

        transform = getattr(self.view, "feature_transformation", None)
        sql_expr = getattr(transform, "sql", None) or getattr(transform, "expr", None)
        if not sql_expr or not isinstance(sql_expr, str):
            raise NotImplementedError(
                "RisingWaveComputeEngine can only push down transformations expressed "
                "as a RisingWave SQL string (feature_transformation.sql). Arbitrary "
                "python/pandas/UDF transforms are not expressible in-engine "
                "(not supported). Pre-transform upstream, or use a SQL transformation."
            )
        # The SQL transform replaces the projection over the input relation; the
        # output column set is the view's declared features (unvalidated: we trust the
        # transform SQL to emit exactly these names).
        select = f"SELECT {sql_expr} FROM ({relation}) AS _t"
        output_columns = (
            [f.name for f in self.view.features]
            if getattr(self.view, "features", None)
            else columns
        )
        return self._value(
            f"({select})", output_columns, metadata={"transformed": True}
        )


class RWValidationNode(_RWNode):
    """Column-presence validation. Mirrors ``FlinkValidationNode``: assert every
    expected feature column is present in the relation's carried column list; raise
    with the actual columns if any are missing. (Type/JSON validation would require
    pulling data out of RisingWave and is not yet supported.)"""

    def __init__(self, name, view, column_info, *, expected_columns=None, inputs=None):
        super().__init__(name, view, column_info, inputs=inputs)
        self.expected_columns = expected_columns or []

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        columns = self._columns(input_value)
        missing = set(self.expected_columns) - set(columns)
        if missing:
            raise ValueError(
                f"[Validation: {self.name}] Missing expected columns: "
                f"{sorted(missing)}. Actual columns: {sorted(columns)}"
            )
        return self._value(
            input_value.data,
            columns,
            metadata={**(input_value.metadata or {}), "validated": True},
        )


class RWOutputNode(_RWNode):
    """Terminal node.

    For online + offline serving of a StreamFeatureView, the MV + Iceberg sink
    provisioned in ``engine.update()`` already own the per-row store writes (the live
    MV keeps online fresh; the sink streams the offline history). So the only per-task
    work here is the BOUNDED Iceberg backfill INSERT, and only for a materialize task:
    ``write_output = isinstance(task, MaterializationTask)`` (mirrors Flink, which
    gates the write on the task type).

    - retrieval terminal: carry the final SELECT as ``sql`` so the plan/job can run it
      over pgwire (``RisingWaveDAGRetrievalJob``).
    - materialize terminal: emit the ``INSERT ... SELECT`` into the offline staging
      table, preserving window_end-as-event_timestamp and the append-only/composite-PK
      invariants owned by ``_iceberg_sink_ddl``.
    """

    def __init__(self, name, view, column_info, *, write_output: bool, inputs=None):
        super().__init__(name, view, column_info, inputs=inputs)
        self.write_output = write_output

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        relation = input_value.data
        columns = self._columns(input_value)

        # The relation is either a bare object name ("foo") or a parenthesized
        # subquery "(SELECT ...)"; normalize to a runnable SELECT.
        if relation.startswith("(") and relation.endswith(")"):
            select_sql = relation[1:-1]
        else:
            select_sql = f"SELECT * FROM {relation}"

        sql: Optional[str] = select_sql
        if self.write_output and getattr(self.view, "offline", False):
            # window_end is already the event timestamp emitted by the aggregation
            # node; the staging table mirrors the live sink's projection so backfill
            # rows are byte-compatible with streamed rows.
            # UNVERIFIED end-to-end: the bounded backfill INSERT and
            # its late-data parity with the live stream are not yet proven in-repo.
            # Preferred long-term: read the live sink's Iceberg history so backfill ==
            # what was served. The bounded [start, end) predicate is applied
            # by the upstream filter node before this INSERT.
            staging = _quote(offline_staging_name(context.project, self.view.name))
            sql = f"INSERT INTO {staging} {select_sql}"

        return self._value(
            relation, columns, metadata={**(input_value.metadata or {}), "sql": sql}
        )
