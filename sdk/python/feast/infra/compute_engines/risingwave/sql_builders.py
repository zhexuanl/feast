"""The RisingWave engine's SQL builders — the pure functions that compose a SELECT/relation
string for every materialization + retrieval shape, and the validation helpers that hold this
engine's correctness invariants.

Each builder returns a RisingWave SQL relation string (a CTE/subquery); none opens a database
connection. The DAG nodes (``nodes``) compose these into a single query the engine runs at the
edges. This module sits atop the partial-aggregate algebra (``tiling``) and the view-tag carriers
(``aggregation_carriers``), reading the per-tile partials + recombine expressions from there and
the per-aggregation parameters/offsets/lifetimes/series from the carriers.

Every SQL fragment traces to a RisingWave end-to-end example validated against a live instance.
Anything not yet validated end-to-end is marked ``UNVERIFIED`` and listed under the unvalidated
surfaces in ``README.md``.
"""

from typing import Dict, List, Optional, Sequence

from feast.aggregation import Aggregation
from feast.infra.compute_engines.dag.context import ColumnInfo
from feast.infra.compute_engines.risingwave.aggregation_carriers import (
    _agg_offset_secs,
    group_aggregations_by_window,
    is_lifetime_agg,
    is_series_agg,
)
from feast.infra.compute_engines.risingwave.names import SERIES_SNAPSHOT_ENDS_COL
from feast.infra.compute_engines.risingwave.tiling import (
    _SEQUENCE_ORDER,
    _assert_tile_supported,
    _cumulative_partials,
    _cumulative_recombine_expr,
    _series_recombine,
    _tile_recombine,
    _tile_value_expr,
    _view_partials,
    is_invertible_agg,
    snapshot_series_aggs,
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
#   - approx_percentile: parameterized by the quantile, emitted as
#     approx_percentile(<p>) WITHIN GROUP (ORDER BY <col>). The quantile has no home on
#     feast.Aggregation (which carries no parameter field), so it rides out-of-band in the
#     ``agg_params`` map (keyed by the aggregation's resolved_name) the builders take. A monoid
#     with no inverse, so plain (windowed/EOWC) only — never tile-decomposed (no additive merge).
#   - first(n) / last(n) / first_distinct(n) / last_distinct(n) (sequence features): Array-valued —
#     the n earliest/most-recent values per key+window, ordered. Emitted as
#     (array_agg(<col> ORDER BY <ts> ASC|DESC))[1:n], wrapped in array_distinct for the _distinct
#     variants (RisingWave rejects array_agg(DISTINCT ... ORDER BY <other column>)). n rides the
#     same ``agg_params`` carrier. Monoids (no inverse), plain (windowed/EOWC) only for now — the
#     bounded tile recombine (top-n per tile, re-merged at rollup) is a later delivery.
# Deliberately EXCLUDED — rejected at apply with a reason, not silently:
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
        "approx_percentile",
        "first",
        "last",
        "first_distinct",
        "last_distinct",
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
MONOID_FUNCTIONS = frozenset(
    {
        "min",
        "max",
        "count_distinct",
        "approx_count_distinct",
        "approx_percentile",
        "first",
        "last",
        "first_distinct",
        "last_distinct",
    }
)

# Feast Aggregation.function -> RisingWave SQL function (only names that differ).
_SQL_FUNCTION = {"mean": "avg"}

DEDUP_ROW_NUMBER = "_feast_row_number"


def _fmt_param(value: float) -> str:
    # Render a numeric aggregate parameter for SQL: an integer-valued float as an int literal
    # (5.0 -> "5"), otherwise its plain decimal form (0.95 -> "0.95"). Keeps the emitted SQL
    # free of "5.0" / scientific notation.
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _agg_expr(
    agg: Aggregation,
    agg_params: Optional[Dict[str, List[float]]] = None,
    ts_col: Optional[str] = None,
) -> str:
    # Output column == resolved_name(time_window) so the online MV and the offline
    # sink emit byte-identical column names — no online/offline column-name skew
    # (aggregation/__init__.py:106-118).
    out = agg.resolved_name(agg.time_window)
    if agg.function == "count_distinct":
        return f"count(distinct {agg.column}) AS {out}"
    if agg.function in _SEQUENCE_ORDER:
        # Array-valued: the n earliest/most-recent values, ordered by event time. n rides in
        # agg_params keyed by resolved_name. CAVEAT: array_agg materializes the WHOLE window's
        # values per key before the [1:n] slice, so MV state grows with the window's event count
        # (unbounded for high-cardinality windows) — the bounded tile recombine is a later delivery.
        params = (agg_params or {}).get(out)
        if not params:
            raise ValueError(
                f"{agg.function} on {agg.column!r} needs an n parameter (the number of values to "
                f"keep); none was provided. n rides in agg_params keyed by the aggregation's "
                f"resolved name ({out!r})."
            )
        n = int(params[0])
        ordered = (
            f"array_agg({agg.column} ORDER BY {ts_col} {_SEQUENCE_ORDER[agg.function]})"
        )
        if agg.function.endswith("_distinct"):
            ordered = f"array_distinct({ordered})"
        return f"({ordered})[1:{n}] AS {out}"
    if agg.function == "approx_percentile":
        # Parameterized by the quantile (params[0]) and an optional precision (params[1]); neither
        # has a field on feast.Aggregation, so they ride in agg_params keyed by this aggregation's
        # resolved_name. RisingWave's approx_percentile takes the quantile plus an optional
        # relative-error bound, so a precision maps to relative_error = 1 / precision (higher
        # precision => tighter error; precision 100 => 0.01, RisingWave's own default).
        params = (agg_params or {}).get(out)
        if not params:
            raise ValueError(
                f"approx_percentile on {agg.column!r} needs a quantile parameter (a number in "
                f"(0, 1)); none was provided. The quantile rides in agg_params keyed by the "
                f"aggregation's resolved name ({out!r})."
            )
        args = _fmt_param(params[0])
        if len(params) > 1 and params[1]:
            args += f", {_fmt_param(1.0 / params[1])}"
        return (
            f"approx_percentile({args}) WITHIN GROUP (ORDER BY {agg.column}) AS {out}"
        )
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
    agg_params: Optional[Dict[str, List[float]]] = None,
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
            f"supports {sorted(SUPPORTED_AGG_FUNCTIONS)}. aggregation_secondary_key "
            f"(per-key Array/Map output) is not yet supported."
        )

    # Two aggregations that resolve to the same output column would emit `... AS feat, ... AS feat`
    # (a duplicate column RisingWave rejects) and, for a parameterized agg, collide on the
    # resolved_name-keyed param carrier. The tile builders already guard this; the plain/EOWC path
    # needs it too — it is the only path the parameterized/monoid aggregates run on.
    _assert_distinct_output_names(aggregations)

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
    exprs = ", ".join(
        _agg_expr(a, agg_params, column_info.timestamp_column) for a in aggregations
    )
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


def _assert_offset_multiple_of_interval(offset_secs: int, aggregation_interval) -> None:
    # |offset| shifts the window by a COUNT of tiles, so it must be a whole number of
    # aggregation_intervals — the same invariant (for the same reason) the window obeys: only when
    # |offset| is a multiple of the interval does the online now()-anchored upper bound (now() - |offset|)
    # select the SAME tiles as the offline floor-anchored bound (end - |offset|), so online == offline.
    # A non-multiple offset would silently diverge online from training, so guard it at the builder too
    # (not only at the authoring factory), where every writer of the offset carrier is forced through.
    interval_secs = int(aggregation_interval.total_seconds())
    if abs(int(offset_secs)) % interval_secs != 0:
        raise ValueError(
            f"offset ({int(offset_secs)}s) must be a whole multiple of aggregation_interval "
            f"({interval_secs}s) for the tile model (the offset shifts the window by a count of tiles)."
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
            f"aggregations resolve to duplicate output column name(s) {dupes}; give each "
            f"aggregation a distinct name (resolved_name uses an explicit name verbatim, ignoring the "
            f"window, so same-name aggregations on different windows still collide)."
        )


def _validate_windows(
    aggregations: List[Aggregation],
    aggregation_interval,
    lifetimes: Optional[Dict[str, Optional[int]]] = None,
    series: Optional[Dict[str, Sequence[int]]] = None,
) -> List[int]:
    """Multi-window precondition for the offline PIT builder: tile-supported aggs only, distinct output
    names, and EVERY WINDOWED aggregation's window a positive whole multiple of the interval. Returns the
    DISTINCT window seconds ascending. The aggregations may carry different windows over the ONE shared
    tile set (tiles reused across time-windows), so unlike ``_validate_window_rollup`` this does NOT
    require a single window.

    A LIFETIME aggregation (per the lifetime carrier) carries no finite window — its lower bound is
    dropped at the rollup, not a count of tiles — so it is excluded from the window grouping/check. A
    null/zero window NOT in the carrier is still rejected (``group_aggregations_by_window`` raises on a
    null window, ``_assert_window_multiple_of_interval`` on a zero one): that is the non-servable plain
    GROUP BY, distinguished from a lifetime aggregation only by the carrier — never by the value alone."""
    _assert_tile_supported(aggregations)
    _assert_distinct_output_names(aggregations)
    # A lifetime aggregation (no finite window) and a window-series (a fan of windows assembled
    # positionally, its geometry on the series carrier) both lower to a null window, so both are excluded
    # from the single-window grouping/check; a series instead has its step validated below.
    windowed = [
        a
        for a in aggregations
        if not is_lifetime_agg(a, lifetimes) and not is_series_agg(a, series)
    ]
    windows = [secs for secs, _ in group_aggregations_by_window(windowed)] if windowed else []
    for secs in windows:
        _assert_window_multiple_of_interval(secs, aggregation_interval)
    # A window-series element i is the recombine over the tiles in (end - W - i*step, end - i*step] — a
    # single-scan CASE over the shared tile set, NOT a one-tile placement. So the only alignment the
    # recombine needs is that the step AND the window are each a whole number of tiles (multiples of the
    # aggregation_interval), so every window edge lands on a tile boundary. A coarser step (step =
    # k*interval) and overlapping windows (window > step) are both fine: each element re-selects its own
    # tile set, and the recombine over multiple tiles is exact for every supported function (overlap means
    # adjacent windows intentionally share tiles). length must be positive (an empty ARRAY[] is rejected
    # by RisingWave as an untyped array literal).
    for a in aggregations:
        if is_series_agg(a, series):
            w, s, length = series[a.resolved_name(a.time_window)]
            _assert_window_multiple_of_interval(int(s), aggregation_interval)
            _assert_window_multiple_of_interval(int(w), aggregation_interval)
            if int(length) < 1:
                raise ValueError(
                    f"window-series length must be a positive number of windows; got {length}."
                )
    return windows


def _tile_rollup_exprs(
    aggregations: List[Aggregation],
    prefix: str = "",
    agg_params: Optional[Dict[str, List[float]]] = None,
) -> str:
    """The per-aggregation recombine projection for the SINGLE-window rollup builders (the surrounding
    WHERE bounds the window, so no per-agg ``partial_filter``). Shared by online + floored rollups so
    they recombine per-tile partials IDENTICALLY (no-drift — one source of truth, via
    ``_tile_recombine``). ``prefix`` qualifies the partial columns for a joined relation."""
    return ", ".join(
        _tile_recombine(a, prefix=prefix, agg_params=agg_params) for a in aggregations
    )


def _assert_secondary_key_distinct(
    secondary_key: Optional[str], column_info: ColumnInfo, aggregations: List[Aggregation]
) -> None:
    # The secondary key is a SEPARATE GROUP BY dimension, so it must not coincide with a join key, the
    # timestamp, or an aggregation output: a collision would list the column twice in the tile GROUP BY
    # or shadow an output column. Fail fast with a clear message rather than an opaque RisingWave error.
    if not secondary_key:
        return
    clashes = set(column_info.join_keys_columns) | {column_info.timestamp_column} | {
        a.resolved_name(a.time_window) for a in aggregations
    }
    if secondary_key in clashes:
        raise ValueError(
            f"aggregation_secondary_key '{secondary_key}' must be a distinct raw column — it cannot be a "
            f"join key, the timestamp, or an aggregation output column."
        )


def _secondary_key_map_projection(
    aggregations: List[Aggregation], secondary_key: str, output_names: List[str]
) -> str:
    """The OUTER projection that collapses the secondary-key dimension into a per-aggregation Map: for
    each aggregation, ``jsonb_object_agg(secondary_key, <its scalar>)`` keyed by the secondary key,
    aliased back to the aggregation's output name. RisingWave maintains ``jsonb_object_agg`` incrementally
    in an MV and psycopg/pgx decode the jsonb column to a dict; ``map_agg`` is rejected, so jsonb is the
    carrier. NULL secondary-key rows are filtered (a NULL breakdown bucket is not meaningful, and
    jsonb_object_agg rejects a NULL key). An entity with no in-window tiles, or only NULL keys, would
    otherwise yield an EMPTY map ``{}`` offline (the LEFT-JOIN miss / filtered rows) while the online MV
    simply has NO row for it — a train/serve skew — so ``NULLIF(..., '{}')`` maps the empty breakdown to
    NULL on BOTH sides, matching the online absent-entity and the scalar combiners' empty -> NULL
    convention (cf. count_distinct ``NULLIF(cardinality(...), 0)``)."""
    return ", ".join(
        "NULLIF("
        f'jsonb_object_agg("{secondary_key}", {out}) '
        f'FILTER (WHERE "{secondary_key}" IS NOT NULL), \'{{}}\'::jsonb) AS {out}'
        for out in output_names
    )


def _tile_partials_projection(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    agg_params: Optional[Dict[str, List[float]]] = None,
) -> str:
    """The deduped window-independent partial columns as a ``{expr} AS {name}`` projection — the ONE
    source of the tile partial set, shared by the batch and streaming tile builders so they cannot drift.
    Includes the partial-name vs join-key clash guard (a bare ``{family}_{col}`` partial that equals an
    entity column would make the tiles MV have two identically-named columns, which RisingWave rejects)."""
    view_partials = _view_partials(
        aggregations, column_info.timestamp_column, agg_params
    )
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
    agg_params: Optional[Dict[str, List[float]]] = None,
    secondary_key: Optional[str] = None,
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
    _assert_secondary_key_distinct(secondary_key, column_info, aggregations)
    unit = _tile_unit(aggregation_interval)
    keys = ", ".join(column_info.join_keys_columns)
    bucket = f"date_trunc('{unit}', {column_info.timestamp_column})"
    partials = _tile_partials_projection(column_info, aggregations, agg_params)
    # A secondary key adds a second GROUP BY dimension to the tiles: tiles become per-(entity,
    # secondary_key, tile_end), collapsed into a per-key Map at rollup.
    sk_sel = f'"{secondary_key}", ' if secondary_key else ""
    sk_grp = f', "{secondary_key}"' if secondary_key else ""
    return (
        f"SELECT {keys}, {sk_sel}{bucket} + INTERVAL '1 {unit}' AS tile_end, {partials} "
        f"FROM {relation} GROUP BY {keys}{sk_grp}, {bucket}"
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
    agg_params: Optional[Dict[str, List[float]]] = None,
    secondary_key: Optional[str] = None,
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
    _assert_secondary_key_distinct(secondary_key, column_info, aggregations)
    secs = int(aggregation_interval.total_seconds())
    if secs not in _STREAMING_TILE_INTERVAL_SECS:
        raise ValueError(
            f"streaming tile aggregation_interval must be 1 hour (3600s) or 1 day (86400s); got {secs}s. "
            f"The TUMBLE grid is epoch-anchored and must match the offline date_trunc floor — week/arbitrary "
            f"intervals need an epoch-aligned offline floor, which is not yet supported."
        )
    keys = ", ".join(column_info.join_keys_columns)
    ts = column_info.timestamp_column
    partials = _tile_partials_projection(column_info, aggregations, agg_params)
    sk_sel = f'"{secondary_key}", ' if secondary_key else ""
    sk_grp = f', "{secondary_key}"' if secondary_key else ""
    return (
        f"SELECT {keys}, {sk_sel}window_end AS tile_end, {partials} "
        f"FROM tumble({relation}, {ts}, INTERVAL '{secs}' SECOND) "
        f"GROUP BY window_start, window_end, {keys}{sk_grp} EMIT ON WINDOW CLOSE"
    )


def build_cumulative_tile_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    tile_relation: str,
    *,
    agg_params: Optional[Dict[str, List[float]]] = None,
) -> str:
    """The CUMULATIVE-tile MV: per ``(entity, tile_end)``, the RUNNING TOTAL of each invertible partial
    over all tiles up to and including ``tile_end`` — ``sum(partial) OVER (PARTITION BY keys ORDER BY
    tile_end)``. ONE now()-free, window-agnostic MV from which the serving layer derives EVERY invertible
    window by 2-point asof subtraction: ``windowed = cum_at_T - cum_at_(T - window)``, ``lifetime =
    cum_at_T``, ``offset`` = shifted asof points, ``series`` = L points. It REPLACES the N per-(window,
    offset) + M per-floor lifetime now()-anchored rollup MVs for invertible aggregations (sum/count/mean/
    var/stddev), collapsing them to one MV.

    Reads the tiles MV (``build_*_tile_select`` output: bare per-(entity, tile_end) partials), so it is
    source-agnostic (batch or streaming tiles). now() is deliberately ABSENT — request-time anchoring
    happens in the READ query's asof bounds (``tile_end <= now()`` / ``<= now() - window``), which is why
    a single window-agnostic MV serves all windows AND sidesteps RisingWave's restriction that now() may
    appear only in a streaming MV's WHERE/HAVING. ``tile_end`` is unique per entity in the tiles MV (one
    row per (entity, tile_end)), so the default ORDER BY frame is an exact prefix cumulative sum. The
    running-total form ``sum(_) OVER (PARTITION BY _ ORDER BY tile_end)`` is validated as incrementally
    maintained under out-of-order tiles on RisingWave (see spike/verify_cumulative_maintenance.py)."""
    invertible = [a for a in aggregations if is_invertible_agg(a)]
    if not invertible:
        raise ValueError(
            "build_cumulative_tile_select requires at least one invertible aggregation "
            "(sum/count/mean/var/stddev); min/max/count_distinct/sequence keep the interval read."
        )
    keys = ", ".join(column_info.join_keys_columns)
    cum_cols = _cumulative_partials(aggregations, column_info.timestamp_column, agg_params)
    win = f"(PARTITION BY {keys} ORDER BY tile_end)"
    proj = ", ".join(f"{sql} OVER {win} AS {name}" for (name, sql) in cum_cols)
    return f"SELECT {keys}, tile_end, {proj} FROM {tile_relation}"


def build_series_snapshot_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    tile_relation: str,
    *,
    aggregation_interval,
    agg_params: Optional[Dict[str, List[float]]] = None,
    series: Optional[Dict[str, Sequence[int]]] = None,
) -> Optional[str]:
    """A per-entity LAST-L tile SNAPSHOT MV for the window-series whose step == the tile interval. Per
    entity it stores the last ``depth`` tiles' end timestamps (``SERIES_SNAPSHOT_ENDS_COL``) and, for each
    eligible series, that series' per-tile finalized value — two parallel ``array_agg(... ORDER BY tile_end
    DESC)`` over a ``row_number() <= depth`` per-entity TopN of the tiles MV. The online read becomes a
    single-row point lookup (vs the read-time range-scan single-scan), and the reader positions each
    (tile_end, value) into its frontier-relative slot (``now()`` online / ``label_ts`` offline),
    NULL-padding empty steps — index arithmetic, NOT a recombine, so the snapshot equals the offline
    single-scan element-for-element (proven on dense/gap/stale entities). Returns None when no series is
    snapshot-eligible.

    Reads the tiles MV (``build_*_tile_select`` output), so it is source-agnostic (batch or streaming
    tiles). now() is deliberately ABSENT — the snapshot is frontier-agnostic; anchoring happens in the
    reader. ``array_agg`` over a per-entity TopN is incrementally maintained on RisingWave (a new tile
    shifts the array). Snapshot-eligibility (step == interval, scalar finalizable function) is decided by
    ``snapshot_series_aggs``; coarser steps, overlapping windows, and array-valued aggregates keep the
    single-scan."""
    interval_secs = int(aggregation_interval.total_seconds())
    eligible = snapshot_series_aggs(aggregations, series, interval_secs)
    if not eligible:
        return None
    keys = ", ".join(column_info.join_keys_columns)
    depth = max(length for _a, length in eligible)
    names = [a.resolved_name(a.time_window) for a, _ in eligible]
    vals = ", ".join(
        f"{_tile_value_expr(a, agg_params=agg_params)} AS {name}"
        for (a, _), name in zip(eligible, names)
    )
    inner = (
        f"SELECT {keys}, tile_end, {vals}, "
        f"row_number() OVER (PARTITION BY {keys} ORDER BY tile_end DESC) AS __rn "
        f"FROM {tile_relation}"
    )
    # Each value array is trimmed to ITS OWN series length via a FILTER on the shared TopN, so a short
    # series in a view that also carries a longer one stores only its L values (not the view-wide max depth).
    # The shared tile_end array stays at the max depth — the reader zip-truncates it to each value's length,
    # so the (tile_end, value) pairing still aligns per tile. No reader change.
    arrs = ", ".join(
        f"array_agg({name} ORDER BY tile_end DESC) FILTER (WHERE __rn <= {length}) AS {name}"
        for (_a, length), name in zip(eligible, names)
    )
    # GROUP BY the join keys makes them the MV's primary/distribution key, so the serving read's
    # WHERE keys = ? is an index-only scan_ranges point lookup with no separate index — provided the reader
    # filters on the FULL join-key set (a partial-key read would degrade to a scan).
    return (
        f"SELECT {keys}, array_agg(tile_end ORDER BY tile_end DESC) AS {SERIES_SNAPSHOT_ENDS_COL}, {arrs} "
        f"FROM ({inner}) z WHERE __rn <= {depth} GROUP BY {keys}"
    )


def build_cumulative_read_query(
    entity_df_sql: str,
    entity_columns: List[str],
    label_ts_column: str,
    *,
    cumulative_relation: str,
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    aggregation_interval,
    full_feature_names: bool = False,
    view_name: Optional[str] = None,
    agg_params: Optional[Dict[str, List[float]]] = None,
    offsets: Optional[Dict[str, int]] = None,
    lifetimes: Optional[Dict[str, Optional[int]]] = None,
    series: Optional[Dict[str, Sequence[int]]] = None,
) -> str:
    """Read INVERTIBLE windowed features from the CUMULATIVE-tile MV by 2-point asof subtraction. For each
    entity-spine row it anchors ``end = date_trunc(aggregation_interval, label_ts)`` and derives every
    window by subtracting two cumulative rows fetched by an asof LATERAL (the latest ``tile_end <= bound``).
    Window-type -> asof bounds, IDENTICAL to ``build_offline_tile_pit_query`` (so cumulative == offline):
    trailing ``(end-W, end]``; offset ``(end-off-W, end-off]``; lifetime ``(floor, end]`` or ``(-inf, end]``;
    series = ``length`` offset-windows ``(end-W-i*step, end-i*step]`` oldest-first. ``label_ts`` is the
    spine's as-of column — ``now()`` for online serving, the label timestamp for an offline parity check —
    so ONE builder serves both. The Go feature server and Python client emit this SAME SQL shape (parity).

    Only invertible aggregations (sum/count/mean/var/stddev) belong here; non-invertible aggregations
    (min/max/count_distinct/sequence) are served from the interval tiles, not by subtraction."""
    if not aggregations:
        raise ValueError("build_cumulative_read_query requires at least one aggregation")
    if series:
        # A window-series is served by its own single-scan read (build_offline_tile_pit_query +
        # _series_recombine), not by the cumulative MV: assembling L windows as L stacked asof LATERALs
        # over the cumulative MV is an O(L)-deep correlated decorrelation that the RisingWave optimizer
        # cannot plan at series scale. The cumulative read stays for trailing/offset/lifetime only.
        raise ValueError(
            "build_cumulative_read_query does not serve a window-series; use build_offline_tile_pit_query "
            "(the single-scan series read)."
        )
    for a in aggregations:
        if not is_invertible_agg(a):
            raise ValueError(
                f"{a.function} on {a.column!r} is not invertible; it cannot be read from the cumulative "
                f"MV by subtraction — serve it from the interval tiles."
            )
    unit = _tile_unit(aggregation_interval)
    keys = column_info.join_keys_columns
    e_cols = [f'e."{c}"' for c in entity_columns]
    match = " AND ".join(f'c."{k}" = e."{k}"' for k in keys)
    end = f'date_trunc(\'{unit}\', e."{label_ts_column}")'
    cum_cols = ", ".join(
        name for (name, _sql) in _cumulative_partials(aggregations, column_info.timestamp_column, agg_params)
    )
    output_prefix = (view_name or "") if full_feature_names else ""

    # Dedupe the asof LATERAL joins by their bound expression: a trailing 7d and a series step that land on
    # the same boundary share one join. Each distinct bound becomes one ``aN`` LATERAL (latest tile <= bound).
    joins: dict = {}

    def asof(bound_sql: str) -> str:
        return joins.setdefault(bound_sql, f"a{len(joins)}")

    def back(secs: int) -> str:
        return end if int(secs) == 0 else f"{end} - INTERVAL '{int(secs)}' SECOND"

    def floor_bound(floor: int) -> str:
        return f"(to_timestamp({int(floor)}) AT TIME ZONE 'UTC')"

    projs = list(e_cols)
    for a in aggregations:
        name = a.resolved_name(a.time_window)
        out = f'"{output_prefix}__{name}"' if output_prefix else name
        if lifetimes and name in lifetimes:
            floor = lifetimes[name]
            lo = None if floor is None else asof(floor_bound(floor))
            projs.append(f"{_cumulative_recombine_expr(a, hi=asof(back(0)), lo=lo)} AS {out}")
        else:
            w = int(a.time_window.total_seconds())
            off = abs(_agg_offset_secs(a, offsets))
            projs.append(
                f"{_cumulative_recombine_expr(a, hi=asof(back(off)), lo=asof(back(off + w)))} AS {out}"
            )

    laterals = " ".join(
        f"LEFT JOIN LATERAL (SELECT {cum_cols} FROM {cumulative_relation} c WHERE {match} "
        f"AND c.tile_end <= {bound} ORDER BY c.tile_end DESC LIMIT 1) {alias} ON true"
        for bound, alias in joins.items()
    )
    return f"SELECT {', '.join(projs)} FROM ({entity_df_sql}) e {laterals}"


def build_tile_rollup_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    tile_relation: str,
    *,
    aggregation_interval,
    as_of_sql: str,
    agg_params: Optional[Dict[str, List[float]]] = None,
    offset_secs: int = 0,
) -> str:
    """Roll up tiles to the requested window, ANCHORED TO THE REQUEST/LABEL time (a request-anchored
    sliding window over a fixed tile set). Recombine each aggregation's per-tile
    partial with its rollup combiner (sum/min/max). The window is ``(end - time_window, end]`` where
    ``end = date_trunc(aggregation_interval, as_of)`` = the most-recent aggregation_interval boundary
    at or before the request/label time. ``as_of_sql`` is a SQL expression: a bind placeholder for
    online serving, or the entity-row timestamp column for offline PIT. ``tile_end`` carries the
    event-time PIT boundary, so there is no future leakage.

    ``offset_secs`` (<= 0) shifts the window into the past off the SAME ``end`` anchor — the window
    becomes ``(end - W - |offset|, end - |offset|]``. offset=0 emits the exact un-shifted SQL."""
    if not aggregations:
        raise ValueError("build_tile_rollup_select requires at least one aggregation")
    window_secs = _validate_window_rollup(aggregations, aggregation_interval)
    _assert_offset_multiple_of_interval(offset_secs, aggregation_interval)
    unit = _tile_unit(aggregation_interval)
    keys = ", ".join(column_info.join_keys_columns)
    rollups = _tile_rollup_exprs(aggregations, agg_params=agg_params)
    end = f"date_trunc('{unit}', {as_of_sql})"
    off = abs(int(offset_secs))
    upper = end if off == 0 else f"{end} - INTERVAL '{off}' SECOND"
    return (
        f"SELECT {keys}, {rollups} FROM {tile_relation} "
        f"WHERE tile_end > {end} - INTERVAL '{window_secs + off}' SECOND AND tile_end <= {upper} "
        f"GROUP BY {keys}"
    )


def build_online_rollup_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    tile_relation: str,
    *,
    aggregation_interval,
    agg_params: Optional[Dict[str, List[float]]] = None,
    offset_secs: int = 0,
    secondary_key: Optional[str] = None,
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
    point-lookup orders by (one row per entity, so LIMIT 1 is that row).

    ``offset_secs`` (<= 0) shifts the whole window into the past — a 7d window at offset -7d is the
    PREVIOUS week ``(now-14d, now-7d]``. The lower bound deepens by ``|offset|`` and the upper bound
    retreats from ``now()`` to ``now() - |offset|`` — both still plain ``now() - const`` expressions, the
    same accepted-and-maintained class as the un-shifted window (the upper bound is simply a constant
    below now(), so aging tiles are evicted from the TOP of the window too). offset=0 emits the exact
    un-shifted SQL (byte-identical) so an existing MV is never needlessly re-materialized. The caller
    groups aggregations by (window, offset) so every aggregation in one MV shares this offset."""
    if not aggregations:
        raise ValueError("build_online_rollup_select requires at least one aggregation")
    window_secs = _validate_window_rollup(aggregations, aggregation_interval)
    _assert_offset_multiple_of_interval(offset_secs, aggregation_interval)
    keys = ", ".join(column_info.join_keys_columns)
    rollups = _tile_rollup_exprs(aggregations, agg_params=agg_params)
    off = abs(int(offset_secs))
    upper = "now()" if off == 0 else f"now() - INTERVAL '{off}' SECOND"
    where = f"tile_end > now() - INTERVAL '{window_secs + off}' SECOND AND tile_end <= {upper}"
    if secondary_key:
        return _wrap_rollup_secondary_key(keys, secondary_key, tile_relation, rollups, where, aggregations)
    return (
        f"SELECT {keys}, {rollups}, max(tile_end) AS window_end FROM {tile_relation} "
        f"WHERE {where} GROUP BY {keys}"
    )


def _wrap_rollup_secondary_key(keys, secondary_key, tile_relation, rollups, where, aggregations):
    """The nested secondary-key form shared by the online + lifetime rollup MVs: an INNER rollup per
    (entity, secondary_key) over the same WHERE, then an OUTER ``jsonb_object_agg`` per entity collapsing
    the secondary-key dimension into a per-aggregation Map. ``max(tile_end)`` is carried through both
    levels as the window_end PIT stamp."""
    out_names = [a.resolved_name(a.time_window) for a in aggregations]
    inner = (
        f'SELECT {keys}, "{secondary_key}", {rollups}, max(tile_end) AS window_end '
        f"FROM {tile_relation} WHERE {where} GROUP BY {keys}, \"{secondary_key}\""
    )
    maps = _secondary_key_map_projection(aggregations, secondary_key, out_names)
    return (
        f"SELECT {keys}, {maps}, max(window_end) AS window_end "
        f"FROM ({inner}) AS _sk GROUP BY {keys}"
    )


def build_lifetime_rollup_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    tile_relation: str,
    *,
    agg_params: Optional[Dict[str, List[float]]] = None,
    lifetime_start_secs: Optional[int] = None,
    secondary_key: Optional[str] = None,
) -> str:
    """Online rollup MV for LIFETIME aggregations over the tiles: a continuous RisingWave materialized
    view maintaining the ALL-HISTORY rollup for ``as_of = now()``, with the lower window bound DROPPED.

    The window is one-sided — ``WHERE tile_end <= now()`` (optionally floored at
    ``tile_end > <lifetime_start>`` when ``lifetime_start_secs`` is set). Validated on RisingWave v3.0.0:
    a one-sided ``now()`` upper bound is accepted in a CREATE MATERIALIZED VIEW and incrementally
    maintained (tiles admitted as ``now()`` crosses them; none ever evicted from the bottom). The recombine
    is identical to the windowed rollup — only the WHERE differs — so the SAME per-tile partials serve both.

    UNBOUNDED STATE: unlike the windowed rollup, a lifetime MV never evicts old tiles, so its per-entity
    aggregate-of-tiles state grows with the tile COUNT (not the event count — the tile model already
    compacts events into per-interval partials). ``max(tile_end) AS window_end`` is the point-in-time stamp
    the online point-lookup orders by (one row per entity)."""
    if not aggregations:
        raise ValueError("build_lifetime_rollup_select requires at least one aggregation")
    _assert_tile_supported(aggregations)
    _assert_distinct_output_names(aggregations)
    keys = ", ".join(column_info.join_keys_columns)
    rollups = _tile_rollup_exprs(aggregations, agg_params=agg_params)
    where = "tile_end <= now()"
    if lifetime_start_secs is not None:
        # The floor is an absolute instant (epoch seconds). ``AT TIME ZONE 'UTC'`` renders it as the UTC
        # wall-clock timestamp regardless of the session timezone, so it compares against tile_end (a
        # timestamp without time zone, bucketed from UTC event times) deterministically — the same form
        # the offline PIT uses, so the online and offline tile sets cannot diverge on the session tz.
        where += f" AND tile_end > (to_timestamp({int(lifetime_start_secs)}) AT TIME ZONE 'UTC')"
    if secondary_key:
        return _wrap_rollup_secondary_key(keys, secondary_key, tile_relation, rollups, where, aggregations)
    return (
        f"SELECT {keys}, {rollups}, max(tile_end) AS window_end FROM {tile_relation} "
        f"WHERE {where} GROUP BY {keys}"
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
    agg_params: Optional[Dict[str, List[float]]] = None,
    offsets: Optional[Dict[str, int]] = None,
    lifetimes: Optional[Dict[str, Optional[int]]] = None,
    series: Optional[Dict[str, Sequence[int]]] = None,
    secondary_key: Optional[str] = None,
) -> str:
    """Offline point-in-time training rollup for a tile BatchFeatureView. For EACH entity row, rolls
    the tiles up in the request-anchored window ``(end - W, end]`` where ``end = floor(label_ts,
    aggregation_interval)`` — anchored to THAT ROW's label timestamp (NOT a global now()).

    ``offsets`` (keyed by resolved_name, <= 0) shifts an aggregation's window into the past off the
    same ``end`` anchor: a shifted aggregation's window is ``(end - W - |offset|, end - |offset|]``, so
    its per-agg ``CASE`` gains an UPPER bound ``t.tile_end <= end - |offset|`` (an un-shifted aggregation
    keeps the lower-only CASE the join's ``<= end`` already caps). The join reads back to the deepest
    tile ANY aggregation needs — ``max(W + |offset|)`` — so every shifted/un-shifted window is covered in
    one pass. An absent/zero offset reproduces today's SQL byte-for-byte.

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
    _validate_windows(aggregations, aggregation_interval, lifetimes, series)
    for a in aggregations:
        if not is_lifetime_agg(a, lifetimes) and not is_series_agg(a, series):
            _assert_offset_multiple_of_interval(_agg_offset_secs(a, offsets), aggregation_interval)
    unit = _tile_unit(aggregation_interval)
    keys = column_info.join_keys_columns
    e_cols = ", ".join(f'e."{c}"' for c in entity_columns)
    join_on = " AND ".join(f't."{k}" = e."{k}"' for k in keys)
    end = f'date_trunc(\'{unit}\', e."{label_ts_column}")'
    output_prefix = (view_name or "") if full_feature_names else ""

    def _partial_filter(a: Aggregation):
        # A LIFETIME aggregation has no lower window bound: with no floor it recombines EVERY joined tile
        # (the join's `<= end` is its only bound, so no per-agg CASE — None reuses the raw partial); with a
        # floor it adds `tile_end > floor`. A WINDOWED aggregation keeps the (offset-aware) window CASE.
        if is_lifetime_agg(a, lifetimes):
            floor = (lifetimes or {})[a.resolved_name(a.time_window)]
            return (
                None
                if floor is None
                else f"t.tile_end > (to_timestamp({int(floor)}) AT TIME ZONE 'UTC')"
            )
        w = int(a.time_window.total_seconds())
        off = abs(_agg_offset_secs(a, offsets))
        lower = f"t.tile_end > {end} - INTERVAL '{w + off}' SECOND"
        # An un-shifted window keeps the lower-only CASE (the join's `<= end` already caps it). A shifted
        # window's upper edge sits below `end`, so it needs its own explicit upper bound.
        return lower if off == 0 else f"{lower} AND t.tile_end <= {end} - INTERVAL '{off}' SECOND"

    # A lifetime aggregation reads ALL history up to `end`, so when one is present the join drops its
    # lower bound (the per-agg CASE then narrows each windowed/floored aggregation). Otherwise the join
    # reads back only to the deepest tile any windowed aggregation needs: max(window + |offset|).
    def _join_depth(a: Aggregation) -> int:
        # How far back the join must read for this aggregation. A series reads to its DEEPEST step:
        # window + (length - 1) * step. A windowed aggregation reads window + |offset|.
        if is_series_agg(a, series):
            w, s, length = series[a.resolved_name(a.time_window)]
            return int(w) + (int(length) - 1) * int(s)
        return int(a.time_window.total_seconds()) + abs(_agg_offset_secs(a, offsets))

    def _recombine(a: Aggregation) -> str:
        if is_series_agg(a, series):
            w, s, length = series[a.resolved_name(a.time_window)]
            return _series_recombine(
                a, end_expr=end, window_secs=int(w), step_secs=int(s), length=int(length),
                prefix="t.", output_prefix=output_prefix, agg_params=agg_params,
            )
        return _tile_recombine(
            a, prefix="t.", partial_filter=_partial_filter(a),
            output_prefix=output_prefix, agg_params=agg_params,
        )

    has_lifetime = any(is_lifetime_agg(a, lifetimes) for a in aggregations)
    if has_lifetime:
        join_lower = ""
    else:
        max_lower = max(_join_depth(a) for a in aggregations)
        join_lower = f"t.tile_end > {end} - INTERVAL '{max_lower}' SECOND AND "
    rollups = ", ".join(_recombine(a) for a in aggregations)
    join = (
        f"FROM ({entity_df_sql}) e LEFT JOIN {tiles_relation} t ON {join_on} "
        f"AND {join_lower}t.tile_end <= {end}"
    )
    if secondary_key:
        # Per (entity row, secondary_key) recombine, then collapse the secondary-key dimension into a
        # per-aggregation Map per entity row. A LEFT-JOIN miss yields a NULL secondary_key the
        # jsonb_object_agg FILTER drops; the resulting empty map is mapped to NULL (the NULLIF in
        # _secondary_key_map_projection), matching the online absent-entity — so no train/serve skew.
        out_entity = ", ".join(f'"{c}"' for c in entity_columns)
        out_names = [
            f'"{output_prefix}__{a.resolved_name(a.time_window)}"'
            if output_prefix
            else a.resolved_name(a.time_window)
            for a in aggregations
        ]
        inner = (
            f'SELECT {e_cols}, t."{secondary_key}" AS "{secondary_key}", {rollups} {join} '
            f'GROUP BY {e_cols}, t."{secondary_key}"'
        )
        maps = _secondary_key_map_projection(aggregations, secondary_key, out_names)
        return f"SELECT {out_entity}, {maps} FROM ({inner}) AS _sk GROUP BY {out_entity}"
    return f"SELECT {e_cols}, {rollups} {join} GROUP BY {e_cols}"


def compose_multi_view_pit_query(
    per_view_queries: List[str],
    entity_columns: List[str],
    per_view_feature_cols: List[List[str]],
) -> str:
    """Combine each feature view's per-view point-in-time read into ONE training frame by LEFT JOINing
    them over the shared entity spine.

    Each per-view query (``build_offline_tile_pit_query`` or ``build_passthrough_pit_query``) projects the
    FULL entity-column set plus that view's feature columns, exactly one row per distinct entity-spine row —
    the tile builder GROUPs BY the entity columns, the passthrough builder PARTITIONs BY them. So the entity
    columns ARE the row identity: the first view anchors the spine (it LEFT JOINs from the entity rows, so it
    retains every one of them) and each remaining view LEFT JOINs back on an equality over all entity
    columns, contributing only its feature columns. A view with no as-of match for an entity row still yields
    that row (NULL features), so no entity row is dropped. ``per_view_feature_cols`` is the OUTPUT column
    name each per-view query emits for its features — already ``"{view}__{feature}"`` when the caller built
    the views with full_feature_names, so the entity columns (never prefixed) are the only shared columns.

    A single view returns its query verbatim, so the one-view read is unchanged by multi-view support."""
    if len(per_view_queries) == 1:
        return per_view_queries[0]

    # Without full_feature_names each view emits bare feature names, so two views projecting the same output
    # column would put a duplicate, ambiguous column in the joined frame. Refuse clearly instead of emitting
    # it; the join only requires the feature names across views not to clash. (full_feature_names prefixes
    # each name with its view, so this never trips there.)
    owner_of: dict = {}
    for view_index, feature_cols in enumerate(per_view_feature_cols):
        for col in feature_cols:
            if col in owner_of:
                raise NotImplementedError(
                    f"feature views at positions {owner_of[col]} and {view_index} both project a feature "
                    f"column '{col}'; RisingWave offline retrieval cannot join feature views with colliding "
                    "feature names in one call. Request full_feature_names, give the features distinct "
                    "names, or retrieve the views separately."
                )
            owner_of[col] = view_index

    aliases = [f"_feast_view_{i}" for i in range(len(per_view_queries))]
    ctes = ", ".join(f'"{alias}" AS ({query})' for alias, query in zip(aliases, per_view_queries))
    anchor = aliases[0]
    select_cols = [f'"{anchor}"."{c}"' for c in entity_columns]
    for alias, feature_cols in zip(aliases, per_view_feature_cols):
        select_cols.extend(f'"{alias}"."{c}"' for c in feature_cols)
    # Plain equi-join on the entity columns: Feast join keys and the label timestamp are non-NULL
    # identifiers, so equality is the right row identity AND keeps the join hashable (a NULL-safe
    # predicate would force a nested loop). A NULL value in an entity column would not match here.
    joins = ""
    for alias in aliases[1:]:
        on = " AND ".join(f'"{anchor}"."{c}" = "{alias}"."{c}"' for c in entity_columns)
        joins += f' LEFT JOIN "{alias}" ON {on}'
    return f'WITH {ctes} SELECT {", ".join(select_cols)} FROM "{anchor}"{joins}'
