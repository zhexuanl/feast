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

import json
from typing import Dict, List, Optional, Sequence, Tuple

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

# Sequence (Array-valued) aggregates -> the ORDER BY direction of their array_agg: last* keep the
# n MOST-RECENT (event-time DESC), first* the n EARLIEST (ASC). The _distinct variants wrap
# array_distinct (which preserves first occurrence, so most-recent/earliest distinct is kept).
_SEQUENCE_ORDER = {
    "last": "DESC",
    "last_distinct": "DESC",
    "first": "ASC",
    "first_distinct": "ASC",
}

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


# Per-aggregation numeric parameters that feast.Aggregation cannot carry (it has no parameter
# field): the quantile of approx_percentile, the N of a sequence aggregate. They ride on the feature
# view's tags under this engine-owned key, as a JSON map keyed by each aggregation's resolved_name ->
# ordered params. The carrier is owned by this engine (its only reader): an authoring layer populates
# it via ``encode_agg_params`` and the builders read it via ``view_agg_params``, so a re-applied /
# registry-rehydrated view reproduces byte-identical SQL (the verbatim-catalog reconcile keys on the
# rendered SELECT). The key is deliberately engine-namespaced — no upstream-layer name is baked in.
AGG_PARAMS_TAG = "feast_rw_agg_params"


def encode_agg_params(
    params_by_resolved_name: Dict[str, Sequence[float]],
) -> Dict[str, str]:
    """The view-tags fragment carrying per-aggregation numeric parameters, keyed by resolved_name.
    Drops empty entries and returns ``{}`` when there is nothing to carry, so a parameter-free view's
    tags are left untouched. The inverse of ``view_agg_params``."""
    cleaned = {
        name: [float(p) for p in params]
        for name, params in params_by_resolved_name.items()
        if params
    }
    return {AGG_PARAMS_TAG: json.dumps(cleaned)} if cleaned else {}


def view_agg_params(view) -> Dict[str, List[float]]:
    """The per-aggregation numeric parameters a view carries in its tags, keyed by resolved_name.
    Absent => {} (the common case: every aggregation is parameter-free). The inverse of
    ``encode_agg_params``."""
    raw = (getattr(view, "tags", None) or {}).get(AGG_PARAMS_TAG)
    if not raw:
        return {}
    return {name: list(params) for name, params in json.loads(raw).items()}


# The per-aggregation window OFFSET (a shift of the rollup window into the past, in whole negative
# seconds), keyed by each aggregation's resolved_name. Like the parameters above, feast.Aggregation
# has no field for it, so it rides this engine-owned tag — a carrier parallel to AGG_PARAMS_TAG. The
# SAME tag mechanism serves BOTH tile flavors: a batch tile view's aggregations live on its
# IcebergSource and a streaming tile view's on the StreamFeatureView proto, but both carry their tags
# through the registry, so one offset carrier covers both (no per-flavor split, no proto fork). A zero
# offset (the trailing window, the default) is omitted, so a non-shifted view's tags are untouched.
AGG_OFFSET_TAG = "feast_rw_agg_offset"


def encode_agg_offsets(offsets_by_resolved_name: Dict[str, int]) -> Dict[str, str]:
    """The view-tags fragment carrying per-aggregation window offsets (whole seconds), keyed by
    resolved_name. Drops zero/empty entries and returns ``{}`` when nothing is shifted, so a
    trailing-window-only view's tags are left untouched. The inverse of ``view_agg_offsets``."""
    cleaned = {name: int(secs) for name, secs in offsets_by_resolved_name.items() if secs}
    return {AGG_OFFSET_TAG: json.dumps(cleaned)} if cleaned else {}


def view_agg_offsets(view) -> Dict[str, int]:
    """The per-aggregation window offsets (whole seconds, negative = shifted into the past) a view
    carries in its tags, keyed by resolved_name. Absent => {} (the common case: every aggregation is a
    trailing window). The inverse of ``encode_agg_offsets``."""
    raw = (getattr(view, "tags", None) or {}).get(AGG_OFFSET_TAG)
    if not raw:
        return {}
    return {name: int(secs) for name, secs in json.loads(raw).items()}


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


def _partials_for(
    agg: Aggregation,
    ts_col: Optional[str] = None,
    agg_params: Optional[Dict[str, List[float]]] = None,
) -> List[Tuple[str, str]]:
    """The WINDOW-INDEPENDENT per-tile partial columns (name, SQL aggregate) one aggregation needs.
    Named by (function-family, column) so multiple windows / functions on the same column share a
    partial. Additive functions need ONE partial; composite (mean/var/stddev) need the additive
    sub-partials that merge across tiles; count_distinct stores the tile's distinct set; sequence
    aggregates store the tile's bounded top-n (``ts_col`` orders it, ``n`` from agg_params)."""
    col, fn = agg.column, agg.function
    if fn == "sum":
        return [(f"sum_{col}", f"sum({col})")]
    if fn == "count":
        return [(f"count_{col}", f"count({col})")]
    if fn in {"min", "max"}:
        return [(f"{fn}_{col}", f"{fn}({col})")]
    if fn in _SEQUENCE_ORDER:
        # The tile's OWN top-n ordered array (bounded). n is part of the partial (the slice + the
        # name), so last(3) and last(5) on one column are distinct partials.
        n = _sequence_n(agg, agg_params)
        ordered = f"array_agg({col} ORDER BY {ts_col} {_SEQUENCE_ORDER[fn]})"
        if fn.endswith("_distinct"):
            ordered = f"array_distinct({ordered})"
        return [(f"{fn}_{col}_{n}", f"({ordered})[1:{n}]")]
    if fn == "count_distinct":
        # The tile's DISTINCT SET. FILTER out NULL so the union+count matches count(distinct <col>),
        # which excludes NULL (RisingWave's array_agg(DISTINCT) would otherwise carry a NULL element).
        return [
            (
                f"distinct_{col}",
                f"array_agg(DISTINCT {col}) FILTER (WHERE {col} IS NOT NULL)",
            )
        ]
    partials = [(f"sum_{col}", f"sum({col})"), (f"count_{col}", f"count({col})")]
    if fn in {"var_pop", "var_samp", "stddev_pop", "stddev_samp"}:
        partials.append((f"sumsq_{col}", f"sum({col} * {col})"))
    return partials


def _view_partials(
    aggregations: List[Aggregation],
    ts_col: Optional[str] = None,
    agg_params: Optional[Dict[str, List[float]]] = None,
) -> List[Tuple[str, str]]:
    """The deduped union of every aggregation's window-independent partials = the tiles MV's partial
    columns. ``setdefault`` keeps one entry per partial name (the materialize-SQL is identical for a
    given partial name, so dedup is safe — a sequence partial's name carries its n + column)."""
    out: dict = {}
    for a in aggregations:
        for name, sql in _partials_for(a, ts_col, agg_params):
            out.setdefault(name, sql)
    return list(out.items())


def _tile_recombine(
    agg: Aggregation,
    *,
    prefix: str = "",
    partial_filter: Optional[str] = None,
    output_prefix: str = "",
    agg_params: Optional[Dict[str, List[float]]] = None,
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
    if fn == "count_distinct":
        # Union the per-tile distinct sets and count: concat the tile arrays (array_flatten skips the
        # NULL inner arrays a multi-window CASE produces), dedup, count. NULLIF(_, 0) maps an empty
        # window (no tiles, or only NULL values) to NULL, so the offline LEFT-JOIN result matches the
        # online MV — where such an entity is simply absent.
        sets = merged("distinct", "array_agg")
        return f"NULLIF(cardinality(array_distinct(array_flatten({sets}))), 0) AS {out}"
    if fn in _SEQUENCE_ORDER:
        # Concat the per-tile top-n arrays in tile_end order (array_flatten preserves it AND skips the
        # NULL inner arrays a multi-window CASE produces), then slice to n. Each tile array is already
        # ordered within the tile, so the flattened order is the global event-time order; the _distinct
        # variants dedup again across tiles (a value may repeat in two tiles).
        n = _sequence_n(agg, agg_params)
        p = f"{prefix}{fn}_{col}_{n}"
        inner = f"CASE WHEN {partial_filter} THEN {p} END" if partial_filter else p
        flat = f"array_flatten(array_agg({inner} ORDER BY {prefix}tile_end {_SEQUENCE_ORDER[fn]}))"
        if fn.endswith("_distinct"):
            flat = f"array_distinct({flat})"
        return f"({flat})[1:{n}] AS {out}"
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


def _validate_windows(aggregations: List[Aggregation], aggregation_interval) -> List[int]:
    """Multi-window precondition for the offline PIT builder: tile-supported aggs only, distinct output
    names, and EVERY aggregation's (non-null) window a whole multiple of the interval. Returns the
    DISTINCT window seconds ascending. The aggregations may carry different windows over the ONE shared
    tile set (tiles reused across time-windows), so unlike ``_validate_window_rollup`` this does
    NOT require a single window."""
    _assert_tile_supported(aggregations)
    _assert_distinct_output_names(aggregations)
    # group_aggregations_by_window owns the non-null-window check + the distinct-ascending window set
    # (the multiple-of-interval precondition is offset-independent, so the window-only grouping is what
    # this needs); here we only layer on the multiple-of-interval check the rollup requires.
    windows = [secs for secs, _ in group_aggregations_by_window(aggregations)]
    for secs in windows:
        _assert_window_multiple_of_interval(secs, aggregation_interval)
    return windows


def group_aggregations_by_window(
    aggregations: List[Aggregation],
) -> List[Tuple[int, List[Aggregation]]]:
    """Group aggregations by their (distinct, non-null) window seconds, ascending. The window-only
    grouping, used where the offset does not matter (the offline multiple-of-interval precondition, and
    the no-offset spike harnesses). The offset-aware ``group_aggregations_by_window_offset`` is what the
    online MV provisioning + serving-shard derivation use, since a shifted window needs its own MV."""
    groups: dict = {}
    for a in aggregations:
        if a.time_window is None:
            raise ValueError(
                "tile rollup needs a non-null time_window on every aggregation; "
                f"got None on {a.function}({a.column})."
            )
        groups.setdefault(int(a.time_window.total_seconds()), []).append(a)
    return [(secs, groups[secs]) for secs in sorted(groups)]


def _agg_offset_secs(agg: Aggregation, offsets: Optional[Dict[str, int]]) -> int:
    """This aggregation's window offset (whole seconds, <= 0 = shifted into the past) from the
    out-of-band offsets map keyed by resolved_name. Absent => 0 (the trailing window)."""
    return int((offsets or {}).get(agg.resolved_name(agg.time_window), 0))


def group_aggregations_by_window_offset(
    aggregations: List[Aggregation],
    offsets: Optional[Dict[str, int]] = None,
) -> List[Tuple[Tuple[int, int], List[Aggregation]]]:
    """Group aggregations by their ``(window_secs, offset_secs)`` pair, ascending. Two aggregations
    sharing a window but differing in offset (a trailing 7d and the previous week) cannot share one
    now()-anchored online MV — the rollup WHERE bounds differ — so each (window, offset) pair becomes its
    OWN online rollup MV and OnlineView shard. The offset rides the out-of-band carrier keyed by
    resolved_name (``offsets``); absent => 0, so an all-trailing view groups one shard per window. The
    engine (provisioning) and apply (serving spec) MUST group identically from THIS helper, so the
    per-(window, offset) MV names cannot drift."""
    groups: dict = {}
    for a in aggregations:
        if a.time_window is None:
            raise ValueError(
                "tile rollup needs a non-null time_window on every aggregation; "
                f"got None on {a.function}({a.column})."
            )
        key = (int(a.time_window.total_seconds()), _agg_offset_secs(a, offsets))
        groups.setdefault(key, []).append(a)
    return [(key, groups[key]) for key in sorted(groups)]


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
    partials = _tile_partials_projection(column_info, aggregations, agg_params)
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
    agg_params: Optional[Dict[str, List[float]]] = None,
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
    partials = _tile_partials_projection(column_info, aggregations, agg_params)
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
    return (
        f"SELECT {keys}, {rollups}, max(tile_end) AS window_end FROM {tile_relation} "
        f"WHERE tile_end > now() - INTERVAL '{window_secs + off}' SECOND AND tile_end <= {upper} "
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
    agg_params: Optional[Dict[str, List[float]]] = None,
    offsets: Optional[Dict[str, int]] = None,
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
    _validate_windows(aggregations, aggregation_interval)
    for a in aggregations:
        _assert_offset_multiple_of_interval(_agg_offset_secs(a, offsets), aggregation_interval)
    unit = _tile_unit(aggregation_interval)
    keys = column_info.join_keys_columns
    e_cols = ", ".join(f'e."{c}"' for c in entity_columns)
    join_on = " AND ".join(f't."{k}" = e."{k}"' for k in keys)
    end = f'date_trunc(\'{unit}\', e."{label_ts_column}")'
    output_prefix = (view_name or "") if full_feature_names else ""

    def _partial_filter(a: Aggregation) -> str:
        w = int(a.time_window.total_seconds())
        off = abs(_agg_offset_secs(a, offsets))
        lower = f"t.tile_end > {end} - INTERVAL '{w + off}' SECOND"
        # An un-shifted window keeps the lower-only CASE (the join's `<= end` already caps it). A shifted
        # window's upper edge sits below `end`, so it needs its own explicit upper bound.
        return lower if off == 0 else f"{lower} AND t.tile_end <= {end} - INTERVAL '{off}' SECOND"

    # The join reads back to the deepest tile any aggregation needs: max(window + |offset|).
    max_lower = max(
        int(a.time_window.total_seconds()) + abs(_agg_offset_secs(a, offsets))
        for a in aggregations
    )
    rollups = ", ".join(
        _tile_recombine(
            a,
            prefix="t.",
            partial_filter=_partial_filter(a),
            output_prefix=output_prefix,
            agg_params=agg_params,
        )
        for a in aggregations
    )
    return (
        f"SELECT {e_cols}, {rollups} FROM ({entity_df_sql}) e "
        f"LEFT JOIN {tiles_relation} t ON {join_on} "
        f"AND t.tile_end > {end} - INTERVAL '{max_lower}' SECOND AND t.tile_end <= {end} "
        f"GROUP BY {e_cols}"
    )


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
            agg_params=view_agg_params(self.view),
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
