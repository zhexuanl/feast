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

Every SQL fragment traces to a verified RisingWave e2e example (cited inline).
Anything not yet verified end-to-end is marked ``UNVERIFIED`` / spike-gated (see
``README.md`` → "Spike-gated").
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
#     crash-RECOVERY state bug for updatable approx_count_distinct
#     (e2e_test/streaming/aggregate/approx_count_distinct.slt.bug) — harmless for our
#     append-only EOWC model (the source never retracts), but flagged.
# Deliberately EXCLUDED — rejected at apply with a reason, not silently:
#   - first(n) / last(n) / first_distinct / last_distinct (sequence features): RisingWave has
#     no bare first()/last() aggregate; they need ordered-set / Array outputs (a later phase).
#   - approx_percentile: parameterized (takes the percentile) — needs a parameter field on
#     feast.Aggregation, which has none today (a later phase).
#   - aggregation_secondary_key: produces a per-secondary-key breakdown = an Array/Map output
#     that the scalar engine and the ServingSpec wire do not carry yet (a later phase).
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
# (monoid) split, api.thrift:156-172. sum/count/mean and stddev/variance are Abelian-group
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
    # aggregation in one MV must share a single window. Verified:
    # time_window.slt:33-36 (TUMBLE), time_window.slt:45-48 (HOP; 3rd arg = slide,
    # 4th arg = size).
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

    ``emit_on_close`` appends ``EMIT ON WINDOW CLOSE`` (eowc_group_agg.slt:18-23),
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
                "(Chronon api.thrift:156-164). Use an append-only source, or only "
                "Abelian-group ops (sum/count/mean)."
            )

    keys = ", ".join(column_info.join_keys_columns)
    exprs = ", ".join(_agg_expr(a) for a in aggregations)
    windowed = aggregations[0].time_window is not None
    src = _window_relation(aggregations, column_info.timestamp_column, relation)

    if windowed:
        # window_END is the row's event timestamp: a window [t, t+w) is only knowable
        # at t+w, so an as-of (<=) join never sees a window before it closes
        # (time_window.slt:50-61). Timestamping by window_start would leak the full
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


# --- Batch tile aggregation (established feature stores' aggregation engine, tile model) ---------------------
# A BATCH feature view materializes PARTIAL aggregates at the aggregation_interval (tiles), then
# rolls them up to the requested window AT RETRIEVAL, anchored to the request/label time. This is
# distinct from the streaming TUMBLE path above: tiles are a plain batch GROUP BY over a batch
# relation (e.g. an Iceberg source) and one fixed tile set serves any window size, sliding with the
# request time. Verified end-to-end live on RW v3.0.0: spike/sql/05c_batch_tiles.sql.

# The tile model materializes per-(entity, tile) PARTIALS that recombine additively across the tiles
# in a window. Two families:
#   ADDITIVE — one partial == the aggregate; recombine: sum/sum/min/max (count rolls up by SUMMING
#     per-tile counts). Named by the feature's resolved_name so tile/rollup/serve share one column.
#   COMPOSITE — the aggregate is NOT additive, but decomposes into additive partials that DO merge
#     and a recombine formula (Chronon's IR: Average = {sum, count}; Variance via {sum, sumsq, count},
#     var = (Σx² − (Σx)²/n)/n). Partials are named ``<resolved>__sm/__cnt/__sqs``.
# count_distinct/approx have no safe additive sketch merge — still rejected.
_ADDITIVE_TILE_FN = frozenset({"sum", "count", "min", "max"})
_COMPOSITE_TILE_FN = frozenset({"mean", "var_pop", "var_samp", "stddev_pop", "stddev_samp"})
_TILE_SUPPORTED_FN = _ADDITIVE_TILE_FN | _COMPOSITE_TILE_FN


def _tile_partials(agg: Aggregation) -> List[Tuple[str, str]]:
    """The per-tile partial columns (name, SQL aggregate) for one aggregation. Additive functions
    have ONE partial (the aggregate itself); composite functions (mean/var/stddev) have the additive
    sub-partials that merge across tiles."""
    out = agg.resolved_name(agg.time_window)
    col = agg.column
    fn = agg.function
    if fn in _ADDITIVE_TILE_FN:
        return [(out, f"{fn}({col})")]
    partials = [(f"{out}__sm", f"sum({col})"), (f"{out}__cnt", f"count({col})")]
    if fn in {"var_pop", "var_samp", "stddev_pop", "stddev_samp"}:
        partials.append((f"{out}__sqs", f"sum({col} * {col})"))
    return partials


def _tile_recombine(agg: Aggregation, prefix: str = "") -> str:
    """The retrieval-time recombine for one aggregation: an expression over its per-tile partials
    aliased to the FINAL ``resolved_name``. ``prefix`` qualifies the partial columns for a joined
    relation (``"t."`` in the offline PIT range-join)."""
    out = agg.resolved_name(agg.time_window)
    fn = agg.function
    if fn in {"sum", "count"}:  # count recombines by SUMMING per-tile counts
        return f"sum({prefix}{out}) AS {out}"
    if fn in {"min", "max"}:
        return f"{fn}({prefix}{out}) AS {out}"
    sm, cnt = f"sum({prefix}{out}__sm)", f"sum({prefix}{out}__cnt)"
    if fn == "mean":
        return f"{sm} / NULLIF({cnt}, 0) AS {out}"
    # variance/stddev: (Σx² − (Σx)²/n) / n  (population) or / (n−1) (sample); stddev = sqrt(var)
    centered = f"(sum({prefix}{out}__sqs) - {sm} * {sm} / NULLIF({cnt}, 0))"
    denom = f"NULLIF({cnt} - 1, 0)" if fn.endswith("_samp") else f"NULLIF({cnt}, 0)"
    var = f"{centered} / {denom}"
    return f"sqrt({var}) AS {out}" if fn.startswith("stddev") else f"{var} AS {out}"

# aggregation_interval (the tile size) -> RisingWave date_trunc unit. Standard units only for now
# (date_trunc is what the spike validated); arbitrary intervals (e.g. 15min) need epoch-bucketing.
_TILE_INTERVAL_UNIT = {3600: "hour", 86400: "day", 604800: "week"}


def _tile_unit(aggregation_interval) -> str:
    unit = _TILE_INTERVAL_UNIT.get(int(aggregation_interval.total_seconds()))
    if unit is None:
        raise ValueError(
            f"aggregation_interval {aggregation_interval} is not supported yet: the batch tile "
            f"builder buckets with date_trunc, so it must be 1 "
            f"{'/'.join(sorted(_TILE_INTERVAL_UNIT.values()))}. Arbitrary intervals need "
            f"epoch-bucketing (a later phase)."
        )
    return unit


def _assert_tile_supported(aggregations: List[Aggregation]) -> None:
    # The tile model supports any aggregation that recombines from additive partials: sum/count/min/max
    # directly, and mean/var/stddev via composite partials (Chronon's IR). count_distinct/approx have
    # no safe additive sketch merge across tiles (cf. Chronon's monoid split, api.thrift) — rejected.
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
            f"(multiple windows from one tile set is a later phase)."
        )
    return int(next(iter(windows)).total_seconds())


def _assert_window_multiple_of_interval(window_secs: int, aggregation_interval) -> None:
    # The window is a COUNT of tiles, so it must be a whole number of aggregation_intervals. This is
    # also what makes the online now()-anchored rollup equal the offline floor-anchored rollup:
    # for interval-boundary tiles, (now - W, now] selects the SAME tiles as
    # (floor(now, interval) - W, floor(now, interval)] only when W is a multiple of the interval.
    interval_secs = int(aggregation_interval.total_seconds())
    if window_secs % interval_secs != 0:
        raise ValueError(
            f"time_window ({window_secs}s) must be a whole multiple of aggregation_interval "
            f"({interval_secs}s) for the tile model (the window is a count of tiles)."
        )


def _validate_window_rollup(aggregations: List[Aggregation], aggregation_interval) -> int:
    """Shared precondition for the three rollup builders (offline floored, online now(), offline PIT):
    tile-supported aggs only, a single window, and window a whole multiple of the interval. Returns
    window_secs. Centralized so the online and offline rollups CANNOT validate differently (ADR-0004)."""
    _assert_tile_supported(aggregations)
    window_secs = _single_window_secs(aggregations)
    _assert_window_multiple_of_interval(window_secs, aggregation_interval)
    return window_secs


def _tile_rollup_exprs(aggregations: List[Aggregation], prefix: str = "") -> str:
    """The per-aggregation recombine projection, shared by ALL rollup builders so online and offline
    recombine per-tile partials IDENTICALLY (ADR-0004 no-drift — a single source of truth, via
    ``_tile_recombine``). ``prefix`` qualifies the partial columns for a joined relation (``"t."`` in
    the offline PIT range-join)."""
    return ", ".join(_tile_recombine(a, prefix) for a in aggregations)


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
    # additive functions emit one partial (the aggregate); composite (mean/var/stddev) emit several.
    partials = ", ".join(
        f"{expr} AS {name}" for a in aggregations for (name, expr) in _tile_partials(a)
    )
    return (
        f"SELECT {keys}, {bucket} + INTERVAL '1 {unit}' AS tile_end, {partials} "
        f"FROM {relation} GROUP BY {keys}, {bucket}"
    )


def build_tile_rollup_select(
    column_info: ColumnInfo,
    aggregations: List[Aggregation],
    tile_relation: str,
    *,
    aggregation_interval,
    as_of_sql: str,
) -> str:
    """Roll up tiles to the requested window, ANCHORED TO THE REQUEST/LABEL time (established feature stores'
    request-anchored sliding window over a fixed tile set). Recombine each aggregation's per-tile
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
    ``build_tile_rollup_select``'s ``date_trunc(now())`` form. Verified live on RW v3.0.0: a two-sided
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
) -> str:
    """Offline point-in-time training rollup for a tile BatchFeatureView. For EACH entity row, rolls
    the tiles up in the request-anchored window ``(end - W, end]`` where ``end = floor(label_ts,
    aggregation_interval)`` — anchored to THAT ROW's label timestamp (NOT a global now()).

    This CANNOT reuse Feast's standard latest-row PIT template: that picks the latest tile <= label,
    which anchors the window at the latest tile WITH DATA, not at floor(label) — they diverge when the
    most recent intervals have no events (verified live: 180/280/230 vs a wrong 180/280/280). So we
    range-JOIN the inlined entity rows to the tiles and GROUP BY the entity row. LEFT JOIN so a row
    with no tiles in range still appears (NULL feature). Single tile feature view per query for now.

    TTL note: the feature view's ``ttl`` is intentionally NOT applied as a second lower bound. For an
    aggregation feature view the ``time_window`` IS the lookback bound (Chronon semantics); a
    ttl shorter than the window would silently shrink the aggregation below what the user requested, and
    a longer ttl is a no-op. So the window is the single, authoritative bound."""
    window_secs = _validate_window_rollup(aggregations, aggregation_interval)
    unit = _tile_unit(aggregation_interval)
    keys = column_info.join_keys_columns
    e_cols = ", ".join(f'e."{c}"' for c in entity_columns)
    join_on = " AND ".join(f't."{k}" = e."{k}"' for k in keys)
    end = f'date_trunc(\'{unit}\', e."{label_ts_column}")'
    rollups = _tile_rollup_exprs(aggregations, prefix="t.")
    return (
        f"SELECT {e_cols}, {rollups} FROM ({entity_df_sql}) e "
        f"LEFT JOIN {tiles_relation} t ON {join_on} "
        f"AND t.tile_end > {end} - INTERVAL '{window_secs}' SECOND AND t.tile_end <= {end} "
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
            # Spike-gated: a pandas entity_df must be staged into RisingWave (e.g. a
            # temporary table / VALUES list) before it can be joined over pgwire. We
            # reference a conventional staging relation here; the spike must implement
            # the upload (mirrors Flink's pandas_to_flink_table staging).
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
                "(spike-gated). Pre-transform upstream, or use a SQL transformation."
            )
        # The SQL transform replaces the projection over the input relation; the
        # output column set is the view's declared features (spike-gated: we trust the
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
    pulling data out of RisingWave and is spike-gated.)"""

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
            # UNVERIFIED end-to-end (spike risk #1/#5): the bounded backfill INSERT and
            # its late-data parity with the live stream are not yet proven in-repo.
            # Preferred long-term: read the live sink's Iceberg history so backfill ==
            # what was served (risk #8). The bounded [start, end) predicate is applied
            # by the upstream filter node before this INSERT.
            staging = _quote(offline_staging_name(context.project, self.view.name))
            sql = f"INSERT INTO {staging} {select_sql}"

        return self._value(
            relation, columns, metadata={**(input_value.metadata or {}), "sql": sql}
        )
