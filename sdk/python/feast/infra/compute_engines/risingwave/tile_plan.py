"""TilePlan — the structured materialization plan for a tile feature view's online serving.

The tile model serves a feature view's aggregations from a SET of standing materialized views: ONE tiles MV
holding the window-independent per-(entity, tile) partials, plus the online rollups derived from it — ONE
cumulative MV (invertible aggs, served by 2-point asof subtraction), a now()-anchored rollup MV per
(window, offset) and per lifetime floor (non-invertible aggs), and at most one per-entity window-series
snapshot MV. Which of those exist for a given view, and how they map to MV names, is the composition/grouping
logic that used to be open-coded inside ``_desired_online_mvs`` and re-derived (by hand, riskily) in the drop
path, the offline read, and the serving-shard derivation.

TilePlan makes that one structured object. It OWNS the composition (the invertible/interval/series/lifetime
classification, the (window, offset)/floor/snapshot split, the resolved-name -> MV-name mapping); it CALLS
the leaf SQL builders UNCHANGED (``build_*_tile_select`` / ``build_online_rollup_select`` /
``build_lifetime_rollup_select`` / ``build_cumulative_tile_select`` / ``build_series_snapshot_select``), so
the emitted SQL is byte-identical to today — the refactor changes structure, not SQL. Every node renders its
own SQL behind ``.select()``, so the codegen substrate (today f-strings) is isolated to one place per node.

Semantic model (not an implementation borrowed from any engine): a feature view is a set of aggregations
over an entity key; each decomposes into mergeable per-tile partials (the partial-aggregate IR in
``tiling``); windows are recombines over those partials; invertible aggregations additionally admit a
running-total (cumulative) representation served by subtraction. TilePlan is the physical plan of that model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from feast.aggregation import Aggregation
from feast.infra.compute_engines.dag.context import ColumnInfo
from feast.infra.compute_engines.risingwave.aggregation_carriers import (
    group_aggregations_by_window_offset,
    group_lifetime_aggregations,
    is_lifetime_agg,
    is_series_agg,
)
from feast.infra.compute_engines.risingwave.names import (
    online_cumulative_mv_name,
    online_lifetime_mv_name,
    online_series_mv_name,
    online_window_mv_name,
    tiles_name,
)
from feast.infra.compute_engines.risingwave.sql_builders import (
    build_batch_tile_select,
    build_cumulative_tile_select,
    build_lifetime_rollup_select,
    build_online_rollup_select,
    build_series_snapshot_select,
    build_streaming_tile_select,
)
from feast.infra.compute_engines.risingwave.tiling import is_invertible_agg


@dataclass(frozen=True)
class TilesNode:
    """The tiles MV: per-(entity, tile_end) window-independent partials. Batch buckets with date_trunc;
    streaming materializes the SAME partials by an EOWC TUMBLE over a watermarked source."""

    name: str
    column_info: ColumnInfo
    aggregations: List[Aggregation]
    source_relation: str
    flavor: str  # "batch" | "streaming"
    aggregation_interval: object
    agg_params: Optional[Dict[str, List[float]]] = None
    secondary_key: Optional[str] = None
    filters: Optional[Dict[str, str]] = None

    def select(self) -> str:
        builder = build_batch_tile_select if self.flavor == "batch" else build_streaming_tile_select
        return builder(
            self.column_info, self.aggregations, self.source_relation,
            aggregation_interval=self.aggregation_interval, agg_params=self.agg_params,
            secondary_key=self.secondary_key, filters=self.filters,
        )


@dataclass(frozen=True)
class CumulativeNode:
    """The single cumulative-tile MV: running totals of the invertible aggregations' partials, from which
    every invertible window/offset/lifetime is derived at read time by 2-point asof subtraction."""

    name: str
    column_info: ColumnInfo
    invertible_aggregations: List[Aggregation]
    tiles_relation: str
    agg_params: Optional[Dict[str, List[float]]] = None
    filters: Optional[Dict[str, str]] = None

    def select(self) -> str:
        return build_cumulative_tile_select(
            self.column_info, self.invertible_aggregations, self.tiles_relation,
            agg_params=self.agg_params, filters=self.filters,
        )


@dataclass(frozen=True)
class WindowNode:
    """One now()-anchored rollup MV per (window_secs, offset_secs) for the non-invertible (interval-served)
    aggregations; read as a point lookup (one row per entity = the current window value)."""

    name: str
    column_info: ColumnInfo
    aggregations: List[Aggregation]
    tiles_relation: str
    aggregation_interval: object
    window_secs: int
    offset_secs: int
    agg_params: Optional[Dict[str, List[float]]] = None
    secondary_key: Optional[str] = None
    filters: Optional[Dict[str, str]] = None

    def select(self) -> str:
        return build_online_rollup_select(
            self.column_info, self.aggregations, self.tiles_relation,
            aggregation_interval=self.aggregation_interval, agg_params=self.agg_params,
            offset_secs=self.offset_secs, secondary_key=self.secondary_key, filters=self.filters,
        )


@dataclass(frozen=True)
class LifetimeNode:
    """One lifetime rollup MV per floor for the interval-served lifetime aggregations (all-history, optional
    start floor); a one-sided now() rollup read as a point lookup."""

    name: str
    column_info: ColumnInfo
    aggregations: List[Aggregation]
    tiles_relation: str
    floor: Optional[int]
    agg_params: Optional[Dict[str, List[float]]] = None
    secondary_key: Optional[str] = None
    filters: Optional[Dict[str, str]] = None

    def select(self) -> str:
        return build_lifetime_rollup_select(
            self.column_info, self.aggregations, self.tiles_relation, agg_params=self.agg_params,
            lifetime_start_secs=self.floor, secondary_key=self.secondary_key, filters=self.filters,
        )


@dataclass(frozen=True)
class SeriesSnapshotNode:
    """At most one per-entity last-L snapshot MV carrying every step==interval scalar window-series of the
    view, so its online read is a point lookup. ``select()`` is None when no series is snapshot-eligible
    (coarser-step / overlapping / array-valued series stay on the read-time single-scan and add no MV)."""

    name: str
    column_info: ColumnInfo
    aggregations: List[Aggregation]
    tiles_relation: str
    aggregation_interval: object
    agg_params: Optional[Dict[str, List[float]]] = None
    series: Optional[Dict[str, List[int]]] = None
    filters: Optional[Dict[str, str]] = None

    def select(self) -> Optional[str]:
        return build_series_snapshot_select(
            self.column_info, self.aggregations, self.tiles_relation,
            aggregation_interval=self.aggregation_interval, agg_params=self.agg_params,
            series=self.series, filters=self.filters,
        )


@dataclass(frozen=True)
class TilePlan:
    """The materialization plan: the tiles MV + the online rollup nodes. ``online_mvs()`` reproduces
    ``_desired_online_mvs`` byte-identically (same keys, same SELECTs, same order)."""

    project: str
    view_name: str
    tiles: TilesNode
    cumulative: Optional[CumulativeNode] = None
    windows: List[WindowNode] = field(default_factory=list)
    lifetimes: List[LifetimeNode] = field(default_factory=list)
    series_snapshot: Optional[SeriesSnapshotNode] = None

    @staticmethod
    def from_inputs(
        project: str,
        view_name: str,
        column_info: ColumnInfo,
        aggregations: List[Aggregation],
        tiles_relation: str,
        *,
        aggregation_interval,
        agg_params: Optional[Dict[str, List[float]]] = None,
        secondary_key: Optional[str] = None,
        offsets: Optional[Dict[str, int]] = None,
        lifetimes: Optional[Dict[str, Optional[int]]] = None,
        series: Optional[Dict[str, List[int]]] = None,
        filters: Optional[Dict[str, str]] = None,
        flavor: str = "batch",
        source_relation: str = "",
    ) -> "TilePlan":
        """Build the plan from the same inputs ``_desired_online_mvs`` takes (plus the tiles flavor + source
        for the tiles node). The node construction lifts the v2 serving split VERBATIM from
        ``_desired_online_mvs`` — the single home of that classification. ``filters`` (resolved_name ->
        canonical predicate) rides every node so a filtered aggregation's FILTER reaches both the tiles MV
        partials and every recombine identically (the offline read is built from the same filters)."""
        cumulative_ok = secondary_key is None
        invertible = [a for a in aggregations if cumulative_ok and is_invertible_agg(a)]
        cumulative = (
            CumulativeNode(online_cumulative_mv_name(project, view_name), column_info, invertible,
                           tiles_relation, agg_params, filters)
            if invertible else None
        )

        def _interval_served(a: Aggregation) -> bool:
            # served by a per-window / lifetime now()-MV: not invertible-via-cumulative, and not a series.
            return not (cumulative_ok and is_invertible_agg(a)) and not is_series_agg(a, series)

        windowed = [a for a in aggregations if _interval_served(a) and not is_lifetime_agg(a, lifetimes)]
        windows = [
            WindowNode(online_window_mv_name(project, view_name, w, off), column_info, wa, tiles_relation,
                       aggregation_interval, w, off, agg_params, secondary_key, filters)
            for (w, off), wa in group_aggregations_by_window_offset(windowed, offsets)
        ]
        lifetime = [a for a in aggregations if _interval_served(a) and is_lifetime_agg(a, lifetimes)]
        lifes = [
            LifetimeNode(online_lifetime_mv_name(project, view_name, floor), column_info, la, tiles_relation,
                         floor, agg_params, secondary_key, filters)
            for floor, la in group_lifetime_aggregations(lifetime, lifetimes)
        ]
        snapshot = (
            SeriesSnapshotNode(online_series_mv_name(project, view_name), column_info, aggregations,
                               tiles_relation, aggregation_interval, agg_params, series, filters)
            if cumulative_ok else None
        )
        tiles = TilesNode(tiles_relation, column_info, aggregations, source_relation, flavor,
                          aggregation_interval, agg_params, secondary_key, filters)
        return TilePlan(project, view_name, tiles, cumulative, windows, lifes, snapshot)

    def online_mvs(self) -> Dict[str, str]:
        """The desired ``{mv_name: SELECT}`` for the online rollup MVs — byte-identical to
        ``_desired_online_mvs``: cumulative (if any invertible), then per-(window, offset) ascending, then
        per-lifetime-floor, then the series snapshot last (omitted when no series is snapshot-eligible)."""
        out: Dict[str, str] = {}
        if self.cumulative is not None:
            out[self.cumulative.name] = self.cumulative.select()
        for w in self.windows:
            out[w.name] = w.select()
        for lf in self.lifetimes:
            out[lf.name] = lf.select()
        if self.series_snapshot is not None:
            snapshot_sql = self.series_snapshot.select()
            if snapshot_sql is not None:
                out[self.series_snapshot.name] = snapshot_sql
        return out

    def tiles_ddl(self) -> Tuple[str, str]:
        """The tiles MV's (name, SELECT) — the relation every online node reads."""
        return (self.tiles.name, self.tiles.select())


def tile_plan_from_view(project: str, view) -> TilePlan:
    """Convenience builder from an engine view object: derive column-info + flavor + carriers and build the
    plan. (Used by the provisioning / reconcile / offline / serving consumers; kept thin so the carrier
    decode lives in ONE place.)"""
    from feast.infra.compute_engines.risingwave.aggregation_carriers import (
        view_agg_filters,
        view_agg_lifetime,
        view_agg_offsets,
        view_agg_params,
        view_agg_series,
        view_secondary_key,
    )
    from feast.infra.compute_engines.risingwave.ddl import _registry_free_column_info
    from feast.infra.compute_engines.risingwave.iceberg_source import is_streaming_tile, tile_interval, view_aggregations

    streaming = is_streaming_tile(view)
    column_info = _registry_free_column_info(view) if streaming else _batch_column_info(view)
    return TilePlan.from_inputs(
        project, view.name, column_info, view_aggregations(view), tiles_name(project, view.name),
        aggregation_interval=tile_interval(view), agg_params=view_agg_params(view),
        secondary_key=view_secondary_key(view), offsets=view_agg_offsets(view),
        lifetimes=view_agg_lifetime(view), series=view_agg_series(view), filters=view_agg_filters(view),
        flavor="streaming" if streaming else "batch",
        source_relation=(view.stream_source.name if streaming else view.batch_source.table),
    )


def _batch_column_info(view) -> ColumnInfo:
    return ColumnInfo(
        join_keys=[f.name for f in view.entity_columns],
        feature_cols=[f.name for f in view.features],
        ts_col=view.batch_source.timestamp_field,
        created_ts_col=None,
        field_mapping=None,
    )
