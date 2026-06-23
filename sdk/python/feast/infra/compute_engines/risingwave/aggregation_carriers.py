"""Engine-owned view-tag carriers + aggregation grouping for the RisingWave engine.

feast.Aggregation carries no field for several per-aggregation parameters this engine
needs (the quantile/N of a parameterized aggregate, a window offset, a lifetime floor, a
window-series geometry, or an FV-level secondary key). Each rides an engine-namespaced
view tag, encoded/decoded here. The grouping helpers fold a view's aggregations into the
(window[, offset]) / lifetime-floor buckets the provisioning + serving-spec layers must
group identically. This is the leaf module of the engine's SQL-builder stack: it depends
only on ``feast.Aggregation`` (and these helpers on each other), nothing else in the
package.
"""

import json
from typing import Dict, List, Optional, Sequence, Tuple

from feast.aggregation import Aggregation

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


# Which aggregations are LIFETIME (aggregate over ALL of an entity's history, no trailing bound),
# keyed by resolved_name -> the optional floor as epoch seconds (None = no floor, aggregate from the
# beginning). A lifetime aggregation has no window length, so it cannot ride feast.Aggregation's
# time_window; this engine-owned tag (a carrier parallel to the offset/param tags) is the marker, and
# PRESENCE of a resolved_name here is what makes the aggregation lifetime. Unlike the offset/param tags,
# a None value is meaningful (a lifetime with no floor), so entries are never dropped — an empty map
# means no lifetime aggregations. Works for both tile flavors (the tag round-trips on the view), and is
# robust to feast.Aggregation rendering the null window as None (batch) or timedelta(0) (streaming proto
# round-trip): both resolve to the same suffix-less name, so the carrier key is stable either way.
AGG_LIFETIME_TAG = "feast_rw_agg_lifetime"


def encode_agg_lifetime(
    lifetimes_by_resolved_name: Dict[str, Optional[int]],
) -> Dict[str, str]:
    """The view-tags fragment marking the lifetime aggregations, keyed by resolved_name -> optional floor
    epoch seconds (None = no floor). Returns ``{}`` when there are no lifetime aggregations, so a view
    with only windowed aggregations is left untouched. The inverse of ``view_agg_lifetime``."""
    cleaned = {
        name: (int(secs) if secs is not None else None)
        for name, secs in lifetimes_by_resolved_name.items()
    }
    return {AGG_LIFETIME_TAG: json.dumps(cleaned)} if cleaned else {}


def view_agg_lifetime(view) -> Dict[str, Optional[int]]:
    """The lifetime aggregations a view carries in its tags, keyed by resolved_name -> optional floor
    epoch seconds (None = no floor). A resolved_name present here is a lifetime aggregation. Absent => {}
    (the common case: no lifetime aggregations). The inverse of ``encode_agg_lifetime``."""
    raw = (getattr(view, "tags", None) or {}).get(AGG_LIFETIME_TAG)
    if not raw:
        return {}
    return {
        name: (int(secs) if secs is not None else None)
        for name, secs in json.loads(raw).items()
    }


def is_lifetime_agg(agg: Aggregation, lifetimes: Optional[Dict[str, Optional[int]]]) -> bool:
    """Whether this aggregation is a lifetime aggregation, per the lifetime carrier (membership keyed by
    resolved_name). resolved_name is suffix-less for a lifetime aggregation whether the carried window is
    None or timedelta(0), so the lookup is stable across the batch and streaming round-trips."""
    return agg.resolved_name(agg.time_window) in (lifetimes or {})


# The FEATURE-VIEW-LEVEL aggregation secondary key: a raw column naming a second GROUP BY dimension, so
# each aggregation produces a per-secondary-key breakdown (a key -> value Map) per entity per window
# (e.g. per user, a map of ad_id -> click count). It is one column shared by every aggregation in the
# view (an FV-level param, not per-aggregation), so it rides a single engine-owned view tag rather than
# the resolved_name-keyed carriers. Absent => no breakdown (the scalar output). HIGH-CARDINALITY CAVEAT:
# the breakdown adds a GROUP BY dimension to the tiles and a per-entity Map that grows with the entity's
# distinct-key count (and a lifetime+secondary-key MV never evicts), so a high-cardinality secondary key
# inflates tile + MV state — the same unbounded-state class as an exact count_distinct.
SECONDARY_KEY_TAG = "feast_rw_secondary_key"


def encode_secondary_key(secondary_key: Optional[str]) -> Dict[str, str]:
    """The view-tags fragment carrying the aggregation secondary key (a raw column name), or ``{}`` when
    the view has none (so a view without a breakdown is left untouched). The inverse of
    ``view_secondary_key``."""
    return {SECONDARY_KEY_TAG: secondary_key} if secondary_key else {}


def view_secondary_key(view) -> Optional[str]:
    """The aggregation secondary key (a raw column name) a view carries in its tags, or None when there
    is no breakdown. The inverse of ``encode_secondary_key``."""
    return (getattr(view, "tags", None) or {}).get(SECONDARY_KEY_TAG) or None


# The per-aggregation window-SERIES geometry: an aggregation fanned into a series of trailing windows,
# emitted as one ARRAY-valued feature (e.g. 24 hourly sums). feast.Aggregation carries a single window,
# not a series, so the geometry rides this engine-owned tag — a carrier parallel to the offset/param tags
# — as a JSON map keyed by resolved_name -> ``[window_secs, step_secs, length]`` (length = the count of
# windows, L). PRESENCE of a resolved_name here is what makes the aggregation a series (so it is excluded
# from the per-window/lifetime rollups and read by the single-scan series query instead — one range-scan
# of the tiles plus an ARRAY of per-step recombines). step_secs and window_secs are each whole multiples
# of the tile aggregation_interval; window_secs may exceed step_secs (overlapping windows). Works for both
# tile flavors (the tag round-trips on the view).
AGG_SERIES_TAG = "feast_rw_agg_series"


def encode_agg_series(
    series_by_resolved_name: Dict[str, Sequence[int]],
) -> Dict[str, str]:
    """The view-tags fragment carrying per-aggregation window-series geometry, keyed by resolved_name ->
    ``[window_secs, step_secs, length]`` (whole seconds + window count). Drops empty entries and returns
    ``{}`` when there is no series aggregation, so a series-free view's tags are left untouched. The
    inverse of ``view_agg_series``."""
    cleaned = {
        name: [int(v) for v in geometry]
        for name, geometry in series_by_resolved_name.items()
        if geometry
    }
    return {AGG_SERIES_TAG: json.dumps(cleaned)} if cleaned else {}


def view_agg_series(view) -> Dict[str, List[int]]:
    """The window-series geometry a view carries in its tags, keyed by resolved_name ->
    ``[window_secs, step_secs, length]``. A resolved_name present here is a series aggregation. Absent =>
    {} (the common case: no series aggregations). The inverse of ``encode_agg_series``."""
    raw = (getattr(view, "tags", None) or {}).get(AGG_SERIES_TAG)
    if not raw:
        return {}
    return {name: [int(v) for v in geometry] for name, geometry in json.loads(raw).items()}


def is_series_agg(agg: Aggregation, series: Optional[Dict[str, Sequence[int]]]) -> bool:
    """Whether this aggregation is a window-series, per the series carrier (membership keyed by
    resolved_name). A series lowers to a NULL window like a lifetime aggregation, so the two are told
    apart only by which carrier holds the resolved_name — never by the window value alone."""
    return agg.resolved_name(agg.time_window) in (series or {})


# The per-aggregation FILTER predicate: a filtered aggregation (e.g. a count of only DEBIT transactions, or
# only 11PM-3AM events) is the SAME aggregation as its unfiltered sibling but with a boolean predicate over
# STATIC source columns, applied as a SQL ``FILTER (WHERE ...)`` on the tile partial — so total / DEBIT / QR
# / EOD counts share ONE tile scan (one GROUP BY, filtered partial columns side by side). feast.Aggregation
# carries no predicate, so it rides this engine-owned tag as a JSON map keyed by resolved_name -> the
# canonical predicate SQL (validated + canonicalized by the authoring layer through DataFusion). The
# predicate is on static source columns (evaluated at tile build, free at read), so online == offline holds.
AGG_FILTER_TAG = "feast_rw_agg_filter"


def encode_agg_filters(filter_by_resolved_name: Dict[str, str]) -> Dict[str, str]:
    """The view-tags fragment carrying per-aggregation FILTER predicates, keyed by resolved_name -> the
    canonical predicate SQL. Drops empty entries and returns ``{}`` when there is no filtered aggregation,
    so an unfiltered view's tags are left untouched. The inverse of ``view_agg_filters``."""
    cleaned = {name: pred for name, pred in filter_by_resolved_name.items() if pred}
    return {AGG_FILTER_TAG: json.dumps(cleaned)} if cleaned else {}


def view_agg_filters(view) -> Dict[str, str]:
    """The FILTER predicate a view carries in its tags, keyed by resolved_name -> canonical predicate SQL.
    A resolved_name present here is a filtered aggregation. Absent => {} (the common case). The inverse of
    ``encode_agg_filters``."""
    raw = (getattr(view, "tags", None) or {}).get(AGG_FILTER_TAG)
    if not raw:
        return {}
    return dict(json.loads(raw))


def is_filtered_agg(agg: Aggregation, filters: Optional[Dict[str, str]]) -> bool:
    """Whether this aggregation carries a FILTER predicate, per the filter carrier (membership keyed by
    resolved_name)."""
    return agg.resolved_name(agg.time_window) in (filters or {})


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


def group_lifetime_aggregations(
    aggregations: List[Aggregation],
    lifetimes: Optional[Dict[str, Optional[int]]],
) -> List[Tuple[Optional[int], List[Aggregation]]]:
    """Group the LIFETIME aggregations by their floor (epoch seconds, or None for no floor), ascending
    (no-floor first). Each distinct floor becomes its own now()-anchored lifetime rollup MV — the WHERE
    differs (``tile_end <= now()`` with an optional ``tile_end > floor``), so floors can't share an MV.
    Non-lifetime aggregations are skipped. The engine (provisioning) and apply (serving spec) MUST group
    identically from this one helper so the per-floor lifetime MV names cannot drift."""
    groups: dict = {}
    for a in aggregations:
        if is_lifetime_agg(a, lifetimes):
            floor = (lifetimes or {})[a.resolved_name(a.time_window)]
            groups.setdefault(floor, []).append(a)
    return [
        (floor, groups[floor])
        for floor in sorted(groups, key=lambda f: (f is not None, f or 0))
    ]
