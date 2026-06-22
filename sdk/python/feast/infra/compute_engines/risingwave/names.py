"""Canonical RisingWave object names — the ONE source of truth for the provisioning naming
contract.

The same physical objects are referenced by the engine (DDL create/drop), the online store
(point-lookup read), the DAG nodes (source/staging), and the platform's apply step (which
derives the ServingSpec's MV name). If any of those re-template the name independently they
can silently drift and the online read returns 0 rows. All callers MUST use these helpers.
"""

from __future__ import annotations


def base_name(project: str, view_name: str) -> str:
    return f"{project}_{view_name}"


def source_name(project: str, view_name: str) -> str:
    return f"{base_name(project, view_name)}_src"


def online_mv_name(project: str, view_name: str) -> str:
    # Single online rollup MV for a STREAM feature view (one EOWC windowed-agg MV per view).
    return f"{base_name(project, view_name)}_online"


def online_window_mv_name(
    project: str, view_name: str, window_secs: int, offset_secs: int = 0
) -> str:
    # Per-(window, offset) online rollup MV for a tile feature view. A tile FV reuses ONE tile set across
    # many time-windows, but RisingWave rejects now() inside a CASE in a two-sided temporal-filter MV, so
    # each distinct window gets its OWN now()-anchored rollup MV. A window SHIFTED into the past (a
    # non-zero offset) cannot share the trailing window's MV either (its now()-anchored WHERE differs),
    # so the offset is part of the name. The engine (provisioning) and apply (serving spec) derive this
    # name from the SAME (window, offset) split, so they cannot drift. offset=0 (the trailing window, the
    # common case) keeps the bare ``_online_{secs}s`` name — back-compatible with existing deployments.
    name = f"{base_name(project, view_name)}_online_{window_secs}s"
    off = abs(int(offset_secs))
    if off:
        name += f"_off{off}s"
    return name


def online_lifetime_mv_name(
    project: str, view_name: str, lifetime_start_secs=None
) -> str:
    # Per-floor LIFETIME online rollup MV for a tile feature view: the all-history rollup (one-sided
    # now() upper bound), distinct from the per-window MVs. A floored lifetime (aggregate since a fixed
    # start) gets its own MV per floor — its now()-anchored WHERE differs — named with the floor epoch.
    name = f"{base_name(project, view_name)}_online_lifetime"
    if lifetime_start_secs is not None:
        name += f"_from{int(lifetime_start_secs)}s"
    return name


def online_cumulative_mv_name(project: str, view_name: str) -> str:
    # The single CUMULATIVE-tile online MV for a tile feature view: per-(entity, tile_end) running totals
    # of the invertible partials, from which the serving layer derives EVERY invertible window (trailing/
    # offset/lifetime/series) by read-time 2-point asof subtraction. Replaces the N per-(window, offset) +
    # M lifetime now()-anchored MVs for the invertible aggregations (sum/count/mean/var/stddev). The
    # ``_online_`` infix keeps it inside the ``_existing_online_mv_names`` reconcile sweep.
    return f"{base_name(project, view_name)}_online_cum"


# The reserved column the series snapshot MV stores the per-entity last-L tile end timestamps under,
# alongside one per-series value array. The serving reader reads this column to position each value into
# its frontier-relative slot; the leading ``__`` keeps it from colliding with a feature's resolved_name.
SERIES_SNAPSHOT_ENDS_COL = "__series_tile_ends"


def online_series_mv_name(project: str, view_name: str) -> str:
    # The per-entity window-series SNAPSHOT MV for a tile feature view: one row per entity holding the last
    # L tile end timestamps (SERIES_SNAPSHOT_ENDS_COL) plus each step==interval series' per-tile finalized
    # values, read as a single-row point lookup (the reader positions the values against the request
    # frontier). ONE per view — it carries every snapshot-eligible series. The ``_online_`` infix keeps it
    # inside the ``_existing_online_mv_names`` reconcile sweep.
    return f"{base_name(project, view_name)}_online_series"


def tiles_name(project: str, view_name: str) -> str:
    # Internal tile MV for a BATCH feature view: holds the per-(entity, tile_end) partial
    # aggregates. The point-lookup never reads this; it reads the online rollup MV
    # (online_mv_name), which rolls these tiles up to the current request-anchored window.
    return f"{base_name(project, view_name)}_tiles"


def offline_sink_name(project: str, view_name: str) -> str:
    return f"{base_name(project, view_name)}_offline"


def offline_staging_name(project: str, view_name: str) -> str:
    return f"{base_name(project, view_name)}_offline_staging"


def passthrough_history_source_name(project: str, view_name: str) -> str:
    # The Iceberg source over a STREAMING passthrough's batch_source (the historical log backing the
    # stream): online serves from the latest-row MV over the Kafka source, but offline point-in-time
    # training reads the raw history here. A BATCH passthrough has no separate history source — its online
    # source (source_name, an Iceberg source over the batch table) IS the history.
    return f"{base_name(project, view_name)}_history"
