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


def online_window_mv_name(project: str, view_name: str, window_secs: int) -> str:
    # Per-window online rollup MV for a tile BATCH feature view. A tile FV reuses ONE tile set across
    # many time-windows, but RisingWave rejects now() inside a CASE in a two-sided
    # temporal-filter MV, so each distinct window gets its OWN now()-anchored rollup MV. The point-lookup
    # reads the per-window MV holding the requested feature's window. The engine (provisioning) and apply
    # (serving spec) derive this name from the SAME group_aggregations_by_window split, so they cannot
    # drift.
    return f"{base_name(project, view_name)}_online_{window_secs}s"


def tiles_name(project: str, view_name: str) -> str:
    # Internal tile MV for a BATCH feature view: holds the per-(entity, tile_end) partial
    # aggregates. The point-lookup never reads this; it reads the online rollup MV
    # (online_mv_name), which rolls these tiles up to the current request-anchored window.
    return f"{base_name(project, view_name)}_tiles"


def offline_sink_name(project: str, view_name: str) -> str:
    return f"{base_name(project, view_name)}_offline"


def offline_staging_name(project: str, view_name: str) -> str:
    return f"{base_name(project, view_name)}_offline_staging"
