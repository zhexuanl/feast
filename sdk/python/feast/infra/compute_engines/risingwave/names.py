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
    return f"{base_name(project, view_name)}_online"


def tiles_name(project: str, view_name: str) -> str:
    # Internal tile MV for a BATCH feature view: holds the per-(entity, tile_end) partial
    # aggregates. The point-lookup never reads this; it reads the online rollup MV
    # (online_mv_name), which rolls these tiles up to the current request-anchored window.
    return f"{base_name(project, view_name)}_tiles"


def offline_sink_name(project: str, view_name: str) -> str:
    return f"{base_name(project, view_name)}_offline"


def offline_staging_name(project: str, view_name: str) -> str:
    return f"{base_name(project, view_name)}_offline_staging"
