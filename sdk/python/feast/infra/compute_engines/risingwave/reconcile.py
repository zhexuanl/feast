"""RisingWave catalog readers + the pure tile-view reconcile planner.

The read-back side of the engine's drift detection: given a live cursor, read what RisingWave
actually deployed (MV SELECTs, source ``table.name`` / Kafka opts / column types) and compare it
against the desired definitions so a re-applied view re-materializes only what changed. Depends on
``engine_config`` for the canonical-type map; independent of ``ddl.py``. ``engine.py`` re-exports
every symbol here so existing imports keep resolving.
"""

import re
from typing import Optional

from feast.data_source import KafkaSource
from feast.infra.compute_engines.risingwave.engine_config import _canonical_type
from feast.infra.compute_engines.risingwave.names import base_name


def _existing_online_mv_names(cur, project: str, view_name: str) -> set:
    """The NAMES of the per-(window, offset) online rollup MVs that physically EXIST for a tile view, read
    from RisingWave's pg-compatible catalog. The reconcile diffs these names against the DESIRED set so a
    re-apply that shrinks/changes a view's (window, offset) set drops the now-removed MVs: Feast routes a
    same-name edited view to ``views_to_keep`` (not ``views_to_delete``) and ``CREATE ... IF NOT EXISTS``
    never removes a no-longer-provisioned MV, so it would otherwise run forever, unreachable by any future
    provision/teardown (which only name the current set).

    Matches the trailing-window form ``{base}_online_{secs}s``, the shifted form
    ``{base}_online_{secs}s_off{secs}s`` (``online_window_mv_name``), and the lifetime form
    ``{base}_online_lifetime`` / ``..._from{secs}s`` (``online_lifetime_mv_name``); the anchored regex
    avoids matching a differently-named view that merely shares this view's name as a prefix."""
    base = re.escape(base_name(project, view_name))
    pattern = re.compile(
        rf"^{base}_online_(?:\d+s(?:_off\d+s)?|lifetime(?:_from\d+s)?|cum)$"
    )
    cur.execute("SELECT matviewname FROM pg_matviews")
    return {name for (name,) in cur.fetchall() if pattern.match(name)}


def _deployed_mv_select(cur, name: str) -> Optional[str]:
    """The SELECT of a deployed materialized view as RisingWave stores it (verbatim: RisingWave
    persists ``CREATE MATERIALIZED VIEW <name> AS <select>`` with our SELECT unchanged),
    or None if the MV does not exist. This stored SELECT is an exact definition fingerprint used to
    detect that a kept view's definition changed — the only way to do so, since RW has no CREATE OR
    REPLACE / ALTER ... AS / COMMENT ON, and Feast never tells the engine which kept views changed.

    Assumption: RW round-trips our generated SELECT unchanged (modulo whitespace, which the reconcile
    normalizes). If a future RW version re-rendered the stored definition differently from what we
    generate, the comparison would conservatively see every apply as "changed" and re-materialize each
    time — wasteful, never wrong. If that ever happens, switch to a stored definition hash (a sidecar)
    rather than comparing against RW's rendering."""
    cur.execute(
        "SELECT definition FROM rw_catalog.rw_materialized_views WHERE name = %s", (name,)
    )
    row = cur.fetchone()
    if not row:
        return None
    definition = row[0]
    marker = " AS "  # the MV name precedes the first ' AS '; everything after it is the SELECT
    idx = definition.find(marker)
    return definition[idx + len(marker) :] if idx != -1 else definition


def _deployed_source_table(cur, name: str) -> Optional[str]:
    """The Iceberg ``table.name`` of a deployed source as RisingWave stores it, or None if the source
    does not exist. A tile view's tiles MV reads its source by the (stable) source NAME, so the
    underlying Iceberg table only appears in the ``CREATE SOURCE ... table.name='...'`` definition —
    this is the only way to detect that a kept view was repointed at a different table (which the MV
    definitions cannot reveal). Single quotes in the table are doubled in the DDL; we un-double them.

    Note: unlike a materialized view (whose SELECT RisingWave stores VERBATIM), a source's WITH clause is
    RE-RENDERED in the catalog (spaces around ``=``, expanded types), so we EXTRACT the
    ``table.name`` option with a spacing-tolerant regex rather than comparing the whole definition."""
    cur.execute("SELECT definition FROM rw_catalog.rw_sources WHERE name = %s", (name,))
    row = cur.fetchone()
    if not row:
        return None
    m = re.search(r"(?:^|[\s,(])table\.name\s*=\s*'((?:[^']|'')*)'", row[0])
    return m.group(1).replace("''", "'") if m else None


def _deployed_kafka_source_opts(cur, name: str) -> Optional[tuple]:
    """The Kafka connector options of a deployed source as RisingWave stores them — the tuple
    ``(topic, bootstrap_servers, watermark_secs)`` — or None if the source does not exist. A stream (or
    streaming-tile) view reads its source by the (stable) source NAME, so a repointed topic, a moved
    bootstrap server, or a changed watermark delay live ONLY in the ``CREATE SOURCE`` definition, never in
    any materialized-view SELECT — reading them back from the catalog is the only way to detect that a kept
    view's source changed (a repointed topic would keep feeding stale data; a changed watermark would
    silently shift late-event admission and break train/serve parity).

    Like ``_deployed_source_table``, a source's WITH clause is RE-RENDERED in the catalog (spaces around
    ``=``, expanded types) rather than stored verbatim, so each option is EXTRACTED with a spacing-tolerant
    regex; single quotes doubled in the DDL are un-doubled. The watermark delay is parsed from the column
    list's ``WATERMARK FOR <ts> AS <ts> - INTERVAL '<n>' SECOND`` (the only INTERVAL a Kafka source DDL
    carries); a source with no watermark yields None for that slot."""
    cur.execute("SELECT definition FROM rw_catalog.rw_sources WHERE name = %s", (name,))
    row = cur.fetchone()
    if not row:
        return None
    definition = row[0]
    topic = re.search(r"(?:^|[\s,(])topic\s*=\s*'((?:[^']|'')*)'", definition)
    bootstrap = re.search(
        r"(?:^|[\s,(])properties\.bootstrap\.server\s*=\s*'((?:[^']|'')*)'", definition
    )
    watermark = re.search(r"INTERVAL\s+'(\d+)'\s+SECOND", definition)
    return (
        topic.group(1).replace("''", "'") if topic else None,
        bootstrap.group(1).replace("''", "'") if bootstrap else None,
        int(watermark.group(1)) if watermark else None,
    )


def _desired_kafka_source_opts(source: KafkaSource) -> tuple:
    """The desired ``(topic, bootstrap_servers, watermark_secs)`` from a view's KafkaSource, in the same
    shape ``_deployed_kafka_source_opts`` reads back so the two compare directly. ``watermark_secs`` is the
    integer-second watermark delay (matching ``_source_ddl``'s ``INTERVAL '<n>' SECOND``), or None when the
    source sets no watermark."""
    opts = source.kafka_options
    wm = opts.watermark_delay_threshold
    return (
        opts.topic,
        opts.kafka_bootstrap_servers,
        int(wm.total_seconds()) if wm is not None else None,
    )


def _deployed_source_columns(cur, name: str) -> Optional[dict]:
    """The ``{column: canonical SQL type}`` map of a deployed source as RisingWave reports it in
    information_schema, or None if the source does not exist. A passthrough Kafka source declares its raw
    feature columns with explicit types; a feature dtype change (same column name) shows in NO MV SELECT (the
    latest-row MV projects columns by name only), so it is read back here to detect that the source schema
    changed. information_schema reports canonical type names, so it is compared against ``_canonical_type``
    rather than the re-rendered CREATE SOURCE form."""
    cur.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s", (name,)
    )
    rows = cur.fetchall()
    return {name: dtype for name, dtype in rows} if rows else None


def _desired_passthrough_columns(view) -> dict:
    """The desired ``{column: canonical SQL type}`` a passthrough Kafka source declares — entity keys + raw
    feature columns + the event timestamp — in the same canonical form ``_deployed_source_columns`` reads
    back, so a feature dtype change is detected on reconcile."""
    source = view.stream_source
    cols: dict = {}
    for field in view.entity_columns:
        cols[field.name] = _canonical_type(getattr(field, "dtype", ""))
    for feature in view.features:
        cols.setdefault(feature.name, _canonical_type(getattr(feature, "dtype", "")))
    cols.setdefault(source.timestamp_field, "timestamp without time zone")
    return cols


def _norm_sql(sql):
    """Whitespace-normalize a SELECT for definition comparison (RW stores our SELECT verbatim modulo
    whitespace). None stays None so a missing deployed object never compares equal to a desired one."""
    return None if sql is None else " ".join(sql.split())


def _plan_batch_reconcile(
    *, desired_tiles: str, desired_online: dict, deployed_tiles, deployed_online: dict
):
    """Pure reconcile planner: compare a tile view's DESIRED definitions against the DEPLOYED ones (read
    from RisingWave's catalog) and return ``(full_rebuild, online_drops, online_creates)``.

    ``full_rebuild`` is True when the tiles MV changed (the per-tile PARTIALS changed — a different
    aggregation function/column) or the view is unprovisioned: the caller drops every deployed online MV
    (returned in ``online_drops``) and the tiles MV, then re-provisions the whole graph. Otherwise the
    tiles MV is unchanged (its partials are WINDOW-INDEPENDENT, so adding/removing a window does NOT touch
    it) and only the per-window online MVs are reconciled — drop windows that were removed or whose rollup
    definition changed, create windows that are new or redefined, and leave unchanged windows (and their
    serving) untouched. A materialization-affecting change re-materializes; an unchanged view is a
    no-op (no rebuild, no serving blip)."""

    norm = _norm_sql
    if norm(deployed_tiles) != norm(desired_tiles):
        return True, list(deployed_online), []
    drops = [
        name
        for name, dep in deployed_online.items()
        if name not in desired_online or norm(dep) != norm(desired_online[name])
    ]
    creates = [
        (name, sql)
        for name, sql in desired_online.items()
        if name not in deployed_online or norm(deployed_online[name]) != norm(sql)
    ]
    return False, drops, creates
