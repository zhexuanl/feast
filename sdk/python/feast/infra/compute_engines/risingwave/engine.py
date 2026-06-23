"""RisingWave compute + online-serving engine for Feast (contrib).

Architecture: RisingWave owns real-time computation and online serving via
continuous materialized views; Feast keeps the registry and the point-in-time
training joins. One ``update()`` provisions BOTH the online MV and the offline
Iceberg sink from one feature definition, so online and offline are computed by the
same engine — minimal train/serve skew.

Status: SCAFFOLD. The contract wiring, config, registry-free provisioning, and PIT
delegation are grounded and verified against the Feast/RisingWave source. The SQL
generation and the windowed-agg -> Iceberg composition are NOT yet verified
end-to-end; treat them as unproven and validate against live RisingWave before
relying on them in production (see the inline ``UNVERIFIED`` markers).
"""

import logging
import time
from typing import List, Sequence, Union

from feast import (
    BatchFeatureView,
    Entity,
    FeatureView,
    OnDemandFeatureView,
    StreamFeatureView,
)
from feast.data_source import KafkaSource, PushSource
from feast.infra.common.materialization_job import (
    MaterializationJob,
    MaterializationJobStatus,
    MaterializationTask,
)
from feast.infra.compute_engines.base import ComputeEngine
from feast.infra.compute_engines.risingwave.feature_builder import (
    RisingWaveFeatureBuilder,
)
from feast.infra.compute_engines.risingwave.job import (
    RisingWaveMaterializationJob,
)
from feast.infra.compute_engines.risingwave.names import (
    offline_sink_name,
    online_mv_name,
    passthrough_history_source_name,
    source_name,
    tiles_name,
)
from feast.infra.compute_engines.risingwave.iceberg_source import (
    IcebergSource,
    is_passthrough_stream,
    is_passthrough_view,
    is_streaming_tile,
    is_tile_fv,
    is_tile_view,
    tile_interval,
    view_aggregations,
)
from feast.infra.compute_engines.risingwave.nodes import (
    build_batch_tile_select,
    build_latest_row_select,
    build_streaming_tile_select,
    build_windowed_agg_select,
    view_agg_filters,
    view_agg_lifetime,
    view_agg_offsets,
    view_agg_params,
    view_agg_series,
    view_secondary_key,
)
from feast.infra.offline_stores.offline_store import OfflineStore
from feast.infra.online_stores.online_store import OnlineStore
from feast.infra.registry.base_registry import BaseRegistry
from feast.repo_config import RepoConfig

# The module-level config, DDL-string builders, and catalog/reconcile helpers were split into three
# sibling modules; they are re-imported here (and listed in ``__all__``) so every existing
# ``from ...risingwave.engine import`` — the test suite, the spikes — keeps resolving without churn.
from feast.infra.compute_engines.risingwave.engine_config import (
    _ENGINE_PATH,
    _RW_CANONICAL_TYPE,
    _RW_TYPE,
    RisingWaveComputeEngineConfig,
    _canonical_type,
    _connect,
)
from feast.infra.compute_engines.risingwave.ddl import (
    _batch_column_info,
    _batch_drop_ddl,
    _desired_online_mvs,
    _drop_ddl,
    _iceberg_sink_ddl,
    _iceberg_source_ddl,
    _iceberg_storage_opts,
    _kafka_source_with,
    _materialized_view_ddl,
    _passthrough_column_info,
    _passthrough_drop_ddl,
    _passthrough_source_ddl,
    _registry_free_column_info,
    _source_ddl,
    _source_is_retractable,
    _sql_str,
)
from feast.infra.compute_engines.risingwave.reconcile import (
    _deployed_kafka_source_opts,
    _deployed_mv_select,
    _deployed_source_columns,
    _deployed_source_table,
    _desired_kafka_source_opts,
    _desired_passthrough_columns,
    _existing_online_mv_names,
    _norm_sql,
    _plan_batch_reconcile,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RisingWaveComputeEngine",
    # engine_config
    "_ENGINE_PATH",
    "_RW_TYPE",
    "_RW_CANONICAL_TYPE",
    "_canonical_type",
    "RisingWaveComputeEngineConfig",
    "_connect",
    # ddl
    "_registry_free_column_info",
    "_batch_column_info",
    "_passthrough_column_info",
    "_sql_str",
    "_iceberg_storage_opts",
    "_iceberg_source_ddl",
    "_source_is_retractable",
    "_kafka_source_with",
    "_source_ddl",
    "_passthrough_source_ddl",
    "_materialized_view_ddl",
    "_iceberg_sink_ddl",
    "_drop_ddl",
    "_passthrough_drop_ddl",
    "_batch_drop_ddl",
    "_desired_online_mvs",
    # reconcile
    "_existing_online_mv_names",
    "_deployed_mv_select",
    "_deployed_source_table",
    "_deployed_kafka_source_opts",
    "_desired_kafka_source_opts",
    "_deployed_source_columns",
    "_desired_passthrough_columns",
    "_norm_sql",
    "_plan_batch_reconcile",
]


# RisingWave occasionally fails a CREATE/DROP of a streaming object with a TRANSIENT cluster-state error:
# the meta service is mid-reschedule (a parallel-unit / vnode mapping is not ready, or no compute worker is
# momentarily available), which clears on a brief retry. A permanent error — a SQL or definition mistake —
# does NOT clear, so anything not matching these signatures is re-raised immediately rather than masked.
_TRANSIENT_DDL_ERROR_SIGNATURES = (
    "scheduler error",
    "no available worker",
    "no available parallel unit",
    "vnode mapping",
    "not found in the cluster",
    "service unavailable",
    "connection refused",
    "connection reset",
)


def _is_transient_ddl_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _TRANSIENT_DDL_ERROR_SIGNATURES)


def _execute_ddl(cur, sql: str, *, attempts: int = 4, backoff_ms: int = 250) -> None:
    """Run one CREATE/DROP DDL statement, retrying ONLY a transient RisingWave cluster-state error with a
    short linear backoff. Streaming-object DDL can transiently fail while the meta service is rescheduling;
    a permanent error is re-raised on the first attempt. Safe to retry because the engine connects with
    autocommit (each statement is its own transaction, so a failed statement leaves no aborted-txn state)."""
    for attempt in range(attempts):
        try:
            cur.execute(sql)
            return
        except Exception as exc:  # noqa: BLE001 — re-raised below unless it is a known transient
            if attempt == attempts - 1 or not _is_transient_ddl_error(exc):
                raise
            logger.warning(
                "transient RisingWave DDL error (attempt %d/%d), retrying: %s",
                attempt + 1,
                attempts,
                exc,
            )
            time.sleep(backoff_ms * (attempt + 1) / 1000)


class RisingWaveComputeEngine(ComputeEngine):
    def __init__(
        self,
        *,
        repo_config: RepoConfig,
        offline_store: OfflineStore,
        online_store: OnlineStore,
        **kwargs,
    ):
        # Training/PIT retrieval is handled by RisingWaveOfflineStore — Feast's
        # provider routes get_historical_features to the offline store, never the
        # compute engine — so the engine needs no offline store of a particular type.
        super().__init__(
            repo_config=repo_config,
            offline_store=offline_store,
            online_store=online_store,
            **kwargs,
        )
        self.config = repo_config.batch_engine

    def update(
        self,
        project: str,
        views_to_delete: Sequence[
            Union[BatchFeatureView, StreamFeatureView, FeatureView]
        ],
        views_to_keep: Sequence[
            Union[BatchFeatureView, StreamFeatureView, FeatureView, OnDemandFeatureView]
        ],
        entities_to_delete: Sequence[Entity],
        entities_to_keep: Sequence[Entity],
    ):
        # DDL-in-update precedent: Snowflake (snowflake_engine.py:101-157). The
        # persistent infra (source + MV + Iceberg sink) is provisioned here; the MV
        # then keeps online features continuously fresh — there is no per-task
        # streaming start in materialize().
        with _connect(self.config) as conn, conn.cursor() as cur:
            cur.execute("set sink_decouple = false")  # required before creating Iceberg sinks
            for view in views_to_delete:
                # is_streaming_tile FIRST: a streaming-tile view IS a StreamFeatureView, but its physical
                # objects are the tile graph (N online MVs + tiles MV + source, no Iceberg sink), so it
                # tears down via _batch_drop_ddl, not _drop_ddl.
                if is_streaming_tile(view):
                    for stmt in _batch_drop_ddl(project, view):
                        _execute_ddl(cur, stmt)
                elif is_passthrough_view(view):
                    for stmt in _passthrough_drop_ddl(project, view):
                        _execute_ddl(cur, stmt)
                elif isinstance(view, StreamFeatureView):
                    for stmt in _drop_ddl(project, view):
                        _execute_ddl(cur, stmt)
                elif is_tile_fv(view):
                    for stmt in _batch_drop_ddl(project, view):
                        _execute_ddl(cur, stmt)
            for view in views_to_keep:
                # Precedence is load-bearing: a streaming-tile AND a streaming-passthrough view are both
                # StreamFeatureViews, so is_streaming_tile and is_passthrough_view must be checked BEFORE the
                # generic StreamFeatureView branch — else a passthrough (no-aggregation) view would hit the
                # windowed-agg path, which rejects an empty aggregation set. is_passthrough_view also precedes
                # is_tile_fv, but the two are mutually exclusive (a tile view has aggregations).
                if is_streaming_tile(view):
                    self._reconcile_streaming_tile_view(cur, project, view)
                elif is_passthrough_view(view):
                    self._reconcile_passthrough_view(cur, project, view)
                elif isinstance(view, StreamFeatureView):
                    self._reconcile_stream_view(cur, project, view)
                elif is_tile_fv(view):
                    self._reconcile_batch_view(cur, project, view)

    def teardown_infra(
        self,
        project: str,
        fvs: Sequence[Union[BatchFeatureView, StreamFeatureView, FeatureView]],
        entities: Sequence[Entity],
    ):
        with _connect(self.config) as conn, conn.cursor() as cur:
            for view in fvs:
                # is_streaming_tile FIRST (it is also a StreamFeatureView): tear down the tile graph.
                if is_streaming_tile(view):
                    for stmt in _batch_drop_ddl(project, view):
                        _execute_ddl(cur, stmt)
                elif is_passthrough_view(view):
                    for stmt in _passthrough_drop_ddl(project, view):
                        _execute_ddl(cur, stmt)
                elif isinstance(view, StreamFeatureView):
                    for stmt in _drop_ddl(project, view):
                        _execute_ddl(cur, stmt)
                elif is_tile_fv(view):
                    for stmt in _batch_drop_ddl(project, view):
                        _execute_ddl(cur, stmt)

    def _provision_ddl(self, project: str, view) -> List[str]:
        """Registry-free DDL for one StreamFeatureView (source + MV + Iceberg sink)."""
        source = view.stream_source
        if isinstance(source, PushSource):
            raise ValueError(
                f"StreamFeatureView '{view.name}' uses a PushSource, which is too thin "
                "to compile to a RisingWave CREATE SOURCE (data_source.py:851-882). "
                "Use a KafkaSource, or route pushes through an external ingestion path."
            )
        if not isinstance(source, KafkaSource):
            raise ValueError(
                "RisingWaveComputeEngine requires a KafkaSource-backed "
                f"StreamFeatureView; '{view.name}' has {type(source).__name__}."
            )

        emit_on_close = bool(self.config.emit_on_window_close)
        has_watermark = source.kafka_options.watermark_delay_threshold is not None
        if emit_on_close and not has_watermark:
            raise ValueError(
                "emit_on_window_close=True requires a watermark on the source "
                f"timestamp, but the KafkaSource for '{view.name}' sets no "
                "watermark_delay_threshold. Set one, or "
                "disable emit_on_window_close (losing online/offline consistency)."
            )

        column_info = _registry_free_column_info(view)
        src = source_name(project, view.name)
        mv = online_mv_name(project, view.name)
        return [
            _source_ddl(src, source, view),
            _materialized_view_ddl(mv, self._stream_mv_select(project, view)),
            _iceberg_sink_ddl(offline_sink_name(project, view.name), mv, column_info, self.config),
        ]

    def _stream_mv_select(self, project: str, view) -> str:
        """The EOWC windowed-aggregation SELECT for a stream view's online MV — the ONE definition shared
        by provisioning and reconcile so the two cannot drift."""
        return build_windowed_agg_select(
            _registry_free_column_info(view),
            list(view.aggregations),
            source_name(project, view.name),
            source_is_retractable=_source_is_retractable(view.stream_source),
            emit_on_close=bool(self.config.emit_on_window_close),
            agg_params=view_agg_params(view),
        )

    def _reconcile_stream_view(self, cur, project: str, view) -> None:
        """Reconcile a KEPT StreamFeatureView to its current definition (re-materialize on a change;
        no-op when unchanged) — the stream analogue of ``_reconcile_batch_view``. ``CREATE ... IF NOT
        EXISTS`` would keep the old EOWC MV, so a changed aggregation would silently serve/train under
        the OLD definition. Compare the MV's deployed SELECT (RW catalog, stored verbatim) against the
        desired one; on a change drop the graph (sink -> MV -> source, dependents first) and re-provision.

        A repointed topic/bootstrap or a changed watermark delay lives only in the ``CREATE SOURCE``
        definition (the EOWC MV reads the source by its stable name), not in any MV SELECT, so it is read
        back from the source catalog (``_deployed_kafka_source_opts``) and triggers the same drop+reprovision
        — else a repointed topic keeps feeding stale data, or a changed watermark silently shifts late-event
        admission and breaks train/serve parity. The drop+reprovision already drops and recreates the source.

        Both online serving AND offline training read this EOWC MV (the offline source is a PostgreSQL
        query over it), so re-materializing the MV keeps online == offline under the new
        definition — no skew.

        Iceberg-sink follow-up (deferred): the ``{base}_offline`` Iceberg sink is a durable
        copy training does NOT read today — so its table retaining pre-change rows after a re-materialize
        is harmless now. RisingWave has no sink-level table reset (only ``create_table_if_not_exists``;
        no drop/overwrite/truncate), so if the offline read ever migrates to the Iceberg sink (i.e. once
        MV retention is introduced), the re-materialize must purge that table out-of-band (catalog
        drop, or a definition-versioned table name) to keep the durable archive consistent."""
        mv = online_mv_name(project, view.name)
        src = source_name(project, view.name)
        desired = self._stream_mv_select(project, view)
        mv_changed = _norm_sql(_deployed_mv_select(cur, mv)) != _norm_sql(desired)
        # A repointed topic/bootstrap or a changed watermark shows in NO MV SELECT (the EOWC MV reads the
        # source by its stable name), so read the source opts back from the catalog and reprovision on a
        # difference too. ``deployed is None`` (unprovisioned) is handled by mv_changed (the MV is absent).
        deployed_opts = _deployed_kafka_source_opts(cur, src)
        source_changed = (
            deployed_opts is not None
            and deployed_opts != _desired_kafka_source_opts(view.stream_source)
        )
        if mv_changed or source_changed:
            for stmt in _drop_ddl(project, view):
                _execute_ddl(cur, stmt)
            for stmt in self._provision_ddl(project, view):
                _execute_ddl(cur, stmt)

    def _passthrough_mv_select(self, project: str, view) -> str:
        """The latest-row SELECT for a passthrough view's online MV — the ONE definition shared by
        provisioning and reconcile so the two cannot drift."""
        return build_latest_row_select(
            _passthrough_column_info(view), source_name(project, view.name)
        )

    def _provision_passthrough_ddl(self, project: str, view) -> List[str]:
        """Registry-free DDL for one passthrough (non-aggregated) feature view: a source + ONE latest-row
        online MV (the newest row per entity, served by the same point-lookup as an aggregation MV). A
        streaming passthrough reads a Kafka source that declares the raw feature columns; a batch passthrough
        reads an Iceberg source that infers them. No window/tile and no watermark: the latest-row MV is a
        Group-TopN over an append-only source. No Iceberg sink.

        Offline training reads the RAW history with an as-of cut, not this MV. A batch passthrough's own
        Iceberg source IS that history. A streaming passthrough's Kafka stream is not queryable as history,
        so when its stream source declares an Iceberg batch_source (the historical log backing the stream)
        we ALSO provision an Iceberg source over it for the offline read. A PostgreSQL batch_source needs
        no provisioning here — RisingWave reads it directly over pgwire at training time. Absent a readable
        batch_source, the view is online-only and offline retrieval is refused."""
        src = source_name(project, view.name)
        mv = online_mv_name(project, view.name)
        if is_passthrough_stream(view):
            source = view.stream_source
            if isinstance(source, PushSource):
                raise ValueError(
                    f"passthrough StreamFeatureView '{view.name}' uses a PushSource, which is too thin to "
                    "compile to a RisingWave CREATE SOURCE. Use a KafkaSource."
                )
            if not isinstance(source, KafkaSource):
                raise ValueError(
                    "RisingWaveComputeEngine requires a KafkaSource-backed passthrough stream view; "
                    f"'{view.name}' has {type(source).__name__}."
                )
            ddl = [
                _passthrough_source_ddl(src, source, view),
                _materialized_view_ddl(mv, self._passthrough_mv_select(project, view)),
            ]
            history = getattr(source, "batch_source", None)
            if isinstance(history, IcebergSource):
                ddl.append(
                    _iceberg_source_ddl(
                        passthrough_history_source_name(project, view.name), history.table, self.config
                    )
                )
            # A PostgreSQL batch_source is intentionally NOT provisioned: the offline read queries it
            # directly over pgwire, so there is no history source to create here.
            return ddl
        source_ddl = _iceberg_source_ddl(src, view.batch_source.table, self.config)
        return [source_ddl, _materialized_view_ddl(mv, self._passthrough_mv_select(project, view))]

    def _reconcile_passthrough_view(self, cur, project: str, view) -> None:
        """Reconcile a KEPT passthrough view to its current definition (re-materialize on a change; no-op
        when unchanged) — the passthrough analogue of ``_reconcile_stream_view``. ``CREATE ... IF NOT
        EXISTS`` would keep the old latest-row MV, so a changed feature set would silently serve the OLD
        definition; compare the MV's deployed SELECT (RW catalog, stored verbatim) against the desired one
        and, on a change, drop the MV + source and re-provision.

        A repointed source (a Kafka topic/bootstrap/watermark change, or an Iceberg-table repoint) lives only
        in the CREATE SOURCE definition, not in the MV SELECT, so it is read back from the source catalog and
        triggers the same drop+reprovision — else the MV keeps reading the stale source."""
        mv = online_mv_name(project, view.name)
        src = source_name(project, view.name)
        desired = self._passthrough_mv_select(project, view)
        mv_changed = _norm_sql(_deployed_mv_select(cur, mv)) != _norm_sql(desired)
        if is_passthrough_stream(view):
            # (topic, bootstrap): a passthrough source carries NO watermark (the latest-row MV is a
            # Group-TopN, not an EOWC window), so the watermark slot is not part of its identity.
            deployed_opts = _deployed_kafka_source_opts(cur, src)
            opts_changed = (
                deployed_opts is not None
                and deployed_opts[:2] != _desired_kafka_source_opts(view.stream_source)[:2]
            )
            # Column schema: a passthrough Kafka source declares the raw feature columns with explicit types,
            # so a feature dtype change (same name) must rebuild the source — it shows in no MV SELECT.
            deployed_cols = _deployed_source_columns(cur, src)
            schema_changed = deployed_cols is not None and any(
                deployed_cols.get(name) != dtype
                for name, dtype in _desired_passthrough_columns(view).items()
            )
            # The offline-history Iceberg source must exist and point at the stream's batch_source table:
            # rebuild if it is MISSING (an Iceberg batch_source added to a previously online-only view) or
            # REPOINTED (a different table) — neither shows in the MV SELECT or the Kafka opts.
            history = getattr(view.stream_source, "batch_source", None)
            history_changed = False
            if isinstance(history, IcebergSource):
                deployed_history = _deployed_source_table(
                    cur, passthrough_history_source_name(project, view.name)
                )
                history_changed = deployed_history is None or deployed_history != history.table
            source_changed = opts_changed or schema_changed or history_changed
        else:
            # A batch passthrough reads an Iceberg source that INFERS its schema, so a feature dtype change
            # needs no source rebuild; only a table repoint (read from the source catalog) does.
            deployed_table = _deployed_source_table(cur, src)
            source_changed = deployed_table is not None and deployed_table != view.batch_source.table
        if mv_changed or source_changed:
            for stmt in _passthrough_drop_ddl(project, view):
                _execute_ddl(cur, stmt)
            for stmt in self._provision_passthrough_ddl(project, view):
                _execute_ddl(cur, stmt)

    def _provision_batch_ddl(self, project: str, view) -> List[str]:
        """Registry-free DDL for one tile-aggregating BatchFeatureView: an Iceberg source, a continuous
        TILES MV (``build_batch_tile_select`` — per-(entity, tile_end) partials over window-independent
        partials), and ONE ONLINE ROLLUP MV PER DISTINCT WINDOW (``build_online_rollup_select`` — the
        request-anchored ``now()`` window rollup, one row per entity). One tile set is reused across all
        windows; each window gets its own now()-anchored MV because RisingWave rejects now()
        inside a CASE in a two-sided temporal-filter MV. RisingWave maintains every MV incrementally as
        the Iceberg table grows (validated end-to-end on RisingWave v3.0.0), so there is no scheduler. The point-lookup reads the
        per-window MV (``online_window_mv_name``) holding the requested feature's window — the tiles MV
        is internal plumbing. NO Iceberg sink yet: the MVs are read directly, and durable tile history
        is not provisioned. Offline training rolls the SAME tiles up per label timestamp."""
        interval = tile_interval(view)
        column_info = _batch_column_info(view)
        aggs = view_aggregations(view)
        params = view_agg_params(view)
        secondary_key = view_secondary_key(view)
        filters = view_agg_filters(view)
        src = source_name(project, view.name)
        tiles = tiles_name(project, view.name)
        ddl = [
            _iceberg_source_ddl(src, view.batch_source.table, self.config),
            _materialized_view_ddl(
                tiles,
                build_batch_tile_select(
                    column_info, aggs, src, aggregation_interval=interval, agg_params=params,
                    secondary_key=secondary_key, filters=filters,
                ),
            ),
        ]
        # The online rollup MVs: ONE cumulative MV for the invertible aggregations (read by 2-point
        # subtraction) + a now()-anchored per-(window, offset)/lifetime MV for each non-invertible one.
        # _desired_online_mvs is the single home of that split, shared with the reconcile so they cannot
        # drift; a series reuses the tiles MV / cumulative MV at read time, so it gets no MV here.
        for name, select in _desired_online_mvs(
            project, view.name, column_info, aggs, tiles,
            aggregation_interval=interval, agg_params=params, secondary_key=secondary_key,
            offsets=view_agg_offsets(view), lifetimes=view_agg_lifetime(view),
            series=view_agg_series(view), filters=filters,
        ).items():
            ddl.append(_materialized_view_ddl(name, select))
        return ddl

    def _provision_streaming_tile_ddl(self, project: str, view) -> List[str]:
        """Registry-free DDL for one STREAMING tile feature view: a watermarked Kafka source, a
        continuous EOWC TILES MV (``build_streaming_tile_select`` — per-(entity, tile_end)
        window-INDEPENDENT partials tumbled at the aggregation_interval), and ONE now()-anchored online
        rollup MV PER DISTINCT WINDOW (``build_online_rollup_select``). Same topology as the batch tile
        path (one tiles MV + N rollup MVs, NO Iceberg sink — the offline tile PIT reads the tiles MV
        directly); only the tile SOURCE differs: a watermarked Kafka source tumbled EMIT ON WINDOW CLOSE
        vs an Iceberg ``date_trunc`` GROUP BY. The rollup MVs and the offline PIT are byte-identical to
        the batch path, so serving/training reuse the same code.

        EOWC tiles require a watermark on the source timestamp: a late event is dropped once at its tile
        boundary, which is exactly what keeps online == offline (both read the same EOWC tiles).
        This is intrinsic to the tile model, so — unlike a plain stream MV, whose EOWC is opt-in via
        ``emit_on_window_close`` — the watermark is ALWAYS required here; reject a source that sets none
        (the EOWC tiles would never emit)."""
        source = view.stream_source
        if isinstance(source, PushSource):
            raise ValueError(
                f"streaming-tile view '{view.name}' uses a PushSource, which is too thin to compile to "
                "a RisingWave CREATE SOURCE (data_source.py:851-882). Use a KafkaSource."
            )
        if not isinstance(source, KafkaSource):
            raise ValueError(
                "RisingWaveComputeEngine requires a KafkaSource-backed streaming-tile view; "
                f"'{view.name}' has {type(source).__name__}."
            )
        if source.kafka_options.watermark_delay_threshold is None:
            raise ValueError(
                "a streaming-tile view's EOWC tiles require a watermark on the source timestamp (a late "
                "event is dropped once at its tile boundary, keeping online == offline), but the "
                f"KafkaSource for '{view.name}' sets no watermark_delay_threshold. Set one."
            )
        # enable_tiling without a tiling_hop_size: is_streaming_tile is True (it keys on enable_tiling +
        # aggregations) but tile_interval is None — fail loud with the fix rather than an opaque
        # 'NoneType' AttributeError deep in the tile SQL builder. (StreamFeatureView leaves
        # tiling_hop_size None when unset — its 5-min default is only a local for window validation.)
        if view.tiling_hop_size is None:
            raise ValueError(
                f"streaming-tile view '{view.name}' has enable_tiling=True but no tiling_hop_size (the "
                "tile interval). Set tiling_hop_size to 1 minute, 1 hour, or 1 day (the streaming tile grid)."
            )

        interval = tile_interval(view)
        column_info = _registry_free_column_info(view)
        aggs = view_aggregations(view)
        params = view_agg_params(view)
        secondary_key = view_secondary_key(view)
        filters = view_agg_filters(view)
        src = source_name(project, view.name)
        tiles = tiles_name(project, view.name)
        ddl = [
            _source_ddl(src, source, view),
            _materialized_view_ddl(
                tiles,
                build_streaming_tile_select(
                    column_info, aggs, src, aggregation_interval=interval, agg_params=params,
                    secondary_key=secondary_key, filters=filters,
                ),
            ),
        ]
        # Same online-rollup split as the batch path (byte-identical via _desired_online_mvs): ONE
        # cumulative MV for the invertible aggregations + a now()-anchored MV per non-invertible
        # (window, offset)/lifetime. Shared with the reconcile so provisioning and reconcile cannot drift.
        for name, select in _desired_online_mvs(
            project, view.name, column_info, aggs, tiles,
            aggregation_interval=interval, agg_params=params, secondary_key=secondary_key,
            offsets=view_agg_offsets(view), lifetimes=view_agg_lifetime(view),
            series=view_agg_series(view), filters=filters,
        ).items():
            ddl.append(_materialized_view_ddl(name, select))
        return ddl

    def _reconcile_batch_view(self, cur, project: str, view) -> None:
        """Reconcile a KEPT tile view's physical objects to its CURRENT definition (re-materialize on a
        materialization-affecting change; a no-op when unchanged). Feast routes a same-name edited view
        to views_to_keep without telling the engine what changed, and ``CREATE ... IF NOT EXISTS`` would
        silently keep the old MVs — so serving would diverge from the applied definition. RisingWave has
        no CREATE OR REPLACE, so we compare each object's desired definition against the one RW stores
        verbatim in its catalog and drop+recreate only what differs.

        Tiles MV changed (the per-tile partials — a different aggregation function/column) -> full
        re-materialize (drop the online MVs that depend on it, drop the tiles MV, re-provision). Tiles MV
        unchanged (window-independent partials, so adding/removing a window does not touch it) -> reconcile
        only the per-window online MVs: drop removed/redefined windows, create new/redefined ones, leave
        unchanged windows (and their serving) running untouched."""
        interval = tile_interval(view)
        column_info = _batch_column_info(view)
        aggs = view_aggregations(view)
        params = view_agg_params(view)
        secondary_key = view_secondary_key(view)
        filters = view_agg_filters(view)
        src = source_name(project, view.name)
        tiles = tiles_name(project, view.name)
        desired_tiles = build_batch_tile_select(
            column_info, aggs, src, aggregation_interval=interval, agg_params=params,
            secondary_key=secondary_key, filters=filters,
        )
        desired_online = _desired_online_mvs(
            project, view.name, column_info, aggs, tiles,
            aggregation_interval=interval, agg_params=params, secondary_key=secondary_key,
            offsets=view_agg_offsets(view), lifetimes=view_agg_lifetime(view),
            series=view_agg_series(view), filters=filters,
        )
        deployed_online = {
            name: _deployed_mv_select(cur, name)
            for name in _existing_online_mv_names(cur, project, view.name)
        }
        full_rebuild, drops, creates = _plan_batch_reconcile(
            desired_tiles=desired_tiles,
            desired_online=desired_online,
            deployed_tiles=_deployed_mv_select(cur, tiles),
            deployed_online=deployed_online,
        )
        # A repointed Iceberg table doesn't show in any MV definition (they read the source by its
        # stable name), so detect it from the source catalog and force a full re-materialize that also
        # drops+recreates the source — else serving would keep reading the OLD table.
        deployed_table = _deployed_source_table(cur, src)
        source_changed = deployed_table is not None and deployed_table != view.batch_source.table
        if source_changed:
            full_rebuild, drops, creates = True, list(deployed_online), []
        for name in drops:  # online MVs (dependents) first
            _execute_ddl(cur, f'DROP MATERIALIZED VIEW IF EXISTS "{name}"')
        if full_rebuild:
            _execute_ddl(cur, f'DROP MATERIALIZED VIEW IF EXISTS "{tiles}"')
            if source_changed:
                _execute_ddl(cur, f'DROP SOURCE IF EXISTS "{src}"')  # so CREATE SOURCE picks up the new table
            for stmt in self._provision_batch_ddl(project, view):
                _execute_ddl(cur, stmt)
        else:
            for name, select in creates:
                _execute_ddl(cur, f'CREATE MATERIALIZED VIEW IF NOT EXISTS "{name}" AS {select}')

    def _reconcile_streaming_tile_view(self, cur, project: str, view) -> None:
        """Reconcile a KEPT streaming-tile view to its CURRENT definition — the streaming analogue of
        ``_reconcile_batch_view`` (and the tile analogue of ``_reconcile_stream_view``). Feast routes a
        same-name edited view to views_to_keep without saying what changed, and ``CREATE ... IF NOT
        EXISTS`` would keep the old MVs, so serving/training would diverge from the applied definition.
        RisingWave has no CREATE OR REPLACE, so compare each object's desired definition against the one
        RW stores verbatim in its catalog and drop+recreate only what differs — the SAME pure planner
        (``_plan_batch_reconcile``) the batch tile path uses, since the tile graph is identical (one tiles
        MV + N per-window online MVs); only the tiles MV's source/SELECT differs.

        EOWC tiles MV changed (different per-tile partials) -> full re-materialize (drop the online MVs,
        the tiles MV, re-provision). Tiles MV unchanged (window-independent partials, so adding/removing a
        window does not touch it) -> reconcile only the per-window online MVs.

        A repointed topic/bootstrap or a changed watermark delay does not show in any MV definition (the
        tiles MV reads the source by its stable name), so it is read back from the source catalog
        (``_deployed_kafka_source_opts``) and, on a difference, forces a full re-materialize (which already
        drops+recreates the source) — else the tiles keep reading the stale topic, or admit late events
        under the old watermark, diverging online from offline."""
        interval = tile_interval(view)
        column_info = _registry_free_column_info(view)
        aggs = view_aggregations(view)
        params = view_agg_params(view)
        secondary_key = view_secondary_key(view)
        filters = view_agg_filters(view)
        src = source_name(project, view.name)
        tiles = tiles_name(project, view.name)
        desired_tiles = build_streaming_tile_select(
            column_info, aggs, src, aggregation_interval=interval, agg_params=params,
            secondary_key=secondary_key, filters=filters,
        )
        desired_online = _desired_online_mvs(
            project, view.name, column_info, aggs, tiles,
            aggregation_interval=interval, agg_params=params, secondary_key=secondary_key,
            offsets=view_agg_offsets(view), lifetimes=view_agg_lifetime(view),
            series=view_agg_series(view), filters=filters,
        )
        deployed_online = {
            name: _deployed_mv_select(cur, name)
            for name in _existing_online_mv_names(cur, project, view.name)
        }
        full_rebuild, drops, creates = _plan_batch_reconcile(
            desired_tiles=desired_tiles,
            desired_online=desired_online,
            deployed_tiles=_deployed_mv_select(cur, tiles),
            deployed_online=deployed_online,
        )
        # A repointed topic/bootstrap or a changed watermark lives only in the CREATE SOURCE definition (the
        # tiles MV reads the source by its stable name), so detect it from the source catalog and force a
        # full re-materialize — which already drops+recreates the source below — else the tiles keep reading
        # the stale topic, or admit late events under the old watermark, diverging online from offline.
        deployed_opts = _deployed_kafka_source_opts(cur, src)
        source_changed = (
            deployed_opts is not None
            and deployed_opts != _desired_kafka_source_opts(view.stream_source)
        )
        if source_changed:
            full_rebuild, drops, creates = True, list(deployed_online), []
        for name in drops:  # online MVs (dependents) first
            _execute_ddl(cur, f'DROP MATERIALIZED VIEW IF EXISTS "{name}"')
        if full_rebuild:
            _execute_ddl(cur, f'DROP MATERIALIZED VIEW IF EXISTS "{tiles}"')
            # Drop the source too (unlike the batch reconcile): a streaming source declares its agg-input
            # columns EXPLICITLY (_source_ddl), so a partials change that adds an aggregation over a NEW
            # input column also changes the source schema — and CREATE SOURCE IF NOT EXISTS would keep the
            # old columns, so the rebuilt tiles MV would reference a column the source lacks. Dropping it
            # (the source is dependency-free once the tiles MV is gone) lets CREATE SOURCE pick up the new
            # columns. The batch reconcile needs no analogue: an Iceberg source infers its schema, so a new
            # column is read without re-creating the source.
            _execute_ddl(cur, f'DROP SOURCE IF EXISTS "{src}"')
            for stmt in self._provision_streaming_tile_ddl(project, view):
                _execute_ddl(cur, stmt)
        else:
            for name, select in creates:
                _execute_ddl(cur, f'CREATE MATERIALIZED VIEW IF NOT EXISTS "{name}" AS {select}')

    def _materialize_one(
        self, registry: BaseRegistry, task: MaterializationTask, **kwargs
    ) -> MaterializationJob:
        view = task.feature_view
        job_id = f"{view.name}-{task.start_time}-{task.end_time}"
        if not getattr(view, "offline", False):
            # Online-only: the live MV serves online; nothing to backfill offline.
            return RisingWaveMaterializationJob(job_id, MaterializationJobStatus.SUCCEEDED)
        if is_tile_view(view):
            # A tile view's offline training reads the live tiles MV directly (the tile PIT rollup), and that
            # MV already holds the full history its source carries — there is no durable offline table to
            # backfill. The plain windowed-agg backfill below would instead compute non-tile values (skewing
            # offline from the tile rollup served online) and target a staging table a tile view never
            # provisions, so a tile view materializes offline to a no-op.
            return RisingWaveMaterializationJob(job_id, MaterializationJobStatus.SUCCEEDED)
        try:
            builder = RisingWaveFeatureBuilder(
                registry,
                view,
                task,
                source_is_retractable=_source_is_retractable(view.stream_source),
                emit_on_close=False,  # bounded batch backfill, not a streaming MV
            )
            sql = builder.build().to_sql(self.get_execution_context(registry, task))
            # UNVERIFIED end-to-end: the windowed-agg -> offline-table backfill is not yet
            # proven in-repo. Preferred long-term: read the live sink's Iceberg history so
            # backfill == what was served. The bounded [start, end) predicate must be
            # applied here before execution.
            with _connect(self.config) as conn, conn.cursor() as cur:
                cur.execute(sql)
            return RisingWaveMaterializationJob(job_id, MaterializationJobStatus.SUCCEEDED)
        except BaseException as e:
            return RisingWaveMaterializationJob(
                job_id, MaterializationJobStatus.ERROR, error=e
            )

    # NOTE: get_historical_features is intentionally NOT implemented here. Feast's
    # provider routes training/PIT retrieval to the OFFLINE store
    # (RisingWaveOfflineStore), never to the compute engine — implementing it on the
    # engine would be dead code. The base ComputeEngine raises NotImplementedError.
