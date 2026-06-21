"""RisingWaveOfflineStore — point-in-time training retrieval over RisingWave.

Feast routes ``get_historical_features`` to the OFFLINE store (the passthrough
provider calls ``offline_store.get_historical_features``, never the compute engine),
so the RisingWave training/PIT path lives here — not in the compute engine.

RisingWave speaks the Postgres wire protocol, so this subclasses the Postgres offline
store and reuses its proven point-in-time-join SQL (validated end-to-end against
RisingWave). It fixes the ONE RisingWave incompatibility: RisingWave table INSERTs are
processed by the streaming engine and are only visible after a checkpoint, so the
Postgres store's entity-DataFrame temp-table upload yields an empty entity set when the
PIT query runs on a fresh connection. This store inlines the entity DataFrame as SQL
(``embed_query`` / CTE mode) — no upload, no async-visibility gap.
"""

from typing import List, Literal, Optional, Union

import pandas as pd

from feast.feature_view import FeatureView
from feast.infra.compute_engines.dag.context import ColumnInfo
from feast.infra.compute_engines.risingwave.iceberg_source import (
    IcebergSource,
    is_passthrough_stream,
    is_passthrough_view,
    is_tile_view,
    tile_interval,
    view_aggregations,
)
from feast.infra.compute_engines.risingwave.names import (
    passthrough_history_source_name,
    source_name,
    tiles_name,
)
from feast.infra.compute_engines.risingwave.nodes import (
    build_offline_tile_pit_query,
    build_passthrough_pit_query,
)
from feast.infra.offline_stores import offline_utils
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
    EntitySelectMode,
    PostgreSQLOfflineStore,
    PostgreSQLOfflineStoreConfig,
    PostgreSQLRetrievalJob,
)
from feast.infra.offline_stores.offline_store import RetrievalJob
from feast.infra.registry.base_registry import BaseRegistry
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.repo_config import RepoConfig
from feast.utils import _parse_feature_ref

_OFFLINE_STORE_PATH = (
    "feast.infra.compute_engines.risingwave.offline_store.RisingWaveOfflineStore"
)


class RisingWaveOfflineStoreConfig(PostgreSQLOfflineStoreConfig):
    """RisingWave offline store config. Subclasses the Postgres config (RisingWave is
    Postgres-wire-compatible) so the parent's ``assert isinstance(..., PostgreSQL
    OfflineStoreConfig)`` holds. Defaults tuned for RisingWave: no SSL on pgwire, and
    inline the entity query (RisingWave INSERTs are async)."""

    type: Literal[_OFFLINE_STORE_PATH] = _OFFLINE_STORE_PATH  # type: ignore[assignment]
    sslmode: Optional[str] = "disable"
    entity_select_mode: EntitySelectMode = EntitySelectMode.embed_query


def _sql_literal(value, dtype) -> str:
    name = str(dtype)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "NULL"
    if name.startswith("datetime64"):
        ts = pd.Timestamp(value)
        if "," in name:  # tz-aware, e.g. datetime64[ns, UTC]
            return f"CAST('{ts.isoformat()}' AS TIMESTAMPTZ)"
        return f"CAST('{ts}' AS TIMESTAMP)"
    if name.startswith(("int", "uint")):
        return f"CAST({int(value)} AS BIGINT)"
    if name.startswith("float"):
        return f"CAST({float(value)} AS DOUBLE PRECISION)"
    if name == "bool":
        return "TRUE" if bool(value) else "FALSE"
    return "CAST('{}' AS VARCHAR)".format(str(value).replace("'", "''"))


def _entity_df_to_sql(entity_df: pd.DataFrame) -> str:
    """Render the entity DataFrame as a bare ``SELECT ... UNION ALL ...`` query.

    ``_get_entity_schema`` wraps a str entity_df as ``({entity_df}) AS sub`` and the
    PIT template embeds it under ``use_cte``, so a bare SELECT is the right form. This
    replaces the temp-table upload that RisingWave's async INSERTs break.
    """
    columns = list(entity_df.columns)
    dtypes = {c: entity_df.dtypes[c] for c in columns}
    selects = []
    for _, row in entity_df.iterrows():
        exprs = [f'{_sql_literal(row[c], dtypes[c])} AS "{c}"' for c in columns]
        selects.append("SELECT " + ", ".join(exprs))
    return " UNION ALL ".join(selects)


class RisingWaveOfflineStore(PostgreSQLOfflineStore):
    @staticmethod
    def get_historical_features(
        config: RepoConfig,
        feature_views: List[FeatureView],
        feature_refs: List[str],
        entity_df: Optional[Union[pd.DataFrame, str]],
        registry: BaseRegistry,
        project: str,
        full_feature_names: bool = False,
        **kwargs,
    ) -> RetrievalJob:
        tile_fvs = [fv for fv in feature_views if is_tile_view(fv)]
        if tile_fvs:
            return _tile_historical_features(
                config, feature_views, tile_fvs, feature_refs, entity_df,
                registry, project, full_feature_names,
            )
        passthrough_fvs = [fv for fv in feature_views if is_passthrough_view(fv)]
        if passthrough_fvs:
            # A passthrough view's point-in-time read is an as-of cut over its RAW history (the batch
            # source), not over the latest-row online MV — a Group-TopN that holds only the current row per
            # entity, so it cannot answer a past label timestamp. Routed to a custom read; the Postgres
            # parent would silently read the inert placeholder offline source or assert on an Iceberg source.
            if len(feature_views) != 1:
                raise NotImplementedError(
                    "RisingWave passthrough offline retrieval currently supports exactly one feature view "
                    f"per retrieval; got {len(feature_views)} ({len(passthrough_fvs)} passthrough). Mixing "
                    "a passthrough view with other views in one get_historical_features call is not yet "
                    "supported."
                )
            return _passthrough_historical_features(
                config, passthrough_fvs[0], feature_refs, entity_df, registry, project,
                full_feature_names,
            )
        # Inline a DataFrame entity_df as SQL so the parent uses its embed_query/CTE
        # path (no temp-table upload). RisingWave INSERTs are async, so an uploaded
        # entity table is empty when the PIT query runs -> 0 training rows.
        if isinstance(entity_df, pd.DataFrame):
            entity_df = _entity_df_to_sql(entity_df)
        return PostgreSQLOfflineStore.get_historical_features(
            config=config,
            feature_views=feature_views,
            feature_refs=feature_refs,
            entity_df=entity_df,
            registry=registry,
            project=project,
            full_feature_names=full_feature_names,
            **kwargs,
        )


def _requested_features(fv, feature_refs) -> List[str]:
    """The subset of ``fv``'s features requested in ``feature_refs``, in the view's declared order.

    The standard Feast offline contract returns ONLY the requested features — a call asking for a
    subset of a view's features must not pull every column. This mirrors the per-view selection
    ``_get_requested_feature_views_to_features_dict`` performs (match each reference's view against
    ``projection.name_to_use()``, collect its feature), reusing Feast's ``_parse_feature_ref`` so
    versioned references parse identically. Single-view by construction (the caller routes exactly one
    feature view here), so references for other views / on-demand views simply do not match.
    """
    view_name = fv.projection.name_to_use()
    requested = set()
    for ref in feature_refs:
        ref_fv, version_num, ref_feature = _parse_feature_ref(ref)
        ref_view = f"{ref_fv}@v{version_num}" if version_num is not None else ref_fv
        if ref_view == view_name:
            requested.add(ref_feature)
    return [f.name for f in fv.features if f.name in requested]


def _tile_historical_features(
    config, feature_views, tile_fvs, feature_refs, entity_df,
    registry, project, full_feature_names,
) -> RetrievalJob:
    # Floor-anchored range-agg PIT for tile feature views (build_offline_tile_pit_query). The
    # standard latest-row template anchors at the latest tile WITH DATA, not floor(label) — wrong
    # when recent intervals are empty. One tile FV per query for now; mixing tile
    # with stream/other FVs in one retrieval is not yet supported.
    if len(feature_views) != 1:
        raise NotImplementedError(
            "RisingWave offline tile rollup currently supports exactly one tile feature view per "
            f"retrieval; got {len(feature_views)} views ({len(tile_fvs)} tile). Mixing tile feature "
            "views with stream/other views in one get_historical_features call is not yet supported."
        )
    if entity_df is None:
        raise NotImplementedError(
            "RisingWave offline tile rollup requires an entity dataframe (the per-label PIT anchor); "
            "entity-less retrieval is not supported."
        )
    fv = tile_fvs[0]
    if isinstance(entity_df, pd.DataFrame):
        entity_columns = list(entity_df.columns)
        label_ts_column = offline_utils.infer_event_timestamp_from_entity_df(
            dict(entity_df.dtypes)
        )
        entity_df_sql = _entity_df_to_sql(entity_df)
    else:
        raise NotImplementedError(
            "RisingWave offline tile rollup currently requires a DataFrame entity_df (it is inlined "
            "as SQL); a SQL-string entity_df is not yet supported."
        )

    # view_aggregations reads the SAME spec the engine provisioned the tiles with — a batch tile view's
    # IcebergSource custom_options, or a streaming tile view's native StreamFeatureView.aggregations —
    # so the resolved column names cannot drift. The PIT below reads the tiles MV by name, identically
    # for both flavors (only the engine's tiles-MV source differs).
    aggregations = view_aggregations(fv)
    # Honor feature_refs: keep only the aggregations whose output feature was requested, so a subset
    # request rolls up (and projects) just those features — not every aggregation of the view.
    requested = set(_requested_features(fv, feature_refs))
    aggregations = [
        a for a in aggregations if a.resolved_name(a.time_window) in requested
    ]
    column_info = ColumnInfo(
        join_keys=[f.name for f in fv.entity_columns],
        feature_cols=[a.resolved_name(a.time_window) for a in aggregations],
        ts_col=label_ts_column,
        created_ts_col=None,
        field_mapping=None,
    )
    query = build_offline_tile_pit_query(
        entity_df_sql,
        entity_columns,
        label_ts_column,
        tiles_relation=tiles_name(project, fv.name),
        column_info=column_info,
        aggregations=aggregations,
        aggregation_interval=tile_interval(fv),
        full_feature_names=full_feature_names,
        view_name=fv.projection.name_to_use(),
    )
    return PostgreSQLRetrievalJob(
        query=query,
        config=config,
        full_feature_names=full_feature_names,
        on_demand_feature_views=OnDemandFeatureView.get_requested_odfvs(
            feature_refs, project, registry
        ),
    )


def _passthrough_historical_features(
    config, fv, feature_refs, entity_df, registry, project, full_feature_names,
) -> RetrievalJob:
    # Point-in-time training for a passthrough view: for each entity row, the latest RAW feature row
    # at-or-before its label timestamp (within ttl) — the as-of cut that makes offline == the latest-row
    # online MV. Reads the raw history relation, not the latest-row MV (a Group-TopN holding only the
    # current row). A batch passthrough's own Iceberg source IS the history; a streaming passthrough reads
    # the Iceberg source provisioned over its stream batch_source (the historical log backing the stream).
    if entity_df is None:
        raise NotImplementedError(
            "RisingWave passthrough offline retrieval requires an entity dataframe (the per-label PIT "
            "anchor); entity-less retrieval is not supported."
        )
    if not isinstance(entity_df, pd.DataFrame):
        raise NotImplementedError(
            "RisingWave passthrough offline retrieval currently requires a DataFrame entity_df (it is "
            "inlined as SQL); a SQL-string entity_df is not yet supported."
        )
    entity_columns = list(entity_df.columns)
    label_ts_column = offline_utils.infer_event_timestamp_from_entity_df(dict(entity_df.dtypes))
    entity_df_sql = _entity_df_to_sql(entity_df)

    if is_passthrough_stream(fv):
        history = getattr(fv.stream_source, "batch_source", None)
        if not isinstance(history, IcebergSource):
            raise NotImplementedError(
                f"passthrough stream view '{fv.name}' has no Iceberg batch source (the historical log "
                "backing the stream), so offline point-in-time training is not available; declare a "
                "batch_source on the stream's KafkaSource, or serve the view online only."
            )
        history_relation = passthrough_history_source_name(project, fv.name)
        ts_col = history.timestamp_field
    else:
        history_relation = source_name(project, fv.name)
        ts_col = fv.batch_source.timestamp_field

    # created_ts is intentionally omitted: the online latest-row MV and this read both order by event time
    # alone (the Iceberg history carries no created_timestamp_column), so online == offline on ties.
    column_info = ColumnInfo(
        join_keys=[f.name for f in fv.entity_columns],
        # Honor feature_refs: project only the requested features, not every feature of the view.
        feature_cols=_requested_features(fv, feature_refs),
        ts_col=ts_col,
        created_ts_col=None,
        field_mapping=None,
    )
    ttl = getattr(fv, "ttl", None)
    query = build_passthrough_pit_query(
        entity_df_sql,
        entity_columns,
        label_ts_column,
        history_relation=history_relation,
        column_info=column_info,
        ttl_seconds=int(ttl.total_seconds()) if ttl else None,
        full_feature_names=full_feature_names,
        view_name=fv.projection.name_to_use(),
    )
    return PostgreSQLRetrievalJob(
        query=query,
        config=config,
        full_feature_names=full_feature_names,
        on_demand_feature_views=OnDemandFeatureView.get_requested_odfvs(
            feature_refs, project, registry
        ),
    )
