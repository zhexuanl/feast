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
    is_tile_view,
    tile_interval,
    view_aggregations,
)
from feast.infra.compute_engines.risingwave.names import tiles_name
from feast.infra.compute_engines.risingwave.nodes import build_offline_tile_pit_query
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
    )
    return PostgreSQLRetrievalJob(
        query=query,
        config=config,
        full_feature_names=full_feature_names,
        on_demand_feature_views=OnDemandFeatureView.get_requested_odfvs(
            feature_refs, project, registry
        ),
    )
