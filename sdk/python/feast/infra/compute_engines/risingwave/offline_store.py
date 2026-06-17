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
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
    EntitySelectMode,
    PostgreSQLOfflineStore,
    PostgreSQLOfflineStoreConfig,
)
from feast.infra.offline_stores.offline_store import RetrievalJob
from feast.infra.registry.base_registry import BaseRegistry
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
