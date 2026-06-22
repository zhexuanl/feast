"""RisingWave compute-engine config + connection + the Feast-dtype type maps.

Leaf module of the RisingWave engine: it depends on no other engine module, and the
DDL builders / reconcile readers depend on its type maps. ``engine.py`` re-exports
every symbol here so existing imports keep resolving.
"""

from typing import Literal, Optional

from feast.repo_config import FeastConfigBaseModel

_ENGINE_PATH = (
    "feast.infra.compute_engines.risingwave.engine.RisingWaveComputeEngine"
)

# Minimal Feast dtype -> RisingWave SQL type. Not yet complete: extend to the full Feast
# type system, and source raw-input column types from the source schema, before
# production. Raw aggregation-input columns default to DOUBLE PRECISION.
_RW_TYPE = {
    "Int64": "BIGINT",
    "Int32": "INT",
    "Float64": "DOUBLE PRECISION",
    "Float32": "REAL",
    "String": "VARCHAR",
    "Bool": "BOOLEAN",
    "Bytes": "BYTEA",
    "UnixTimestamp": "TIMESTAMP",
}

# RisingWave CREATE-clause type -> the canonical name RisingWave reports in information_schema.columns
# (verified live on v3.0.0). Used to compare a deployed source's column types against the desired schema,
# since the catalog reports canonical names ("double precision", "character varying") rather than the
# CREATE-clause form ("DOUBLE PRECISION", "VARCHAR").
_RW_CANONICAL_TYPE = {
    "BIGINT": "bigint",
    "INT": "integer",
    "DOUBLE PRECISION": "double precision",
    "REAL": "real",
    "VARCHAR": "character varying",
    "BOOLEAN": "boolean",
    "BYTEA": "bytea",
    "TIMESTAMP": "timestamp without time zone",
}


def _canonical_type(dtype) -> str:
    # The canonical information_schema type for a Feast dtype, defaulting to VARCHAR's canonical form for an
    # unmapped dtype (matching _passthrough_source_ddl's VARCHAR fallback).
    return _RW_CANONICAL_TYPE.get(_RW_TYPE.get(str(dtype), "VARCHAR"), "character varying")


class RisingWaveComputeEngineConfig(FeastConfigBaseModel):
    """Config for the RisingWave compute engine. Set as ``batch_engine`` (and/or the
    per-view ``stream_engine``) in ``feature_store.yaml``."""

    type: Literal[_ENGINE_PATH] = _ENGINE_PATH
    """Full module path to the engine class (no core repo_config.py registration)."""

    host: str = "localhost"
    port: int = 4566  # standalone.rs:331
    database: str = "dev"  # standalone.rs:332
    user: Optional[str] = "root"  # standalone.rs:333
    password: Optional[str] = None

    # Offline Iceberg sink — the well-governed history Feast point-in-time-joins over.
    catalog_name: str = "feast"
    catalog_type: str = "storage"
    warehouse_path: Optional[str] = None
    iceberg_database: str = "feast"
    s3_endpoint: Optional[str] = None
    s3_region: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None

    # Pin EMIT ON WINDOW CLOSE so the online MV and the offline history agree
    # (consistency over freshness). Requires a watermark on the stream source.
    emit_on_window_close: bool = True


def _connect(config):
    import psycopg  # feast's postgres stores use psycopg v3 (postgres.py:19)

    return psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.database,
        user=config.user,
        password=config.password,
        autocommit=True,
    )
