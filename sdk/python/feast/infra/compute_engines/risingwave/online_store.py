"""RisingWaveOnlineStore — read-only online serving over the materialized views the
RisingWave compute engine provisions.

In this architecture RisingWave both computes and serves: the online feature row IS
the latest bucket of the engine's materialized view, queried over the Postgres wire
protocol (port 4566). Writes are owned by the MV + Iceberg sink, NOT by client
inserts — so ``online_write_batch`` deliberately refuses (spike risk #9).

Status: SCAFFOLD. The point-lookup query is the verified part; the RisingWave-column
-> Feast ``ValueProto`` conversion fidelity is spike-gated (risk #9).
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from feast import Entity
from feast.feature_view import FeatureView
from feast.infra.online_stores.online_store import OnlineStore
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import FeastConfigBaseModel, RepoConfig

_ONLINE_STORE_PATH = (
    "feast.infra.compute_engines.risingwave.online_store.RisingWaveOnlineStore"
)

_VALUE_PROTO_FIELDS = (
    "string_val",
    "int64_val",
    "int32_val",
    "double_val",
    "float_val",
    "bool_val",
    "bytes_val",
)


class RisingWaveOnlineStoreConfig(FeastConfigBaseModel):
    type: Literal[_ONLINE_STORE_PATH] = _ONLINE_STORE_PATH
    host: str = "localhost"
    port: int = 4566
    database: str = "dev"
    user: Optional[str] = "root"
    password: Optional[str] = None


def _connect(config):
    import psycopg

    return psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.database,
        user=config.user,
        password=config.password,
    )


def _entity_value_to_python(value: ValueProto):
    # Minimal ValueProto -> python for the entity-key WHERE params. Spike-gated:
    # extend to the full Feast value type set (lists, timestamps, etc.).
    for field in _VALUE_PROTO_FIELDS:
        if value.HasField(field):
            return getattr(value, field)
    raise ValueError(f"Unsupported entity key value type: {value}")


class RisingWaveOnlineStore(OnlineStore):
    def online_read(
        self,
        config: RepoConfig,
        table: FeatureView,
        entity_keys: List[EntityKeyProto],
        requested_features: Optional[List[str]] = None,
    ) -> List[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]]:
        from feast.type_map import python_values_to_proto_values

        mv = f"{config.project}_{table.name}_online"
        field_types = {f.name: f.dtype.to_value_type() for f in table.features}
        names = requested_features or list(field_types)

        results: List[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]] = []
        with _connect(config.online_store) as conn, conn.cursor() as cur:
            for entity_key in entity_keys:
                join_keys = list(entity_key.join_keys)
                params = [_entity_value_to_python(v) for v in entity_key.entity_values]
                where = " AND ".join(f'"{k}" = %s' for k in join_keys)
                cols = ", ".join(f'"{n}"' for n in names)
                cur.execute(
                    f'SELECT {cols}, "window_end" FROM "{mv}" WHERE {where} '
                    'ORDER BY "window_end" DESC LIMIT 1',
                    params,
                )
                row = cur.fetchone()
                if row is None:
                    results.append((None, None))
                    continue
                event_ts = row[-1]
                # Spike-gated (risk #9): RisingWave column type -> ValueProto fidelity.
                feature_dict = {
                    name: python_values_to_proto_values([value], field_types[name])[0]
                    for name, value in zip(names, row[:-1])
                }
                results.append((event_ts, feature_dict))
        return results

    def online_write_batch(
        self,
        config: RepoConfig,
        table: FeatureView,
        data: List[
            Tuple[EntityKeyProto, Dict[str, ValueProto], datetime, Optional[datetime]]
        ],
        progress: Optional[Callable[[int], Any]],
    ) -> None:
        # Online features are populated by the compute engine's materialized view and
        # Iceberg sink, not by client writes (risk #9). Materialization through this
        # store is a no-op by design.
        raise NotImplementedError(
            "RisingWaveOnlineStore is read-only: online features are maintained by the "
            "RisingWaveComputeEngine's materialized view, not by client writes."
        )

    def update(
        self,
        config: RepoConfig,
        tables_to_delete: Sequence[FeatureView],
        tables_to_keep: Sequence[FeatureView],
        entities_to_delete: Sequence[Entity],
        entities_to_keep: Sequence[Entity],
        partial: bool,
    ):
        # Materialized views are provisioned/torn down by the compute engine
        # (engine.update / engine.teardown_infra), not the online store.
        pass

    def teardown(
        self,
        config: RepoConfig,
        tables: Sequence[FeatureView],
        entities: Sequence[Entity],
    ):
        pass
