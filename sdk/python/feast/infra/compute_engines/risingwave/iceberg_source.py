"""IcebergSource — the batch DataSource for a tile feature view, and the single home for tile-view
detection + spec access.

A tile feature view aggregates over an Iceberg table that RisingWave reads via
``CREATE SOURCE ... connector='iceberg'`` (engine.py ``_iceberg_source_ddl``). This source carries
the per-view coordinate — the Iceberg ``table`` and the event ``timestamp_field`` — AND the tile
aggregation spec (``aggregations`` + ``aggregation_interval``). The catalog / warehouse / S3
connection lives once in the engine config, so it is not repeated per source.

Why the spec lives on the SOURCE, not on the feature view: Feast's registry has no
``batch_feature_views`` list, so a ``BatchFeatureView`` round-trips as a plain ``FeatureView`` proto
that DROPS its aggregations and its type. A ``CUSTOM_SOURCE``'s ``custom_options`` round-trips
cleanly (the contrib sources rely on this) — the source is the one object that survives intact. So a
tile feature view is just a plain ``feast.FeatureView`` whose ``batch_source`` is an ``IcebergSource``
carrying the spec; ``is_tile_fv(view)`` discriminates this BATCH tile flavor at every altitude
(provisioning, serving, training), while ``is_streaming_tile(view)`` discriminates the STREAMING flavor
and ``is_tile_view(view)`` is their union (serving/training key on the union). ``view_aggregations`` /
``tile_interval`` are the ONE way to read the spec for either flavor.
"""

import json
from datetime import timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from feast.aggregation import Aggregation
from feast.data_source import DataSource
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.repo_config import RepoConfig
from feast.value_type import ValueType

_CLASS_PATH = "feast.infra.compute_engines.risingwave.iceberg_source.IcebergSource"


def _encode_aggregations(aggregations: List[Aggregation]) -> List[dict]:
    # slide_interval is intentionally not carried: the tile model is tumbling-at-interval today.
    # A lifetime aggregation has no window, so window_secs is null (its lifetime-ness + floor ride the
    # separate lifetime carrier on the view tags).
    return [
        {
            "function": a.function,
            "column": a.column,
            "window_secs": (
                int(a.time_window.total_seconds()) if a.time_window is not None else None
            ),
            "name": a.name or None,
        }
        for a in aggregations
    ]


def _decode_aggregations(items: List[dict]) -> List[Aggregation]:
    return [
        Aggregation(
            column=d["column"],
            function=d["function"],
            time_window=(
                timedelta(seconds=d["window_secs"])
                if d.get("window_secs") is not None
                else None
            ),
            name=d.get("name"),
        )
        for d in items
    ]


class IcebergSource(DataSource):
    """A batch data source naming the Iceberg ``table`` a tile feature view aggregates over, plus the
    tile aggregation spec it is aggregated with (round-trips via ``custom_options``)."""

    def source_type(self) -> DataSourceProto.SourceType.ValueType:
        return DataSourceProto.CUSTOM_SOURCE

    def __init__(
        self,
        *,
        table: str,
        timestamp_field: str,
        aggregations: Optional[List[Aggregation]] = None,
        aggregation_interval: Optional[timedelta] = None,
        name: Optional[str] = None,
        description: Optional[str] = "",
        tags: Optional[Dict[str, str]] = None,
        owner: Optional[str] = "",
    ):
        """Args:
        table: the Iceberg table name (within the engine config's catalog/database).
        timestamp_field: the event-time column the tiles are bucketed by.
        aggregations / aggregation_interval: the tile spec (set by the BatchFeatureView factory).
        name: source name; defaults to ``table``.
        """
        if not table:
            raise ValueError("IcebergSource requires a non-empty 'table'.")
        if not timestamp_field:
            raise ValueError(
                "IcebergSource requires 'timestamp_field' (the event-time column the tiles bucket by)."
            )
        self.table = table
        self.aggregations = list(aggregations) if aggregations else []
        self.aggregation_interval = aggregation_interval
        super().__init__(
            name=name or table,  # table is validated non-empty above, so the source always has a name
            timestamp_field=timestamp_field,
            description=description,
            tags=tags,
            owner=owner,
        )

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, IcebergSource):
            raise TypeError("Comparisons should only involve IcebergSource class objects.")
        return (
            super().__eq__(other)
            and self.table == other.table
            and self.timestamp_field == other.timestamp_field
            and self.aggregations == other.aggregations
            and self.aggregation_interval == other.aggregation_interval
        )

    @staticmethod
    def from_proto(data_source: DataSourceProto) -> "IcebergSource":
        assert data_source.HasField("custom_options")
        cfg = json.loads(data_source.custom_options.configuration)
        interval_secs = cfg.get("aggregation_interval_secs")
        return IcebergSource(
            name=cfg["name"],
            table=cfg["table"],
            timestamp_field=data_source.timestamp_field,
            aggregations=_decode_aggregations(cfg.get("aggregations") or []),
            aggregation_interval=(
                timedelta(seconds=interval_secs) if interval_secs is not None else None
            ),
            description=data_source.description,
            tags=dict(data_source.tags),
            owner=data_source.owner,
        )

    def _to_proto_impl(self) -> DataSourceProto:
        cfg = {
            "name": self.name,
            "table": self.table,
            "aggregations": _encode_aggregations(self.aggregations),
            "aggregation_interval_secs": (
                int(self.aggregation_interval.total_seconds())
                if self.aggregation_interval is not None
                else None
            ),
        }
        proto = DataSourceProto(
            name=self.name,
            type=DataSourceProto.CUSTOM_SOURCE,
            data_source_class_type=_CLASS_PATH,
            custom_options=DataSourceProto.CustomSourceOptions(
                configuration=json.dumps(cfg).encode()
            ),
            description=self.description,
            tags=self.tags,
            owner=self.owner,
        )
        proto.timestamp_field = self.timestamp_field
        return proto

    def validate(self, config: RepoConfig):
        pass

    @staticmethod
    def source_datatype_to_feast_value_type() -> Callable[[str], ValueType]:
        # Never called: the feature view declares an explicit schema, so Feast does not infer types
        # from this source. Present only to satisfy the DataSource abstract interface.
        def _unsupported(_: str) -> ValueType:
            raise NotImplementedError(
                "IcebergSource does not infer column types; declare the feature view schema."
            )

        return _unsupported

    def get_table_column_names_and_types(
        self, config: RepoConfig
    ) -> Iterable[Tuple[str, str]]:
        raise NotImplementedError(
            "IcebergSource is metadata-only; the feature view declares its schema and training reads "
            "the tiles MV via the RisingWave offline store."
        )

    def get_table_query_string(self) -> str:
        return self.table


# --- tile-view detection + spec access (the ONE discriminator / accessor at every altitude) ---


def is_tile_fv(view) -> bool:
    """A tile feature view: a (plain) feature view whose ``batch_source`` is an ``IcebergSource``
    carrying a tile aggregation spec. The IcebergSource (a CUSTOM_SOURCE) survives Feast's registry
    round-trip with its type + custom_options intact, so this one check works at every altitude —
    provisioning (in-memory), serving, and training (registry-read). A stream view's source is a
    KafkaSource, so it is naturally excluded."""
    src = getattr(view, "batch_source", None)
    return isinstance(src, IcebergSource) and bool(src.aggregations)


def view_aggregations(view) -> List[Aggregation]:
    """The aggregations for an online-servable view, for BOTH a stream view (``view.aggregations``,
    intact via the StreamFeatureView proto) and a tile view (the spec on its ``IcebergSource``, since
    a plain FeatureView's own ``aggregations`` are dropped by the registry). One accessor so the
    resolved feature-column names cannot drift across provisioning / serving / training."""
    if is_tile_fv(view):
        return list(view.batch_source.aggregations)
    return list(getattr(view, "aggregations", None) or [])


def is_streaming_tile(view) -> bool:
    """A STREAMING tile feature view: a ``StreamFeatureView`` with Feast's NATIVE ``enable_tiling`` set and
    aggregations. We REUSE Feast's own ``StreamFeatureView.enable_tiling`` / ``tiling_hop_size`` (both
    round-trip through the registry) rather than inventing a carrier, so this one check works at
    every altitude. The tiles are materialized by an EOWC TUMBLE at ``tiling_hop_size``; everything
    downstream is the shared per-window rollup. Mutually exclusive with ``is_tile_fv`` (a stream view has
    no IcebergSource batch source)."""
    return bool(getattr(view, "enable_tiling", False)) and bool(getattr(view, "aggregations", None))


def is_tile_view(view) -> bool:
    """A tile feature view of EITHER flavor: a BATCH tile view (Iceberg-sourced, ``is_tile_fv``) or a
    STREAMING tile view (Kafka-sourced, ``is_streaming_tile``). Both serve online from per-window rollup
    MVs over a shared tiles MV and read offline via the tile PIT over that tiles MV — so the serving and
    training routing keys on this union (not the flavor), differing only in how the engine provisions the
    tiles MV (Iceberg ``date_trunc`` vs watermarked-Kafka EOWC tumble)."""
    return is_tile_fv(view) or is_streaming_tile(view)


def tile_interval(view) -> timedelta:
    """The aggregation_interval (tile size) for a tile view. A BATCH tile view carries it on its
    ``IcebergSource``; a STREAMING tile view carries it on Feast's native ``StreamFeatureView.tiling_hop_size``."""
    if is_streaming_tile(view):
        return view.tiling_hop_size
    return view.batch_source.aggregation_interval


# --- passthrough-view detection (raw columns, no aggregation — the latest row per entity) ---


def is_passthrough_fv(view) -> bool:
    """A BATCH passthrough feature view: a (plain) feature view whose ``batch_source`` is an ``IcebergSource``
    with feature columns but NO tile aggregation spec — its features are raw columns carried through
    unchanged and served as the latest row per entity (no aggregation). Mirrors ``is_tile_fv`` for the
    no-aggregation case; the two are mutually exclusive (a tile view requires aggregations)."""
    src = getattr(view, "batch_source", None)
    return (
        isinstance(src, IcebergSource)
        and not src.aggregations
        and bool(getattr(view, "features", None))
    )


def is_passthrough_stream(view) -> bool:
    """A STREAMING passthrough feature view: a stream (Kafka-sourced) view with feature columns but NO
    aggregations — its features are raw columns served as the latest row per entity. A streaming
    aggregation or streaming tile view has aggregations, so it is naturally excluded."""
    return (
        getattr(view, "stream_source", None) is not None
        and bool(getattr(view, "features", None))
        and not view_aggregations(view)
    )


def is_passthrough_view(view) -> bool:
    """A passthrough feature view of EITHER flavor — raw feature columns served as the latest row per entity
    (no aggregation), batch (Iceberg-sourced, ``is_passthrough_fv``) or streaming (Kafka-sourced,
    ``is_passthrough_stream``). Both provision one latest-row materialized view served by the same
    point-lookup as an aggregation view, differing only in the source (Iceberg vs Kafka)."""
    return is_passthrough_fv(view) or is_passthrough_stream(view)
