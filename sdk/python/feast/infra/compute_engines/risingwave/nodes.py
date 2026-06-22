"""RisingWave DAG nodes.

Design choice: nodes are **pure SQL builders** — they never open a database
connection. Each node composes a RisingWave SQL relation string (a CTE/subquery)
and returns a ``DAGValue(data=<relation>, format=DAGFormat.RISINGWAVE,
metadata={"columns": [...]})``, flowing the column list forward through metadata
(mirroring Flink's ``_get_columns`` / ``_sql_value``). The single composed query is
executed only at the edges — the ``RisingWaveDAGRetrievalJob`` (a terminal SELECT
over pgwire) or the ``RWOutputNode`` materialize INSERT (run by the engine). This
keeps the correctness logic unit-testable without a live RisingWave, and matches a
SQL-pushdown engine.

The shared SQL builders that hold this engine's correctness invariants live in three
sibling modules and are re-exported here so existing imports keep working:
``aggregation_carriers`` (the engine-owned view-tag readers + aggregation grouping),
``tiling`` (the partial-aggregate IR algebra), and ``sql_builders`` (every SQL builder
+ its validation helpers).

Every SQL fragment traces to a RisingWave end-to-end example validated against a live
instance. Anything not yet validated end-to-end is marked ``UNVERIFIED`` and listed under
the unvalidated surfaces in ``README.md``.
"""

from typing import List, Optional

import pandas as pd

from feast.aggregation import Aggregation
from feast.infra.compute_engines.dag.context import ColumnInfo, ExecutionContext
from feast.infra.compute_engines.dag.model import DAGFormat
from feast.infra.compute_engines.dag.node import DAGNode
from feast.infra.compute_engines.dag.value import DAGValue
from feast.infra.compute_engines.risingwave.aggregation_carriers import (
    AGG_LIFETIME_TAG,
    AGG_OFFSET_TAG,
    AGG_PARAMS_TAG,
    AGG_SERIES_TAG,
    SECONDARY_KEY_TAG,
    _agg_offset_secs,
    encode_agg_lifetime,
    encode_agg_offsets,
    encode_agg_params,
    encode_agg_series,
    encode_secondary_key,
    group_aggregations_by_window,
    group_aggregations_by_window_offset,
    group_lifetime_aggregations,
    is_lifetime_agg,
    is_series_agg,
    view_agg_lifetime,
    view_agg_offsets,
    view_agg_params,
    view_agg_series,
    view_secondary_key,
)
from feast.infra.compute_engines.risingwave.names import (
    offline_staging_name,
    source_name,
)
from feast.infra.compute_engines.risingwave.sql_builders import (
    DEDUP_ROW_NUMBER,
    MONOID_FUNCTIONS,
    SUPPORTED_AGG_FUNCTIONS,
    build_batch_tile_select,
    build_cumulative_read_query,
    build_cumulative_tile_select,
    build_latest_row_select,
    build_lifetime_rollup_select,
    build_offline_tile_pit_query,
    build_online_rollup_select,
    build_passthrough_pit_query,
    build_series_snapshot_select,
    build_streaming_tile_select,
    build_tile_rollup_select,
    build_windowed_agg_select,
    compose_multi_view_pit_query,
)
from feast.infra.compute_engines.risingwave.tiling import (
    _assert_tile_supported,
    _cumulative_partials,
    _cumulative_recombine_expr,
    _partials_for,
    _recombine_expr,
    _sequence_n,
    _series_recombine,
    _tile_recombine,
    _tile_value_expr,
    _view_partials,
    cumulative_tile_recombine,
    is_invertible_agg,
    snapshot_series_aggs,
)
from feast.infra.compute_engines.utils import (
    ENTITY_ROW_ID,
    ENTITY_TS_ALIAS,
    find_entity_timestamp_column,
    infer_entity_timestamp_column,
)

# The SQL builders + view-tag carriers + tiling algebra now live in sibling modules; they are
# re-exported here (listed in ``__all__``) so every existing ``from ...risingwave.nodes import``
# keeps resolving without churn.
__all__ = [
    # aggregation_carriers
    "AGG_LIFETIME_TAG",
    "AGG_OFFSET_TAG",
    "AGG_PARAMS_TAG",
    "AGG_SERIES_TAG",
    "SECONDARY_KEY_TAG",
    "_agg_offset_secs",
    "encode_agg_lifetime",
    "encode_agg_offsets",
    "encode_agg_params",
    "encode_agg_series",
    "encode_secondary_key",
    "group_aggregations_by_window",
    "group_aggregations_by_window_offset",
    "group_lifetime_aggregations",
    "is_lifetime_agg",
    "is_series_agg",
    "view_agg_lifetime",
    "view_agg_offsets",
    "view_agg_params",
    "view_agg_series",
    "view_secondary_key",
    # tiling
    "_assert_tile_supported",
    "_cumulative_partials",
    "_cumulative_recombine_expr",
    "_partials_for",
    "_recombine_expr",
    "_sequence_n",
    "_series_recombine",
    "_tile_recombine",
    "_tile_value_expr",
    "_view_partials",
    "cumulative_tile_recombine",
    "is_invertible_agg",
    "snapshot_series_aggs",
    # sql_builders
    "DEDUP_ROW_NUMBER",
    "MONOID_FUNCTIONS",
    "SUPPORTED_AGG_FUNCTIONS",
    "build_batch_tile_select",
    "build_cumulative_read_query",
    "build_cumulative_tile_select",
    "build_latest_row_select",
    "build_lifetime_rollup_select",
    "build_offline_tile_pit_query",
    "build_online_rollup_select",
    "build_passthrough_pit_query",
    "build_series_snapshot_select",
    "build_streaming_tile_select",
    "build_tile_rollup_select",
    "build_windowed_agg_select",
    "compose_multi_view_pit_query",
    # DAG nodes (defined below)
    "RWSourceNode",
    "RWJoinNode",
    "RWFilterNode",
    "RWAggregationNode",
    "RWDedupNode",
    "RWTransformNode",
    "RWValidationNode",
    "RWOutputNode",
]


def _quote(identifier: str) -> str:
    return f'"{identifier}"'


class _RWNode(DAGNode):
    """Base for RisingWave nodes that carry a SQL relation forward.

    The relation is stored under ``DAGValue.data`` as the SQL string; the column
    list is carried in ``metadata["columns"]`` so downstream nodes can reference
    columns without re-parsing the SQL (mirrors Flink's metadata-carried columns).
    """

    def __init__(self, name, view, column_info: ColumnInfo, inputs=None):
        super().__init__(name, inputs=inputs)
        self.view = view
        self.column_info = column_info

    def _input_value(self, context: ExecutionContext) -> DAGValue:
        value = self.get_single_input_value(context)
        value.assert_format(DAGFormat.RISINGWAVE)
        return value

    def _input_relation(self, context: ExecutionContext) -> str:
        return self._input_value(context).data

    @staticmethod
    def _columns(value: DAGValue) -> List[str]:
        columns = value.metadata.get("columns") if value.metadata else None
        if columns:
            return list(columns)
        raise ValueError(
            "Could not infer columns for RisingWave DAG value from metadata."
        )

    def _value(
        self, relation: str, columns: List[str], *, metadata: Optional[dict] = None
    ) -> DAGValue:
        return DAGValue(
            data=relation,
            format=DAGFormat.RISINGWAVE,
            metadata={**(metadata or {}), "columns": list(columns)},
        )


def _agg_output_columns(column_info: ColumnInfo, aggregations: List[Aggregation]) -> List[str]:
    columns = list(column_info.join_keys_columns)
    columns.extend(a.resolved_name(a.time_window) for a in aggregations)
    if aggregations and aggregations[0].time_window is not None:
        columns.extend(["window_start", "window_end"])
    return columns


class RWSourceNode(_RWNode):
    """Source read.

    For retrieval the relation is the RisingWave object holding governed history (the
    Iceberg-history read-back source, or the online MV); for a materialize backfill it
    is the same provisioned relation, narrowed to the bounded window by the filter
    node. Either way the engine provisions ``{project}_{view}_src`` in
    ``engine.update()``; we reference it and carry the source schema columns.
    """

    def execute(self, context: ExecutionContext) -> DAGValue:
        relation = _quote(source_name(context.project, self.view.name))
        columns = list(self.column_info.join_keys_columns)
        ts = self.column_info.timestamp_column
        if ts and ts not in columns:
            columns.append(ts)
        for feature_col in self.column_info.feature_cols:
            if feature_col not in columns:
                columns.append(feature_col)
        return self._value(
            relation,
            columns,
            metadata={
                "source": "feature_view_source",
                "timestamp_field": ts,
            },
        )


class RWJoinNode(_RWNode):
    """Entity-spine LEFT JOIN for historical retrieval.

    Mirrors ``FlinkJoinNode``: the entity_df (a pandas DataFrame OR a SQL string) is
    the LEFT side; feature rows are joined on the view join keys, and the entity
    event-timestamp column is aliased to ``ENTITY_TS_ALIAS`` so the downstream filter
    node can apply the point-in-time cut. ``ENTITY_ROW_ID`` is attached so dedup picks
    exactly one feature row per entity row.
    """

    def execute(self, context: ExecutionContext) -> DAGValue:
        feature_value = self._input_value(context)
        feature_relation = feature_value.data
        feature_columns = self._columns(feature_value)
        join_keys = self.column_info.join_keys_columns

        if context.entity_df is None:
            raise RuntimeError(
                f"RWJoinNode '{self.name}' requires an entity_df on the execution "
                "context for historical retrieval."
            )

        entity_relation, entity_columns, entity_ts_col = self._entity_relation(
            context.entity_df, join_keys
        )

        # Feature columns that are not join keys / entity columns flow through.
        feature_only = [
            col
            for col in feature_columns
            if col not in join_keys and col not in entity_columns
        ]
        on_clause = " AND ".join(
            f"e.{_quote(key)} = f.{_quote(key)}" for key in join_keys
        )
        select_entity = ", ".join(f"e.{_quote(col)}" for col in entity_columns)
        select_features = ", ".join(f"f.{_quote(col)}" for col in feature_only)
        projection = ", ".join(p for p in (select_entity, select_features) if p)
        select = (
            f"SELECT {projection} FROM ({entity_relation}) AS e "
            f"LEFT JOIN ({feature_relation}) AS f ON {on_clause}"
        )
        output_columns = entity_columns + feature_only
        # Carry the feature side's effective event-timestamp column forward (window_end
        # when the feature relation was already aggregated upstream of this join, on the
        # PIT-aggregated retrieval path). The downstream filter/dedup nodes read it to
        # apply the as-of cut on window_end rather than the (now-absent) raw ts.
        feature_event_ts = (feature_value.metadata or {}).get("event_timestamp_column")
        return self._value(
            f"({select})",
            output_columns,
            metadata={
                "joined_on": join_keys,
                "join_type": "left",
                "entity_timestamp_column": entity_ts_col,
                "event_timestamp_column": feature_event_ts,
            },
        )

    def _entity_relation(self, entity_df, join_keys: List[str]):
        """Build the entity-spine relation + its columns + its timestamp column.

        Handles ``entity_df`` as a pandas DataFrame OR a SQL string, using the shared
        ``find_entity_timestamp_column`` / ``infer_entity_timestamp_column`` helpers
        so the event-timestamp column resolves to ``ENTITY_TS_ALIAS``.
        """
        if isinstance(entity_df, pd.DataFrame):
            entity_columns = list(entity_df.columns)
            entity_schema = dict(zip(entity_df.columns, entity_df.dtypes))
            entity_ts_col = infer_entity_timestamp_column(entity_schema)
            # NOT YET IMPLEMENTED: a pandas entity_df must be staged into RisingWave (e.g. a
            # temporary table / VALUES list) before it can be joined over pgwire. We
            # reference a conventional staging relation here; the upload itself is not yet
            # implemented (mirrors Flink's pandas_to_flink_table staging).
            relation = f"SELECT * FROM {_quote(ENTITY_ROW_ID + '_spine')}"
            select_exprs = [
                f"{_quote(col)} AS {_quote(ENTITY_TS_ALIAS)}"
                if col == entity_ts_col
                else _quote(col)
                for col in entity_columns
            ]
            select_exprs.append(
                f"ROW_NUMBER() OVER () - 1 AS {_quote(ENTITY_ROW_ID)}"
            )
            relation = f"SELECT {', '.join(select_exprs)} FROM ({relation}) AS spine"
            output_columns = [
                ENTITY_TS_ALIAS if col == entity_ts_col else col
                for col in entity_columns
            ]
            output_columns.append(ENTITY_ROW_ID)
            return relation, output_columns, entity_ts_col

        if isinstance(entity_df, str):
            entity_sql = entity_df.strip()
            if not entity_sql:
                raise ValueError("SQL entity_df for RisingWave must be non-empty.")
            # Wrap the user SQL in a subquery; we cannot statically know its columns,
            # so SELECT *, find the timestamp column by name and re-alias it.
            entity_ts_col = find_entity_timestamp_column(
                [ENTITY_TS_ALIAS]
            ) or ENTITY_TS_ALIAS
            # The user SQL must already expose an ``event_timestamp`` (or the alias);
            # the filter node references ENTITY_TS_ALIAS, so re-alias to it.
            relation = (
                f"SELECT spine.*, spine.{_quote('event_timestamp')} AS "
                f"{_quote(ENTITY_TS_ALIAS)}, ROW_NUMBER() OVER () - 1 AS "
                f"{_quote(ENTITY_ROW_ID)} FROM ({entity_sql}) AS spine"
            )
            # Columns of a SQL spine are not statically known; carry only what the
            # filter/dedup nodes reference. Feature columns are added by the join.
            output_columns = [ENTITY_TS_ALIAS, ENTITY_ROW_ID] + join_keys
            return relation, output_columns, ENTITY_TS_ALIAS

        raise TypeError(
            "RisingWave entity_df must be a pandas DataFrame, a SQL string, or None."
        )


class RWFilterNode(_RWNode):
    """Filter: the point-in-time cut + the TTL lower bound + an optional view filter.

    Mirrors ``FlinkFilterNode``: when an entity_df is present (its ``ENTITY_TS_ALIAS``
    is in scope), apply the inclusive ``timestamp_column <= ENTITY_TS_ALIAS`` PIT cut
    (postgres.py:962) plus, if a ttl is set, ``timestamp_column >= ENTITY_TS_ALIAS -
    INTERVAL ttl``. Without an entity_df (the backfill path), apply the TTL window
    relative to ``now()``. An optional ``view.filter`` expression is ANDed on.

    The PIT/TTL cut is applied on the *effective* event-timestamp column: ``window_end``
    when the input relation was already windowed-aggregated (carried via metadata by
    ``RWAggregationNode``), otherwise the raw event ts. Cutting an aggregated relation
    on its raw ts would be impossible (the column is gone) and cutting the raw stream on
    ``ts <= ENTITY_TS_ALIAS`` *before* a window closes would admit partial, future-dated
    windows — so on the aggregated-PIT path the builder runs this node AFTER aggregation
    with ``include_pit_cut=True`` (cut on window_end), and a separate pre-aggregation
    node with ``include_pit_cut=False`` for the raw-row ``view.filter`` only.
    """

    def __init__(
        self,
        name,
        view,
        column_info,
        *,
        filter_expr=None,
        ttl=None,
        inputs=None,
        include_pit_cut: bool = True,
    ):
        super().__init__(name, view, column_info, inputs=inputs)
        self.filter_expr = filter_expr
        self.ttl = ttl
        # When False, the PIT (``ts <= ENTITY_TS_ALIAS``) + TTL predicates are skipped
        # entirely. Used for the pre-aggregation filter on the aggregated-PIT path,
        # where the as-of cut must instead be applied on window_end AFTER aggregation
        # (a raw-ts cut there would leak partial/future-dated windows).
        self.include_pit_cut = include_pit_cut

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        relation = input_value.data
        columns = self._columns(input_value)
        # On an already-aggregated relation the raw event-timestamp column is gone and
        # window_end is the effective event timestamp (carried by RWAggregationNode). The
        # PIT cut MUST be applied on window_end, not the raw ts: cutting the raw ts before
        # the window closes leaks partial/future-dated windows. Fall back to the raw ts
        # column for the non-aggregated path.
        metadata = input_value.metadata or {}
        ts = metadata.get("event_timestamp_column") or self.column_info.timestamp_column
        conditions: List[str] = []

        if self.include_pit_cut and ts and ENTITY_TS_ALIAS in columns and ts in columns:
            # Inclusive <= per Feast's offline PIT join (postgres.py:962); a window
            # row stamped by window_end is only admitted once it has closed.
            conditions.append(f"{_quote(ts)} <= {_quote(ENTITY_TS_ALIAS)}")
            if self.ttl:
                secs = int(self.ttl.total_seconds())
                conditions.append(
                    f"{_quote(ts)} >= {_quote(ENTITY_TS_ALIAS)} - "
                    f"INTERVAL '{secs}' SECOND"
                )
        elif self.include_pit_cut and self.ttl and ts and ts in columns:
            secs = int(self.ttl.total_seconds())
            conditions.append(f"{_quote(ts)} >= now() - INTERVAL '{secs}' SECOND")

        if self.filter_expr:
            conditions.append(f"({self.filter_expr})")

        if not conditions:
            return input_value

        where = " AND ".join(conditions)
        # TUMBLE/HOP's 1st arg cannot be a sub-SELECT but CAN be a CTE/view
        # (window_table_function.rs:66), so we wrap as a subquery relation.
        select = f"SELECT * FROM ({relation}) AS _f WHERE {where}"
        return self._value(
            f"({select})",
            columns,
            # Preserve event_timestamp_column so a downstream dedup on the aggregated
            # path still knows to order by window_end (it is unchanged by the filter).
            metadata={**metadata, "filter_applied": True},
        )


class RWAggregationNode(_RWNode):
    """The value-add node: honors ``time_window`` (TUMBLE/HOP), which the Spark/Flink/
    Ray engines reject (aggregation/__init__.py:132-134). Also supports non-windowed
    GROUP BY (``time_window is None``)."""

    def __init__(
        self, name, view, column_info, *, source_is_retractable, emit_on_close, inputs=None
    ):
        super().__init__(name, view, column_info, inputs=inputs)
        self.source_is_retractable = source_is_retractable
        self.emit_on_close = emit_on_close

    def execute(self, context: ExecutionContext) -> DAGValue:
        aggregations = list(self.view.aggregations)
        input_value = self._input_value(context)
        select = build_windowed_agg_select(
            self.column_info,
            aggregations,
            input_value.data,
            source_is_retractable=self.source_is_retractable,
            emit_on_close=self.emit_on_close,
            agg_params=view_agg_params(self.view),
        )
        columns = _agg_output_columns(self.column_info, aggregations)
        # After windowed aggregation the raw event-timestamp column is gone; window_end
        # is the row's effective event timestamp (a window [t, t+w) is only knowable at
        # t+w). Carry it forward so the downstream PIT filter cuts on window_end — never
        # on the raw ts (which would admit a still-open/partial window with a
        # future-dated stamp). Non-windowed GROUP BY has no event timestamp.
        windowed = bool(aggregations) and aggregations[0].time_window is not None
        event_ts_col = "window_end" if windowed else None
        # Preserve the entity-spine columns (ENTITY_TS_ALIAS / ENTITY_ROW_ID) when the
        # join ran upstream of aggregation; they are NOT in the GROUP BY, so they only
        # exist if the builder placed the join AFTER aggregation (the PIT path).
        return self._value(
            f"({select})",
            columns,
            metadata={
                **(input_value.metadata or {}),
                "aggregated": True,
                "select": select,
                "event_timestamp_column": event_ts_col,
            },
        )


class RWDedupNode(_RWNode):
    """Latest-row-per-key for the only_latest / historical path. Mirrors
    ``FlinkDedupNode`` and the Postgres offline PIT dedup (postgres.py:1002-1012):
    ``ROW_NUMBER() OVER (PARTITION BY <keys> ORDER BY ts DESC[, created_ts DESC]) =
    1``. Partitions by ``ENTITY_ROW_ID`` when present (one feature row per entity
    spine row), else by the join keys.

    Orders by the effective event-timestamp column: ``window_end`` on the aggregated
    path (carried via metadata by ``RWAggregationNode`` — so a HOP/TUMBLE view collapses
    its many closed windows per label row to the single as-of-latest one), or the raw
    event ts (+created ts) otherwise. If neither is present it RAISES rather than
    silently picking an arbitrary row, which would break the latest-per-entity
    guarantee."""

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        relation = input_value.data
        columns = self._columns(input_value)

        dedup_keys = (
            [ENTITY_ROW_ID]
            if ENTITY_ROW_ID in columns
            else list(self.column_info.join_keys_columns)
        )
        dedup_keys = [key for key in dedup_keys if key in columns]
        if not dedup_keys:
            return input_value

        # On an aggregated relation the raw ts column has been dropped and window_end is
        # the effective event timestamp (carried by RWAggregationNode); order by it so
        # "latest per entity" means the latest CLOSED window. Fall back to the raw ts
        # (+created_ts) for the non-aggregated path.
        metadata = input_value.metadata or {}
        event_ts_col = metadata.get("event_timestamp_column")
        if event_ts_col:
            order_cols = [event_ts_col] if event_ts_col in columns else []
        else:
            ts = self.column_info.timestamp_column
            created_ts = self.column_info.created_timestamp_column
            order_cols = [c for c in (ts, created_ts) if c and c in columns]
        if not order_cols:
            # No time-ordering column means "latest per entity" cannot be defined;
            # an arbitrary pick would silently return a non-as-of row. Refuse instead
            # of degrading to dedup_keys[0] ASC.
            raise ValueError(
                f"[Dedup: {self.name}] Cannot select the latest row per "
                f"{dedup_keys}: no event-timestamp column is present in {sorted(columns)}. "
                "An ordered timestamp (raw event ts, or window_end for an aggregated "
                "view) is required to define the point-in-time-latest row."
            )
        order_exprs = [f"{_quote(c)} DESC" for c in order_cols]

        partition = ", ".join(_quote(k) for k in dedup_keys)
        order_by = ", ".join(order_exprs)
        projection = ", ".join(_quote(c) for c in columns)
        select = (
            f"SELECT {projection} FROM (SELECT *, ROW_NUMBER() OVER ("
            f"PARTITION BY {partition} ORDER BY {order_by}) AS "
            f"{_quote(DEDUP_ROW_NUMBER)} FROM ({relation}) AS _d) AS _r "
            f"WHERE {_quote(DEDUP_ROW_NUMBER)} = 1"
        )
        return self._value(
            f"({select})",
            columns,
            metadata={**(input_value.metadata or {}), "deduped": True},
        )


class RWTransformNode(_RWNode):
    """RisingWave SQL/UDF transformation.

    A transformation is only honored when it is expressible as a RisingWave SQL
    string (``view.feature_transformation`` exposing a ``.sql`` / ``.expr``). Anything
    that requires running arbitrary python out-of-engine raises NotImplementedError —
    honest, mirroring Flink's JSON-validation NotImplementedError (flink/nodes.py:723).
    """

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        relation = input_value.data
        columns = self._columns(input_value)

        transform = getattr(self.view, "feature_transformation", None)
        sql_expr = getattr(transform, "sql", None) or getattr(transform, "expr", None)
        if not sql_expr or not isinstance(sql_expr, str):
            raise NotImplementedError(
                "RisingWaveComputeEngine can only push down transformations expressed "
                "as a RisingWave SQL string (feature_transformation.sql). Arbitrary "
                "python/pandas/UDF transforms are not expressible in-engine "
                "(not supported). Pre-transform upstream, or use a SQL transformation."
            )
        # The SQL transform replaces the projection over the input relation; the
        # output column set is the view's declared features (unvalidated: we trust the
        # transform SQL to emit exactly these names).
        select = f"SELECT {sql_expr} FROM ({relation}) AS _t"
        output_columns = (
            [f.name for f in self.view.features]
            if getattr(self.view, "features", None)
            else columns
        )
        return self._value(
            f"({select})", output_columns, metadata={"transformed": True}
        )


class RWValidationNode(_RWNode):
    """Column-presence validation. Mirrors ``FlinkValidationNode``: assert every
    expected feature column is present in the relation's carried column list; raise
    with the actual columns if any are missing. (Type/JSON validation would require
    pulling data out of RisingWave and is not yet supported.)"""

    def __init__(self, name, view, column_info, *, expected_columns=None, inputs=None):
        super().__init__(name, view, column_info, inputs=inputs)
        self.expected_columns = expected_columns or []

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        columns = self._columns(input_value)
        missing = set(self.expected_columns) - set(columns)
        if missing:
            raise ValueError(
                f"[Validation: {self.name}] Missing expected columns: "
                f"{sorted(missing)}. Actual columns: {sorted(columns)}"
            )
        return self._value(
            input_value.data,
            columns,
            metadata={**(input_value.metadata or {}), "validated": True},
        )


class RWOutputNode(_RWNode):
    """Terminal node.

    For online + offline serving of a StreamFeatureView, the MV + Iceberg sink
    provisioned in ``engine.update()`` already own the per-row store writes (the live
    MV keeps online fresh; the sink streams the offline history). So the only per-task
    work here is the BOUNDED Iceberg backfill INSERT, and only for a materialize task:
    ``write_output = isinstance(task, MaterializationTask)`` (mirrors Flink, which
    gates the write on the task type).

    - retrieval terminal: carry the final SELECT as ``sql`` so the plan/job can run it
      over pgwire (``RisingWaveDAGRetrievalJob``).
    - materialize terminal: emit the ``INSERT ... SELECT`` into the offline staging
      table, preserving window_end-as-event_timestamp and the append-only/composite-PK
      invariants owned by ``_iceberg_sink_ddl``.
    """

    def __init__(self, name, view, column_info, *, write_output: bool, inputs=None):
        super().__init__(name, view, column_info, inputs=inputs)
        self.write_output = write_output

    def execute(self, context: ExecutionContext) -> DAGValue:
        input_value = self._input_value(context)
        relation = input_value.data
        columns = self._columns(input_value)

        # The relation is either a bare object name ("foo") or a parenthesized
        # subquery "(SELECT ...)"; normalize to a runnable SELECT.
        if relation.startswith("(") and relation.endswith(")"):
            select_sql = relation[1:-1]
        else:
            select_sql = f"SELECT * FROM {relation}"

        sql: Optional[str] = select_sql
        if self.write_output and getattr(self.view, "offline", False):
            # window_end is already the event timestamp emitted by the aggregation
            # node; the staging table mirrors the live sink's projection so backfill
            # rows are byte-compatible with streamed rows.
            # UNVERIFIED end-to-end: the bounded backfill INSERT and
            # its late-data parity with the live stream are not yet proven in-repo.
            # Preferred long-term: read the live sink's Iceberg history so backfill ==
            # what was served. The bounded [start, end) predicate is applied
            # by the upstream filter node before this INSERT.
            staging = _quote(offline_staging_name(context.project, self.view.name))
            sql = f"INSERT INTO {staging} {select_sql}"

        return self._value(
            relation, columns, metadata={**(input_value.metadata or {}), "sql": sql}
        )
