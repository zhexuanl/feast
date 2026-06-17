"""RisingWaveFeatureBuilder — translates a (Stream)FeatureView into a RisingWave DAG
of pure SQL-building nodes (see ``nodes.py``).

Mirrors ``FlinkFeatureBuilder`` for structural completeness: every ``build_*`` method
appends its node to ``self.nodes``; ``_build`` is overridden to add the entity-spine
join for a ``HistoricalRetrievalTask`` (``_should_join_entity_df``); the output node's
``write_output`` flag is ``isinstance(task, MaterializationTask)`` and it uses
``self.dag_root.view``; the validation node derives its expected columns from
``view.features``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import pandas as pd

from feast.infra.common.materialization_job import MaterializationTask
from feast.infra.common.retrieval_task import HistoricalRetrievalTask
from feast.infra.compute_engines.risingwave.nodes import (
    RWAggregationNode,
    RWDedupNode,
    RWFilterNode,
    RWJoinNode,
    RWOutputNode,
    RWSourceNode,
    RWTransformNode,
    RWValidationNode,
)
from feast.infra.compute_engines.risingwave.plan import RisingWaveExecutionPlan
from feast.infra.compute_engines.dag.node import DAGNode
from feast.infra.compute_engines.dag.plan import ExecutionPlan
from feast.infra.compute_engines.feature_builder import FeatureBuilder
from feast.infra.registry.base_registry import BaseRegistry


class RisingWaveFeatureBuilder(FeatureBuilder):
    def __init__(
        self,
        registry: BaseRegistry,
        feature_view,
        task: Union[MaterializationTask, HistoricalRetrievalTask],
        *,
        source_is_retractable: bool = False,
        emit_on_close: bool = False,
    ):
        super().__init__(registry, feature_view, task)
        self.source_is_retractable = source_is_retractable
        self.emit_on_close = emit_on_close

    def _should_join_entity_df(self) -> bool:
        return isinstance(self.task, HistoricalRetrievalTask) and (
            isinstance(self.task.entity_df, pd.DataFrame)
            or (
                isinstance(self.task.entity_df, str)
                and bool(self.task.entity_df.strip())
            )
        )

    def _build(self, view: Any, input_nodes: Optional[List[DAGNode]]) -> DAGNode:
        # Node ordering DIVERGES from Flink (flink/feature_builder.py:49-77) on the
        # aggregated point-in-time-join path, and it must: Flink rejects time windows
        # (flink/nodes.py:536-542), so its filter-then-aggregate order never produces a
        # windowed relation to leak. RisingWave's value-add IS time windows, so the
        # as-of cut has to be expressed on window_end AFTER aggregation, not on the raw
        # event ts before it.
        #
        # Two orderings:
        #   * Aggregated PIT retrieval (entity_df join + aggregations):
        #       source -> (transform) -> aggregate(full windows, EMIT ON WINDOW CLOSE)
        #       -> join(attach spine: ENTITY_TS_ALIAS + ENTITY_ROW_ID)
        #       -> filter(window_end <= ENTITY_TS_ALIAS [+ ttl lower bound on window_end])
        #       -> dedup(latest CLOSED window per ENTITY_ROW_ID, ORDER BY window_end DESC)
        #     Aggregating BEFORE the cut keeps full windows (online/offline parity); the
        #     post-aggregation cut on window_end never admits an open/partial window, and
        #     the dedup collapses the many closed windows per label row to the as-of one.
        #   * Everything else (materialize backfill, non-aggregated retrieval):
        #       source -> (transform) -> [join] -> filter -> aggregate|dedup
        #     The backfill filter is a bounded [start, end) predicate (NOT a PIT cut), so
        #     filter-then-aggregate computes full windows within the range — matching the
        #     live MV. Non-aggregated retrieval cuts the raw ts then dedups, as before.
        aggregated_pit = self._should_aggregate(view) and self._should_join_entity_df()

        if view.data_source:
            last_node: DAGNode = self.build_source_node(view)

            if self._should_transform(view):
                last_node = self.build_transformation_node(view, [last_node])

            if aggregated_pit:
                # Pre-aggregation: apply the raw-row view.filter ONLY (no PIT/TTL cut on
                # the raw ts — that is the leak being fixed). Then aggregate the FULL
                # stream, join the spine, and cut on window_end.
                last_node = self.build_filter_node(
                    view,
                    last_node,
                    name_suffix="prefilter",
                    include_filter_expr=True,
                    include_pit_cut=False,
                )
                last_node = self.build_aggregation_node(view, last_node)
                last_node = self.build_join_node(view, [last_node])
                # Post-aggregation: the as-of cut on window_end (+ TTL lower bound on
                # window_end). No raw-row view.filter here — its columns are gone.
                last_node = self.build_filter_node(
                    view,
                    last_node,
                    name_suffix="filter",
                    include_filter_expr=False,
                    include_pit_cut=True,
                )
                last_node = self.build_dedup_node(view, last_node)
                if self._should_validate(view):
                    last_node = self.build_validation_node(view, last_node)
                return last_node

            if self._should_join_entity_df():
                last_node = self.build_join_node(view, [last_node])

        elif input_nodes:
            if self._should_transform(view):
                last_node = self.build_transformation_node(view, input_nodes)
            else:
                last_node = self.build_join_node(view, input_nodes)
        else:
            raise ValueError(f"FeatureView {view.name} has no valid source or inputs")

        last_node = self.build_filter_node(view, last_node)

        if self._should_aggregate(view):
            last_node = self.build_aggregation_node(view, last_node)
        elif self._should_dedupe(view):
            last_node = self.build_dedup_node(view, last_node)

        if self._should_validate(view):
            last_node = self.build_validation_node(view, last_node)

        return last_node

    def build_source_node(self, view):
        node = RWSourceNode(f"{view.name}:source", view, self.get_column_info(view))
        self.nodes.append(node)
        return node

    def build_filter_node(
        self,
        view,
        input_node,
        *,
        name_suffix: str = "filter",
        include_filter_expr: bool = True,
        include_pit_cut: bool = True,
    ):
        # ``include_pit_cut`` gates the point-in-time (``ts <= ENTITY_TS_ALIAS``) +
        # TTL predicates; ``include_filter_expr`` gates the row-level ``view.filter``.
        # On the aggregated-PIT path the raw-row ``view.filter`` is applied BEFORE
        # aggregation (pre-agg filter: filter_expr only) and the PIT/TTL cut AFTER it
        # (post-agg filter: pit_cut only, on window_end) — see ``_build``.
        node = RWFilterNode(
            f"{view.name}:{name_suffix}",
            view,
            self.get_column_info(view),
            filter_expr=getattr(view, "filter", None) if include_filter_expr else None,
            ttl=getattr(view, "ttl", None) if include_pit_cut else None,
            inputs=[input_node],
            include_pit_cut=include_pit_cut,
        )
        self.nodes.append(node)
        return node

    def build_aggregation_node(self, view, input_node):
        node = RWAggregationNode(
            f"{view.name}:agg",
            view,
            self.get_column_info(view),
            source_is_retractable=self.source_is_retractable,
            emit_on_close=self.emit_on_close,
            inputs=[input_node],
        )
        self.nodes.append(node)
        return node

    def build_dedup_node(self, view, input_node):
        node = RWDedupNode(
            f"{view.name}:dedup", view, self.get_column_info(view), inputs=[input_node]
        )
        self.nodes.append(node)
        return node

    def build_join_node(self, view, input_nodes):
        node = RWJoinNode(
            f"{view.name}:join", view, self.get_column_info(view), inputs=input_nodes
        )
        self.nodes.append(node)
        return node

    def build_transformation_node(self, view, input_nodes):
        node = RWTransformNode(
            f"{view.name}:transform",
            view,
            self.get_column_info(view),
            inputs=input_nodes,
        )
        self.nodes.append(node)
        return node

    def build_validation_node(self, view, input_node):
        expected_columns = (
            [f.name for f in view.features] if getattr(view, "features", None) else []
        )
        node = RWValidationNode(
            f"{view.name}:validate",
            view,
            self.get_column_info(view),
            expected_columns=expected_columns,
            inputs=[input_node],
        )
        self.nodes.append(node)
        return node

    def build_output_nodes(self, view, final_node):
        node = RWOutputNode(
            f"{view.name}:output",
            self.dag_root.view,
            self.get_column_info(self.dag_root.view),
            write_output=isinstance(self.task, MaterializationTask),
            inputs=[final_node],
        )
        self.nodes.append(node)
        return node

    def build(self) -> ExecutionPlan:
        plan = super().build()
        return RisingWaveExecutionPlan(plan.nodes)
