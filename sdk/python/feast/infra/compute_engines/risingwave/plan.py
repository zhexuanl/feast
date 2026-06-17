"""RisingWaveExecutionPlan — composes the DAG's node SQL into a single statement.

The base ``ExecutionPlan.to_sql()`` is a ``NotImplementedError`` placeholder
(plan.py:56-61); RisingWave is the first SQL-emitting engine, so we provide it here.
Nodes are executed eagerly (each caches its ``DAGValue``); the terminal output node
carries the final ``sql`` in its metadata — the final SELECT for a retrieval task, or
the ``INSERT ... SELECT`` for a materialize task.
"""

from feast.infra.compute_engines.dag.context import ExecutionContext
from feast.infra.compute_engines.dag.plan import ExecutionPlan


class RisingWaveExecutionPlan(ExecutionPlan):
    def to_sql(self, context: ExecutionContext) -> str:
        terminal = self.execute(context)
        sql = terminal.metadata.get("sql") if terminal.metadata else None
        if sql is None:
            raise ValueError(
                "Terminal RisingWave node produced no SQL. For a materialize task with "
                "no offline output, online serving is handled by the materialized view "
                "provisioned in update() and there is nothing to compose here."
            )
        return sql
