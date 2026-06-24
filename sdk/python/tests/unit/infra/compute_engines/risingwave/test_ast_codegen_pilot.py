"""Pilot: f-string -> SQLGlot AST codegen for ONE builder, behind a semantic-equivalence gate.

Demonstrates the pattern for migrating ``sql_builders.py`` off f-strings: build the SQL structurally via the
SQLGlot builder API + the ``RisingWaveStreaming`` dialect, and GATE it against the existing f-string builder by
asserting both render to the SAME canonical RisingWave SQL (``parse(fstring) == ast``). ``build_latest_row_select``
is the pilot target — pure relational (a subquery + a ``ROW_NUMBER()`` window + ``WHERE rn = 1``), no streaming
constructs — so it isolates the codegen pattern from the RisingWave-specific nodes.

This is a ZERO-CHURN pilot: production still calls the f-string ``build_latest_row_select``, and its SQL goldens
are untouched. When a rollout is approved, ``_latest_row_ast`` graduates into ``sql_builders.py`` and that one
builder's goldens re-baseline to the canonical (upper-case) form the dialect renders.

Lesson the gate caught on the first try: a naive ``exp.Ordered(desc=True)`` renders ``DESC NULLS LAST``, but the
engine's bare ``DESC`` is RisingWave's DEFAULT null ordering (``NULLS FIRST``) — a SEMANTIC divergence (a NULL
timestamp would sort to the opposite end, changing which row is "latest"). The AST must set
``nulls_first=True`` to match. Exactly the kind of silent change the equivalence gate exists to block, and the
reason a swap must be gated rather than eyeballed.
"""

import pytest
import sqlglot
from sqlglot import column, exp, select

from feast.infra.compute_engines.dag.context import ColumnInfo
from feast.infra.compute_engines.risingwave.sql_builders import (
    DEDUP_ROW_NUMBER,
    build_latest_row_select,
)
from feast.infra.compute_engines.risingwave.sqlglot_dialect import RisingWaveStreaming


def _latest_row_ast(column_info: ColumnInfo, relation: str) -> str:
    """The AST-codegen equivalent of ``build_latest_row_select`` — the relational structure (subquery, window,
    projection, filter) composed via the SQLGlot builder API instead of f-string concatenation, rendered
    through the RisingWave dialect."""
    keys = column_info.join_keys_columns
    ts = column_info.timestamp_column
    order_cols = [c for c in (ts, column_info.created_timestamp_column) if c]
    projection, seen = [], set()
    for col in [*keys, *column_info.feature_cols, ts]:
        if col not in seen:
            projection.append(col)
            seen.add(col)
    row_number = exp.Window(
        this=exp.RowNumber(),
        partition_by=[column(k) for k in keys],
        # nulls_first=True -> bare DESC (RisingWave's default); desc=True alone emits the semantically
        # different DESC NULLS LAST.
        order=exp.Order(
            expressions=[exp.Ordered(this=column(c), desc=True, nulls_first=True) for c in order_cols]
        ),
    )
    inner = select(*[column(c) for c in projection], row_number.as_(DEDUP_ROW_NUMBER)).from_(relation)
    outer = (
        select(*[column(c) for c in projection])
        .from_(inner.subquery("_ranked"))
        .where(column(DEDUP_ROW_NUMBER).eq(1))
    )
    return outer.sql(dialect=RisingWaveStreaming, normalize_functions=False)


def _canonical(sql: str) -> str:
    return sqlglot.parse_one(sql, read=RisingWaveStreaming).sql(dialect=RisingWaveStreaming, normalize_functions=False)


_CASES = {
    "single_key": ColumnInfo(join_keys=["user_id"], feature_cols=["amount"], ts_col="event_ts",
                             created_ts_col=None, field_mapping=None),
    "created_ts_tiebreak": ColumnInfo(join_keys=["user_id"], feature_cols=["amount", "merchant"],
                                      ts_col="event_ts", created_ts_col="created_ts", field_mapping=None),
    "composite_key": ColumnInfo(join_keys=["acct", "region"], feature_cols=["amt"], ts_col="ts",
                                created_ts_col=None, field_mapping=None),
}


@pytest.mark.parametrize("label", sorted(_CASES))
def test_ast_codegen_matches_fstring_canonical(label):
    # The gate: the AST builder and the existing f-string builder render to the SAME canonical RisingWave SQL,
    # so the AST is a semantically-faithful drop-in (the rollout would re-baseline only the case/spacing).
    ci = _CASES[label]
    assert _latest_row_ast(ci, "src") == _canonical(build_latest_row_select(ci, "src"))
