"""Adversarial unit tests for the RisingWave compute engine.

These tests do NOT cover the happy path. Each one tries to make the engine emit an
incorrect, leaky, or unsafe artifact and asserts that it refuses or produces the
safe form. They encode the 5 blockers + correctness invariants found in the design
review and pin what the de-risking spike must keep green.

They run without a live RisingWave: the SQL builders and the provisioning guards are
pure (no DB connection).
"""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from feast.aggregation import Aggregation
from feast.data_format import JsonFormat
from feast.data_source import KafkaSource, PushSource
from feast.infra.compute_engines.risingwave.engine import (
    RisingWaveComputeEngine,
    _iceberg_sink_ddl,
)
from feast.infra.compute_engines.risingwave.offline_store import (
    RisingWaveOfflineStore,
    RisingWaveOfflineStoreConfig,
    _entity_df_to_sql,
)
from feast.infra.compute_engines.risingwave.nodes import (
    RWFilterNode,
    RWJoinNode,
    build_windowed_agg_select,
)
from feast.infra.compute_engines.dag.context import ColumnInfo, ExecutionContext
from feast.infra.compute_engines.dag.model import DAGFormat
from feast.infra.compute_engines.dag.node import DAGNode
from feast.infra.compute_engines.dag.value import DAGValue
from feast.infra.compute_engines.utils import ENTITY_ROW_ID, ENTITY_TS_ALIAS
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
    EntitySelectMode,
)


def _column_info(feature_cols=("amount_sum_3600s",)):
    return ColumnInfo(
        join_keys=["user_id"],
        feature_cols=list(feature_cols),
        ts_col="event_ts",
        created_ts_col=None,
        field_mapping=None,
    )


def _agg(function, window_seconds=3600, column="amount"):
    return Aggregation(
        column=column,
        function=function,
        time_window=timedelta(seconds=window_seconds),
    )


def _kafka_source(watermark=True):
    return KafkaSource(
        name="txn_stream",
        timestamp_field="event_ts",
        message_format=JsonFormat(schema_json=""),
        kafka_bootstrap_servers="localhost:9092",
        topic="txn",
        watermark_delay_threshold=timedelta(seconds=30) if watermark else None,
    )


def _stream_view(source, aggs, offline=True):
    return SimpleNamespace(
        name="user_txn",
        stream_source=source,
        aggregations=list(aggs),
        entity_columns=[SimpleNamespace(name="user_id", dtype="String")],
        features=[SimpleNamespace(name=a.resolved_name(a.time_window)) for a in aggs],
        offline=offline,
    )


def _engine(emit_on_window_close=True):
    engine = RisingWaveComputeEngine.__new__(RisingWaveComputeEngine)
    engine.config = SimpleNamespace(
        emit_on_window_close=emit_on_window_close,
        catalog_name="feast",
        catalog_type="storage",
        warehouse_path="s3a://feast/wh",
        iceberg_database="feast",
        s3_endpoint=None,
        s3_region=None,
        s3_access_key=None,
        s3_secret_key=None,
    )
    return engine


# --- PIT history: window_end timestamping, append-only / composite-PK retention ---


def test_offline_sink_timestamps_by_window_end_not_window_start():
    sql = _iceberg_sink_ddl("p_v_offline", "p_v_online", _column_info(), _engine().config)
    assert '"window_end" AS event_timestamp' in sql
    # window_start would expose a still-open window to inclusive as-of joins.
    assert "window_start" not in sql


def test_offline_sink_defaults_to_append_only_history():
    sql = _iceberg_sink_ddl("p_v_offline", "p_v_online", _column_info(), _engine().config)
    assert "type='append-only'" in sql
    assert "force_append_only='true'" in sql


def test_offline_sink_upsert_uses_composite_pk_never_entity_only():
    sql = _iceberg_sink_ddl(
        "p_v_offline", "p_v_online", _column_info(), _engine().config, upsert=True
    )
    assert "type='upsert'" in sql
    # entity-only PK would collapse history and leak the latest value to every label.
    assert "primary_key='user_id, window_end'" in sql


# --- Retraction: monoid guard over a retractable source ---


@pytest.mark.parametrize("function", ["min", "max", "count_distinct"])
def test_monoid_aggregation_over_retractable_source_is_rejected(function):
    with pytest.raises(ValueError, match="monoid"):
        build_windowed_agg_select(
            _column_info(),
            [_agg(function)],
            "src",
            source_is_retractable=True,
            emit_on_close=False,
        )


@pytest.mark.parametrize("function", ["sum", "count", "mean"])
def test_abelian_aggregation_over_retractable_source_is_allowed(function):
    sql = build_windowed_agg_select(
        _column_info(),
        [_agg(function)],
        "src",
        source_is_retractable=True,
        emit_on_close=False,
    )
    assert "tumble(" in sql


def test_monoid_aggregation_over_append_only_source_is_allowed():
    sql = build_windowed_agg_select(
        _column_info(),
        [_agg("max")],
        "src",
        source_is_retractable=False,
        emit_on_close=False,
    )
    assert "max(amount)" in sql


# --- Window semantics ---


def test_windowed_select_groups_by_and_emits_window_end():
    sql = build_windowed_agg_select(
        _column_info(),
        [_agg("sum")],
        "src",
        source_is_retractable=False,
        emit_on_close=False,
    )
    assert "GROUP BY window_start, window_end" in sql


def test_mixed_windows_in_one_view_are_rejected():
    aggs = [_agg("sum", 3600), _agg("count", 86400)]
    with pytest.raises(ValueError, match="single"):
        build_windowed_agg_select(
            _column_info(),
            aggs,
            "src",
            source_is_retractable=False,
            emit_on_close=False,
        )


def test_emit_on_window_close_is_appended_only_when_requested():
    base = build_windowed_agg_select(
        _column_info(), [_agg("sum")], "src",
        source_is_retractable=False, emit_on_close=False,
    )
    eowc = build_windowed_agg_select(
        _column_info(), [_agg("sum")], "src",
        source_is_retractable=False, emit_on_close=True,
    )
    assert "EMIT ON WINDOW CLOSE" not in base
    assert eowc.endswith("EMIT ON WINDOW CLOSE")


# --- Provisioning guards ---


def test_pushsource_stream_view_is_rejected():
    # isinstance check fires before any attribute access, so __new__ is enough.
    push = PushSource.__new__(PushSource)
    view = _stream_view(push, [_agg("sum")])
    with pytest.raises(ValueError, match="PushSource"):
        _engine()._provision_ddl("proj", view)


def test_emit_on_window_close_requires_a_source_watermark():
    view = _stream_view(_kafka_source(watermark=False), [_agg("sum")])
    with pytest.raises(ValueError, match="watermark"):
        _engine(emit_on_window_close=True)._provision_ddl("proj", view)


def test_provision_emits_source_mv_and_iceberg_sink():
    view = _stream_view(_kafka_source(watermark=True), [_agg("sum")])
    source_sql, mv_sql, sink_sql = _engine(emit_on_window_close=True)._provision_ddl(
        "proj", view
    )
    assert source_sql.startswith("CREATE SOURCE")
    assert "WATERMARK FOR" in source_sql
    assert mv_sql.startswith("CREATE MATERIALIZED VIEW")
    assert mv_sql.endswith("EMIT ON WINDOW CLOSE")
    assert sink_sql.startswith("CREATE SINK")
    assert "connector='iceberg'" in sink_sql
    assert '"window_end" AS event_timestamp' in sink_sql


# (Removed: engine.get_historical_features tests — retrieval is the offline store's
#  job now, exercised by the RisingWaveOfflineStore tests at the bottom of this file.)


# --- New surface helpers: drive the pure SQL-builder DAG nodes without a live DB ---


class _StubInputNode(DAGNode):
    """A single upstream node whose output DAGValue is pre-seeded into the context.

    The RisingWave nodes pull their input via ``get_single_input_value`` keyed by the
    input node's name, so we register one stub and stash its DAGValue in
    ``context.node_outputs``.
    """

    def __init__(self, name, value: DAGValue):
        super().__init__(name)
        self._value = value

    def execute(self, context):  # pragma: no cover - never executed in these tests
        return self._value


def _rw_value(relation: str, columns, *, metadata=None) -> DAGValue:
    return DAGValue(
        data=relation,
        format=DAGFormat.RISINGWAVE,
        metadata={**(metadata or {}), "columns": list(columns)},
    )


def _context_with_input(input_node: _StubInputNode, *, entity_df=None) -> ExecutionContext:
    """Minimal ExecutionContext carrying only what the nodes read.

    Built via ``__new__`` so we do not need a real RepoConfig / OfflineStore — the
    join/filter nodes only touch ``project``, ``entity_df`` and ``node_outputs``.
    """
    context = ExecutionContext.__new__(ExecutionContext)
    context.project = "proj"
    context.entity_df = entity_df
    context.node_outputs = {input_node.name: input_node._value}
    return context


# --- (b) RWFilterNode applies an INCLUSIVE PIT cut (ts <= entity ts), not strict < ---


def _filter_column_info():
    return ColumnInfo(
        join_keys=["user_id"],
        feature_cols=["amount_sum_3600s"],
        ts_col="window_end",
        created_ts_col=None,
        field_mapping=None,
    )


def test_filter_pit_cut_is_inclusive_on_window_end_not_strict_or_window_start():
    # Feature relation already aggregated: window_end is the effective event timestamp,
    # carried via metadata. The PIT cut must be `window_end <= __entity_event_timestamp`.
    columns = ["user_id", "amount_sum_3600s", "window_end", ENTITY_TS_ALIAS]
    upstream = _StubInputNode(
        "agg",
        _rw_value(
            "(SELECT ...)",
            columns,
            metadata={"event_timestamp_column": "window_end", "aggregated": True},
        ),
    )
    node = RWFilterNode(
        "filter", SimpleNamespace(), _filter_column_info(), inputs=[upstream]
    )
    out = node.execute(_context_with_input(upstream))
    sql = out.data
    assert f'"window_end" <= "{ENTITY_TS_ALIAS}"' in sql
    # Inclusive only: a strict cut would drop the row whose window closes exactly at the
    # label time. Strip the inclusive operator and assert no bare strict `<` remains.
    assert "<" not in sql.replace("<=", "")
    # window_start would admit a still-open window before it closes.
    assert "window_start" not in sql


def test_filter_pit_cut_is_skipped_when_disabled_even_with_entity_ts_present():
    # The pre-aggregation filter on the aggregated-PIT path must NOT emit a raw-ts cut
    # (that would leak partial/future-dated windows); include_pit_cut=False suppresses it.
    columns = ["user_id", "event_ts", ENTITY_TS_ALIAS]
    ci = ColumnInfo(
        join_keys=["user_id"],
        feature_cols=["amount"],
        ts_col="event_ts",
        created_ts_col=None,
        field_mapping=None,
    )
    upstream = _StubInputNode("src", _rw_value("rel", columns))
    node = RWFilterNode(
        "filter", SimpleNamespace(), ci, inputs=[upstream], include_pit_cut=False
    )
    out = node.execute(_context_with_input(upstream))
    # No predicate at all -> the node returns its input unchanged (no WHERE emitted).
    assert ENTITY_TS_ALIAS not in out.data
    assert "WHERE" not in out.data


def test_filter_ttl_lower_bound_is_inclusive_and_anchored_on_entity_ts():
    columns = ["user_id", "window_end", ENTITY_TS_ALIAS]
    upstream = _StubInputNode(
        "agg",
        _rw_value(
            "(SELECT ...)",
            columns,
            metadata={"event_timestamp_column": "window_end"},
        ),
    )
    node = RWFilterNode(
        "filter",
        SimpleNamespace(),
        _filter_column_info(),
        inputs=[upstream],
        ttl=timedelta(hours=1),
    )
    sql = node.execute(_context_with_input(upstream)).data
    assert f'"window_end" <= "{ENTITY_TS_ALIAS}"' in sql
    assert f'"window_end" >= "{ENTITY_TS_ALIAS}" - INTERVAL \'3600\' SECOND' in sql


# --- (c) RWJoinNode produces a LEFT JOIN on the join keys (entity spine on the left) ---


def test_entity_spine_join_is_left_join_on_join_keys():
    entity_df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    feature_columns = ["user_id", "amount_sum_3600s", "window_end"]
    upstream = _StubInputNode(
        "features",
        _rw_value(
            "(SELECT user_id, amount_sum_3600s, window_end FROM agg)",
            feature_columns,
            metadata={"event_timestamp_column": "window_end"},
        ),
    )
    node = RWJoinNode(
        "join", SimpleNamespace(), _filter_column_info(), inputs=[upstream]
    )
    out = node.execute(_context_with_input(upstream, entity_df=entity_df))
    sql = out.data
    # Spine is the LEFT side (alias e), features the RIGHT (alias f) -> a LEFT JOIN so
    # entity rows without a matching feature row are retained (no inner-join row loss).
    assert "LEFT JOIN" in sql
    assert "INNER JOIN" not in sql
    assert sql.count(" JOIN ") == 1
    # ON predicate is keyed on the view join key, qualified e.<key> = f.<key>.
    assert 'e."user_id" = f."user_id"' in sql
    assert out.metadata["join_type"] == "left"
    assert out.metadata["joined_on"] == ["user_id"]


def test_entity_spine_join_left_join_on_composite_join_keys():
    ci = ColumnInfo(
        join_keys=["user_id", "merchant_id"],
        feature_cols=["amount_sum_3600s"],
        ts_col="window_end",
        created_ts_col=None,
        field_mapping=None,
    )
    entity_df = pd.DataFrame(
        {
            "user_id": [1],
            "merchant_id": [9],
            "event_timestamp": pd.to_datetime(["2024-01-01"]),
        }
    )
    feature_columns = ["user_id", "merchant_id", "amount_sum_3600s", "window_end"]
    upstream = _StubInputNode("features", _rw_value("(SELECT ... FROM agg)", feature_columns))
    node = RWJoinNode("join", SimpleNamespace(), ci, inputs=[upstream])
    sql = node.execute(_context_with_input(upstream, entity_df=entity_df)).data
    assert "LEFT JOIN" in sql
    # Both keys ANDed in the ON clause; neither key silently dropped.
    assert 'e."user_id" = f."user_id"' in sql
    assert 'e."merchant_id" = f."merchant_id"' in sql
    assert " AND " in sql


def test_entity_spine_join_rejects_missing_entity_df():
    upstream = _StubInputNode(
        "features", _rw_value("(SELECT ... FROM agg)", ["user_id", "amount_sum_3600s"])
    )
    node = RWJoinNode("join", SimpleNamespace(), _filter_column_info(), inputs=[upstream])
    with pytest.raises(RuntimeError, match="requires an entity_df"):
        node.execute(_context_with_input(upstream, entity_df=None))


# --- RisingWaveOfflineStore: inline the entity_df; never upload a temp table
#     (RisingWave INSERTs are async, so an uploaded entity table is empty at query time)


def test_offline_store_config_defaults_to_embed_query_and_disabled_ssl():
    cfg = RisingWaveOfflineStoreConfig(
        host="localhost", port=4566, database="dev", user="root", password=""
    )
    assert cfg.entity_select_mode == EntitySelectMode.embed_query
    assert cfg.sslmode == "disable"


def test_entity_df_to_sql_is_a_bare_select_with_no_table_upload():
    df = pd.DataFrame(
        {
            "user_id": pd.Series(["u1"], dtype="object"),
            "event_timestamp": pd.to_datetime(["2026-06-18T12:00:00+00:00"]),
        }
    )
    sql = _entity_df_to_sql(df)
    assert sql.startswith("SELECT ")
    assert "CREATE TABLE" not in sql and "INSERT" not in sql  # never uploads
    assert "TIMESTAMPTZ" in sql  # tz-aware label is cast
    assert '"user_id"' in sql and '"event_timestamp"' in sql


def test_get_historical_features_inlines_dataframe_instead_of_uploading():
    # A DataFrame entity_df must be converted to inline SQL before delegating to the
    # parent, so the parent uses its embed_query/CTE path — never the temp-table upload
    # that RisingWave's async INSERTs leave empty.
    from unittest.mock import patch

    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "event_timestamp": pd.to_datetime(["2026-06-18T12:00:00+00:00"]),
        }
    )
    target = (
        "feast.infra.offline_stores.contrib.postgres_offline_store.postgres."
        "PostgreSQLOfflineStore.get_historical_features"
    )
    with patch(target) as parent:
        RisingWaveOfflineStore.get_historical_features(
            config=MagicMock(),
            feature_views=[],
            feature_refs=[],
            entity_df=df,
            registry=MagicMock(),
            project="proj",
        )
    assert parent.called
    passed = parent.call_args.kwargs["entity_df"]
    assert isinstance(passed, str) and passed.startswith("SELECT ")
