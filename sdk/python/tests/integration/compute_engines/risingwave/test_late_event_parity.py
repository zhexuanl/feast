"""ADR-0005 Action 2 — backfill<->streaming late-event parity (live RisingWave).

The MOAT: a late event (event_ts < watermark) is dropped at ingestion by the watermarked source, so
the EOWC MV never sees it; training (get_historical_features) must read that same gated MV and return
the SAME value online served — never a raw, watermark-ungated relation that resurrects the late event.

This pins it through the REAL offline path and avoids the four ways such a test passes vacuously
(ADR-0005 §"The late-event parity test"):
  * runs `FeatureStore.get_historical_features` end-to-end (the Postgres PIT template) — not a
    hand-issued MV SELECT (which would assert MV==MV);
  * `ttl=timedelta(0)` + asserts the value is non-null 100.0 (a non-zero ttl returns NULL, which a
    "!= 600" check would pass vacuously);
  * a negative control routed through the SAME offline store: a second FeatureView whose batch_source
    is a WATERMARK-FREE aggregation returns 600 — proving the watermark gate (not luck) is what keeps
    offline at 100 (a raw re-scan of the *watermarked* source also returns 100, since the row was
    never stored, so only a watermark-free relation yields 600);
  * seeds two windows and queries a label BETWEEN them, exercising window-selection + late-drop together.

Live-guarded: set CP_FS_TEST_DSN to a RisingWave (e.g. postgres://root@localhost:4566/dev). Skips otherwise.
Uses the spike's table variant (a watermarked `CREATE TABLE ... APPEND ONLY` — identical WatermarkFilter,
no Kafka broker needed). Plain FeatureViews over query-based PostgreSQLSources stand in for the stream
view's auto-derived offline source: the invariant under test is the offline store's source, not the FV kind.
"""

import os
import tempfile
import time
from urllib.parse import urlparse

import pandas as pd
import pytest

DSN = os.environ.get("CP_FS_TEST_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="set CP_FS_TEST_DSN (e.g. postgres://root@localhost:4566/dev) to run the live parity test"
)

_PROJECT = "parity"
_MV = "parity_user_txn_online"          # EOWC MV over a watermarked source -> drops late events
_RAW = "parity_user_txn_raw"            # plain table, NO watermark -> retains late events (the leak)
_LABEL = pd.Timestamp("2026-06-18 10:01:30", tz="UTC")  # between W1 close (10:01) and W2 (10:03)


def _exec(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)


def _seed(conn):
    # Watermarked APPEND ONLY source -> EOWC 60s-tumble sum MV (the gated online/offline relation).
    _exec(conn, f"DROP MATERIALIZED VIEW IF EXISTS {_MV}")
    _exec(conn, "DROP TABLE IF EXISTS lt_src")
    _exec(conn, "DROP TABLE IF EXISTS " + _RAW)
    _exec(conn, """
        CREATE TABLE lt_src (user_id varchar, amount double, event_ts timestamptz,
            WATERMARK FOR event_ts AS event_ts - INTERVAL '5' SECOND) APPEND ONLY
    """)
    _exec(conn, f"""
        CREATE MATERIALIZED VIEW {_MV} AS
        SELECT user_id, sum(amount) AS sum_amount_60s, window_start, window_end
        FROM tumble(lt_src, event_ts, INTERVAL '60' SECOND)
        GROUP BY window_start, window_end, user_id EMIT ON WINDOW CLOSE
    """)
    # A watermark-FREE twin that DOES retain every row — the negative control (what raw would leak).
    _exec(conn, f"CREATE TABLE {_RAW} (user_id varchar, amount double, event_ts timestamptz)")
    # W1 [10:00,10:01) on-time, W2 [10:02,10:03), advancer at 10:04 -> watermark 10:03:55 closes W1 & W2.
    rows = "('a',100,'2026-06-18 10:00:10Z'),('a',200,'2026-06-18 10:02:10Z'),('a',1,'2026-06-18 10:04:00Z')"
    _exec(conn, f"INSERT INTO lt_src VALUES {rows}")
    _exec(conn, f"INSERT INTO {_RAW} VALUES {rows}")
    conn.commit()


def _mv_w1_sum(conn):
    with conn.cursor() as cur:
        cur.execute(f"SELECT sum_amount_60s FROM {_MV} "
                    "WHERE user_id='a' AND window_start='2026-06-18 10:00:00Z'")
        row = cur.fetchone()
    return None if row is None else row[0]


def _poll(fn, want, tries=25, delay=0.4):
    """RisingWave INSERTs are async (visible after a checkpoint) — poll until `fn()==want`."""
    last = object()
    for _ in range(tries):
        last = fn()
        if last == want:
            return last
        time.sleep(delay)
    return last


def _store(tmp):
    from feast import FeatureStore, RepoConfig
    u = urlparse(DSN)
    cfg = RepoConfig(
        project=_PROJECT,
        provider="local",
        registry=os.path.join(tmp, "registry.db"),
        online_store=None,
        offline_store={
            "type": "feast.infra.compute_engines.risingwave.offline_store.RisingWaveOfflineStore",
            "host": u.hostname or "localhost", "port": u.port or 4566,
            "database": (u.path or "/dev").lstrip("/") or "dev",
            "user": u.username or "root", "password": u.password or "",
        },
        entity_key_serialization_version=3,
    )
    return FeatureStore(config=cfg)


def _views():
    from feast import Entity, FeatureView, Field
    from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
    from feast.types import Float64
    from feast.value_type import ValueType

    user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.STRING)
    # mv source == exactly what apply._derive_offline_source builds (window_end AS event_timestamp).
    mv_src = PostgreSQLSource(
        name="mv__offline",
        query=f'SELECT "user_id", "sum_amount_60s", "window_end" AS event_timestamp FROM "{_MV}"',
        timestamp_field="event_timestamp",
    )
    # raw source: a WATERMARK-FREE W1 aggregation (includes the late event) -> the leak a raw re-scan
    # would produce. Stamped at W1's window_end so the same label selects it.
    raw_src = PostgreSQLSource(
        name="raw__offline",
        query=(
            f'SELECT "user_id", sum("amount") AS sum_amount_60s, '
            f"TIMESTAMP '2026-06-18 10:01:00Z' AS event_timestamp FROM \"{_RAW}\" "
            "WHERE \"event_ts\" >= '2026-06-18 10:00:00Z' AND \"event_ts\" < '2026-06-18 10:01:00Z' "
            'GROUP BY "user_id"'
        ),
        timestamp_field="event_timestamp",
    )
    common = dict(entities=[user], schema=[Field(name="sum_amount_60s", dtype=Float64)])
    # ttl=0 => no TTL lower bound, so a label minutes past window_end still returns the window (not NULL).
    mv_fv = FeatureView(name="user_txn_mv", source=mv_src, **common)
    raw_fv = FeatureView(name="user_txn_raw", source=raw_src, **common)
    return user, mv_fv, raw_fv


def _ghf(store, ref):
    df = store.get_historical_features(
        entity_df=pd.DataFrame({"user_id": ["a"], "event_timestamp": [_LABEL]}),
        features=[ref],
    ).to_df()
    return df["sum_amount_60s"].iloc[0]


def test_late_event_parity():
    import psycopg

    with tempfile.TemporaryDirectory() as tmp, psycopg.connect(DSN, autocommit=False) as conn:
        _seed(conn)
        # W1 closes (advancer pushed the watermark past 10:01) -> MV sum for W1 == 100.
        assert _poll(lambda: _mv_w1_sum(conn), 100.0) == 100.0, "W1 never closed to 100"

        store = _store(tmp)
        user, mv_fv, raw_fv = _views()
        store.apply([user, mv_fv, raw_fv])  # registry only (online_store=None, no engine provisioning)

        # Pre-late baseline: offline (real PIT over the gated MV) == online == 100, and non-null.
        base = _ghf(store, "user_txn_mv:sum_amount_60s")
        assert base == 100.0, f"offline baseline = {base!r}, want 100.0 (label picks W1 not W2, non-null)"

        # Inject the LATE event into BOTH: dropped by the watermarked source, kept by the raw table.
        _exec(conn, "INSERT INTO lt_src VALUES ('a', 500, '2026-06-18 10:00:50Z')")
        _exec(conn, f"INSERT INTO {_RAW} VALUES ('a', 500, '2026-06-18 10:00:50Z')")
        conn.commit()

        # (a) online unchanged — the MV dropped the late event at ingestion.
        for _ in range(8):
            assert _mv_w1_sum(conn) == 100.0, "online MV changed after a late event (should stay 100)"
            time.sleep(0.25)

        # (b) offline unchanged — parity holds: get_historical_features still returns 100, not 600.
        after = _ghf(store, "user_txn_mv:sum_amount_60s")
        assert after == 100.0, f"offline = {after!r} after late event, want 100.0 (parity broken!)"

        # Negative control through the SAME offline store: the watermark-free source DOES leak 600.
        # This is what proves the gate is load-bearing — and what a regression repointing offline at
        # raw would return. (A raw re-scan of the *watermarked* lt_src would still be 100.)
        leak = _poll(lambda: _ghf(store, "user_txn_raw:sum_amount_60s"), 600.0)
        assert leak == 600.0, f"negative control = {leak!r}, want 600.0 (watermark-free path must leak)"
