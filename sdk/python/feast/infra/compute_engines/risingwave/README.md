# RisingWave compute engine (contrib)

RisingWave is the **real-time computation + online-serving plane**; Feast keeps the
**registry/governance** and the **point-in-time training joins**. A single
`engine.update()` provisions both the online materialized view and the offline
Iceberg sink from one feature definition, so online and offline are computed by the
*same* engine — the structural basis for minimal train/serve skew.

```
KafkaSource ─▶ CREATE SOURCE (+WATERMARK)
                  │
                  ├─▶ CREATE MATERIALIZED VIEW  (windowed agg, EMIT ON WINDOW CLOSE)
                  │       └─ online serving: point lookup over pgwire (RisingWaveOnlineStore)
                  └─▶ CREATE SINK → Iceberg     (append-only, window_end AS event_timestamp)
                          └─ training: RisingWaveOfflineStore PIT-joins this history (get_historical_features)
```

## Status

Three components make up the RisingWave integration:

- **`RisingWaveComputeEngine`** (`engine.py`) — `update()` provisions the source +
  windowed-agg MV + Iceberg sink (registry-free; join keys from `entity_columns`);
  `_materialize_one` is a bounded backfill. It does **not** implement
  `get_historical_features`: Feast routes retrieval to the offline store (never the
  compute engine), so a retrieval method here would be dead code.
- **`RisingWaveOnlineStore`** (`online_store.py`) — low-latency point lookup over the
  MV via pgwire.
- **`RisingWaveOfflineStore`** (`offline_store.py`) — training / point-in-time
  retrieval. Feast's provider calls `offline_store.get_historical_features`, so the PIT
  join lives here. It subclasses the Postgres offline store (RisingWave is
  pgwire-compatible) and reuses its proven PIT-join SQL; the one RisingWave fix is
  inlining the entity DataFrame as SQL — RisingWave INSERTs are async, so the Postgres
  store's entity temp-table upload is empty when the PIT query runs.

**Validated end-to-end on RisingWave v3.0.0 (k3s):** `feast apply` provisions the
MV + Iceberg sink; Kafka events → MV (`sum 150, count 2`); `get_online_features` →
correct typed values; `get_historical_features` → point-in-time-correct training data
(label `12:00` → `150 / 2`).

## Correctness invariants (enforced + tested)

| Invariant | Where | Why |
|---|---|---|
| Offline rows timestamped by `window_end`, never `window_start` | `_iceberg_sink_ddl` | a window is only knowable at close; `window_start` + `<=` join leaks open windows |
| Offline sink append-only, or upsert with composite PK `(entity, window_end)` — never entity-only | `_iceberg_sink_ddl` | entity-only upsert collapses history → every label joins to the latest value |
| Monoid aggs (min/max/count_distinct/…) rejected over a retractable source | `build_windowed_agg_select` | monoids cannot be incrementally retracted (Chronon api.thrift:156-164) |
| All aggs in one view share one window | `build_windowed_agg_select` | RisingWave TUMBLE/HOP is one table function over the relation |
| `EMIT ON WINDOW CLOSE` requires a watermark | `_provision_ddl` | EOWC is only valid with a watermark + append-only source |
| `PushSource` stream views rejected | `_provision_ddl` | too thin to compile to `CREATE SOURCE` |

PIT boundary is **inclusive `<=`** end-to-end (adopting Feast's offline join,
`postgres.py:962`); the Chronon strict-`<` semantics are intentionally **not** used.

## ✅ Validated end-to-end (RisingWave v3.0.0 + MinIO/Iceberg on k3s)

The exact SQL the engine emits was run against a live RisingWave with an Iceberg sink
to MinIO. All of it behaved as designed:

- **agg-MV → Iceberg-sink composition** (previously the untested chain): `CREATE SINK`
  accepts the `EMIT ON WINDOW CLOSE` windowed-agg MV, and the append-only sink
  **retains every window row** as history (not collapsed to one row per entity).
- **Point-in-time correctness**: a label at `10:01:30` joins to `sum=150` (window
  ending `10:01:00`), **not** the `70` from the window ending `10:02:00` — proving the
  `window_end` + inclusive-`<=` invariant blocks the open-window leak.
- **`CREATE INDEX ... INCLUDE` on a materialized view** is supported (covering point
  lookup).
- **online MV point-lookup** returns the latest closed window.
- `CREATE TABLE … APPEND ONLY` + `WATERMARK` + `EMIT ON WINDOW CLOSE` + the
  `catalog.type='storage'` Iceberg sink all behave as emitted.

## ⚠️ Still spike-gated

- Monoid retraction over an **upsert** source — only append-only was validated; the
  `build_windowed_agg_select` guard stays.
- The RisingWave-column → `ValueProto` conversion in `online_store.online_read` — the
  SQL point-lookup works, but the Python proto mapping is untested.
- Bounded `[start, end)` backfill **late-data** parity with the live stream.
- `CREATE SOURCE` raw-column **types** (placeholders) and non-JSON encodings.
- Minor: the Iceberg read-back `CREATE SOURCE` should declare an explicit `*` / column
  list (RisingWave emits a NOTICE otherwise).

## Wiring (`feature_store.yaml`)

No core `repo_config.py` edit — the engine/store are referenced by full module path:

```yaml
batch_engine:
  type: feast.infra.compute_engines.risingwave.engine.RisingWaveComputeEngine
  host: localhost
  port: 4566
  warehouse_path: s3a://feast/iceberg
  emit_on_window_close: true        # consistency over freshness; needs a source watermark

offline_store:                       # PIT training joins, Postgres-wire to RisingWave
  type: postgres
  host: localhost
  port: 4566

online_store:                        # read-only point lookup over the MV
  type: feast.infra.compute_engines.risingwave.online_store.RisingWaveOnlineStore
  host: localhost
  port: 4566
```

The only core change is an additive `DAGFormat.RISINGWAVE` enum member
(`dag/model.py`).

## Tests

Adversarial unit tests (correctness-attacking only, no happy-path smoke):
`sdk/python/tests/unit/infra/compute_engines/risingwave/`. Run:

```bash
uv run pytest sdk/python/tests/unit/infra/compute_engines/risingwave -q
```

End-to-end validation against a live RisingWave + MinIO/Iceberg (k3s / Rancher
Desktop) confirmed the items under "Validated end-to-end" above.
