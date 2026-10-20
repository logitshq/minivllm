# E050: batched GDN + graph on the original warm49 prefix

Source prepared; device **NOT RUN**. The guarded CPU contracts passed **10/10** in `.artifacts/m5-ultra-256gb/c8-batched-prefix-cpu-v2` (no backend/process/socket attempts, source bytes unchanged). The v1 attempt is retained: the guard replaced `socket.socket` before `unittest.mock` imported SSL, causing an import error before any tests. V2 preloaded that standard-library module before applying the same guard; no production or test source changed. An independent source-only review found no blocker. This tests the exact same incomplete 49-step history as E049 against its original eager oracle, while installing the separately pinned batched GDN + graph composition on the actual candidate engine. The original recording, 512-token request caps, admission order, route witness, comparison logic, boundary, and teardown remain unchanged. No request is declared complete and this is no throughput, full-history, HTTP, or Pareto result.

`prefix.py` reuses E049's `run` code object with an isolated globals dictionary. The only substitutions are the integration hook, E050 validation, and descriptive scope. Its inherited output keeps its original schema under `inherited/`; the new top-level E050 result binds that output and never represents it as the prior E049 experiment. `inherited/base/steps.jsonl` is the unchanged comparison ledger.

`bridge.py` wraps only the private qualification module's `adapted` context. An outer observer constructor binds the actual engine before the frozen route observer calls `install(runner, enabled=True)`. The bridge verifies runner identity and calls the original combined installer once with both switches on. No synthetic engine or requests dictionary is used. All 48 actual GDN handles are installed before graph creation or tracing. The returned facade exposes the existing decoder and delegates restore to the original combined scope after the original observers and route witness have been removed.

`composition.jsonl` and top-level `composition` retain actual engine/runner/layer/handle/wrapper IDs, empty initial graph caches, all 48 initial and final branch counters, and restoration status. Counters count **Python candidate/fallback branch execution, including traces**; they do not count GPU dispatches or completed device calls. Every layer must have a positive candidate count and zero fallback for E050 success. Counter qualification occurs after cleanup, so a counter failure cannot suppress restoration.

Preparation requires the actual successful E049 prefix and its exact plan. It binds the E049 complete, inner complete, raw comparison ledger, and raw route-prefix events, plus the original source closure and new adapter/composition sources. The outer owner additionally binds E049's independent audit and restored service lifecycle. It remains responsible for pause/restore, physical monitoring, and the terminal hold handshake. This module cannot stop or start services.

From the repository, after CPU validation, E049 independent audit, and all source freeze:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/c8_batched_prefix_v1/prefix.py --prepare \
  --base-plan bench/m5_ultra_256gb/c8_prefix_replay_v1/prereg-v1/child-plan.json \
  --predecessor .artifacts/m5-ultra-256gb/c8-prefix-v1-numerical/complete.json \
  --out bench/m5_ultra_256gb/c8_batched_prefix_v1/child-prereg-v1
```

The owner invokes the same fixed argument layout as E049:

```text
.venv/bin/python -I -B <absolute prefix.py> --run --plan <absolute plan.json> --plan-sha256 <sha256> --out <fresh absolute output>
```

The exact MLX environment is inherited from the original plan: ops50, MB50, BFS20, fast-sync0. The new complete schema is `c8-batched-gdn-graph-prefix-result-v1`, with status `DIAGNOSTIC_PREFIX_COMPLETE`, `FAIL`, or initially `INCOMPLETE`. It retains `pid`, `ppid`, new `plan_sha256`, `base_result`, `prefix`, `coverage`, `route_prefix`, and scope booleans, adding `inherited_result`, `composition`, and `composition_events`. Only actual E050 execution can supply numerical evidence for this composition.
