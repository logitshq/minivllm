# E049 warm49 faithful prefix diagnostic

Default OFF. This prepares only the original graph route: GDN candidate OFF, fixed MB50/ops50/BFS20/sync0. It runs the unchanged `http_replay.run_recorded_case` with two independent engines, the existing full logit/state/ahead checks, and the original recorded arrivals/schedules/output targets. Recorded token IDs are comparison targets, never model inputs.

The original recording and all max_tokens=512 caps are unchanged. A private iterator boundary occurs only after the original loop has finished all checks, flushed the step-49 ledger, and released current snapshots. Only that exact sentinel is caught. A real mismatch, timeout or another exception remains FAIL. The result says `DIAGNOSTIC_PREFIX_COMPLETE` and `INCOMPLETE_BY_DESIGN`, retains actual active requests, and claims no completed requests. The underlying loader's PASS means its registered prefix case and release/hash checks completed; it does not mean the full recording drained.

Warm D06 diverged from screen at output index 46. Stop49 includes D06 output index48. Every co-batched request and actual history through that step remains present. Screen565 is not in this runnable plan and needs a later decision.

Preparation (CPU metadata only):

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/c8_prefix_replay_v1/prefix.py --prepare \
  --child .artifacts/m5-ultra-256gb/c8-schedule-v1-screen/trials/c8-mb50-two-waves \
  --analysis .artifacts/m5-ultra-256gb/c8-schedule-v1-analysis/analysis.json \
  --out bench/m5_ultra_256gb/c8_prefix_replay_v1/prereg-v1
```

Root supplies the existing bounded device owner, exact four MLX environment variables and current service lease. The inner command is `.venv/bin/python -I -B bench/m5_ultra_256gb/c8_prefix_replay_v1/prefix.py --run --plan <absolute plan.json> --plan-sha256 <printed SHA> --out <new child directory>`. It starts no service/subprocess and performs no service restoration itself.

Do not relax the existing 64 GiB child / 240 GiB system / no swap-growth limits. The prior full 16-request E024 replay took 512.49 s and reported 68.55 GiB MLX peak (allocator, not physical footprint). Warm49 should do much less decode work, but its peak is unknown because two independent model weights, prefill intermediates and full-state observers still coexist. The next run must report actual peak and stop on existing resource guards.

For a later separately pinned E050 adapter, `integration_module(q)` is the single injection seam. E049 resolves it only to the old `route_decode_graph/integration.py`; no candidate configuration is accepted by this plan. A future hook must retain the expected `install(runner, enabled=True)` handle with `.decoder` and `.restore()`, and must separately bind its source and numerical scope. No E050 source is loaded here.
