# E055 proposal: C8 batched GDN, observers OFF

Source implemented and CPU checked; no actual preparation, model execution or performance result. Begin only after the actual E052 v2 capture, E053 replay independent audit and saved resource/restoration checks pass. This is one C8 comparison, not a Pareto curve or a comparison against freshly measured oMLX/Lithos.

## Smallest experiment

Both arms use the **same frozen E052 server** (`c8_batched_http_v1/server.py`). It already supports `--run --phase formal`, forbids `--qualification-trace` and `--route-events`, and independently accepts GDN OFF/ON with graph ON. No runtime, kernel or scheduler change is needed.

| Setting | Baseline | Candidate |
|---|---|---|
| `--gdn-prework` | `off` | `on` (B1..8 candidate) |
| `--route-decode-graph` | `on` | `on` |
| HTTP concurrency / max sequences | 8 | 8 |
| Context / token budget / allocator cache | 2048 / 2048 / 2 GiB | identical |
| Compile / fuse / gates / norm reuse | ON / ON / ON / OFF | identical |
| MLX ops / MB / BFS / fast synch | 50 / 50 / 20 / 0 | identical |
| Model, NVFP4 bytes, tokenizer, greedy/EOS rules | original pinned Qwen3.8-27B snapshot | identical |

Use six fresh processes, three adjacent pairs in the existing C8 order: `r1-on, r1-off, r2-off, r2-on, r3-on, r3-off`. Each process sends the unchanged **8 warm + 64 formal** queue at C8: 432 total requests, of which 384 are timed. Each formal queue repeats the same eight fixtures eight times, with cap512 and normal EOS. Pair labels differ; request bodies/order and repeat-specific IDs remain exactly those of the corresponding original C8 specification. No cache flush between requests; retain the server's once-at-load synchronize/clear policy. Never run the arms concurrently or hot-switch one model.

Keep the existing physical/process/swap/power/thermal guards and same-PID pause/restore owner. The current generic runner has 64 GiB child and 240 GiB system thresholds; do not silently raise them. All six phases need a preregistered absolute deadline; a deadline hit remains an incomplete measurement, not permission to drop a slow arm. Report all processes, including failed attempts.

## Reuse seams and small new code

- `pareto_realistic/native_formal.py`: `async run(spec, out, *, checkpoint, backend=None)`, `System`, `count_contract(spec, phase)`, and `check_group(group, spec, phase, decode_text)` already implement the exact 8/64 queue, full raw SSE/count/text/EOS checks, POST-to-DONE timings and direct-Popen cleanup. Reuse the original `run` code with only a private `validate` dependency; no count-anchor change or new client.
- `gdn_graph_combined_formal/formal.py`: `select_entry`, `prepare`, `validate`, and its thin `run` wrapper are the pattern. Its existing `prepare`/`bind_combined` require the **old** C1+C8 combined-cycle schema and cannot simply accept an E053 receipt. Keep them immutable.
- The new `trial.py` exposes `prepare(base_spec_ref, arm, label, numerical_binding)`, `validate(spec)`, and the same asynchronous `run` API. Start from the source-validated `r{1,2,3}-c8-off.json` under `gdn_graph_combined_formal/prereg-formal-v1/specs`; preserve all jobs/configuration/environment. Change only the new schema/label/provenance, command entry to the frozen E052 server plus its required `--run`, GDN arm, and exact `server.identity_fields(... phase='formal', installed=True, route_observer=False)`. Keep the old count contract distinct from the new arm-specific numerical binding. Return the original result plus `label`, `arm` and a new schema; no producer grants performance qualification.
- `concurrency_config_v2/runner.py::run(plan,out)` already iterates any `plan['specs']` list and checks matching labels, PASS, reap and cleanup. A private `FunctionType` namespace can point `HERE` at the new trial folder. `concurrency_config_v2/execute.py::{bind_runner,bounded_cleanup,run}` retains the existing pause, exact identities and natural-exit restoration. Its literal final scope still describes 8+8: preserve that inherited receipt, append an accurate E055 8+64 summary, and never silently relabel old evidence.
- Independent timing should call `pareto_realistic/audit_native_formal.py::raw_metrics(group)` directly and the original `check_group`; add a small six-phase binding and adjacent-pair reducer. The old combined full audit and `paired_summary` assume twelve C1+C8 processes, so do not invoke them unchanged or copy the whole auditor.

Implemented `trial.py` retains the original native lifecycle and adds source/history binding plus `output-ids.json`. `audit.py` reuses the original raw metric and wire checker, adding only six-phase binding and paired reduction. The trial binding fails closed without actual E053 numerical PASS; the external owner separately requires saved resource/restoration PASS.

## What correctness can and cannot transfer

E053 can establish the exact candidate source/model/environment, full IDs/logits/state and graph/trace behavior for **its complete captured two-wave history**, against the native reference under that same schedule. E051 additionally covers the earlier E047 history. Keep those actual plan/audit hashes attached to the new candidate arm; the OFF arm keeps its route-only reference binding. All observers being OFF is a verified formal-mode configuration, not proof of graph-hit counts.

The 64-request formal queue has new realized arrival schedules and no step observer. E053 therefore does **not** prove every formal state or all future schedules; changing MB threshold, token budget, context, weights, runtime, candidate source or concurrency is outside its scope. Do not claim full-state replay of this formal queue. The original raw count/EOS contract may be reused because request bodies, tokenizer and wire semantics are unchanged.

For every warm and formal request, retain and independently parse the complete raw/committed IDs, all prompt IDs, text, terminal EOS, usage and DONE timestamps. Recompute metrics from raw data. Compare complete ID sequences by `(fixture, occurrence)` within repeats and across paired arms; publish the first mismatch if any, not just token totals. E045 already showed C8 schedule-dependent differences, so a cross-process mismatch alone does not identify a candidate arithmetic bug. Preserve valid timing and report the difference separately; withhold an exact-output/Pareto claim until the bounded correctness interpretation is resolved; do not substitute an E053 output sequence as a universal C8 oracle.

## Decision and reporting

Report each process separately, three paired candidate/baseline throughput ratios and p50/p95 latency ratios, then per-arm medians and min/max across three fresh processes. Do not pool requests as independent replications, multiply prior GDN/graph speedups, or choose the fastest repeat. Throughput is committed output tokens divided by the common first-POST to last-DONE window; latency is per-request POST-to-DONE seconds. C8 is HTTP outstanding concurrency, not a claim that every GPU step had B8.

Proposed predeclared win gate: all three paired throughput ratios >1 and p50 ratios <1; median throughput ratio >=1.03, median p50 ratio <=0.97, and median p95 ratio <=1.02. If the three OFF process rates vary by more than 2% of their median, report the variability and repeat later instead of declaring a stable win. Exact-output checks and independent lifecycle/resource checks remain separate gates. Failure of the win gate retains useful actual measurements and rejects adoption, not numerical correctness.

After a stable C8 gain, tune C1 separately using its already qualified E033 GDN+graph entry and unchanged environment first. The frozen E052 entry explicitly requires max_seqs=8; extending this new batched entry to C1 or C2/C4 needs a separately bound entry/qualification, not an unrecorded flag override. More concurrency points and matched comparator runs are needed before drawing a frontier.

## Prepared interfaces and CPU receipts

`trial.prepare(base_spec_ref, arm, label, numerical_binding)` accepts only `r{1..3}-c8-{off|on}` labels. The binding has exact keys `audit`, `replay_plan` (both path/SHA refs), and `child_directory`. `validate(spec)` reconstructs the permitted derivation; `async run(spec,out,checkpoint=...,backend=None)` returns the original status/reap/cleanup fields plus its label. The owner supplies a six-element specs list and unchanged resource controls.

`audit.py --screen CLOSED_SCREEN --plan REGISTERED_OWNER_PLAN --plan-sha256 SHA --out FRESH_RESULT` independently reparses all six closed trials. `PASS_METRICS_AND_IDS` means valid retained timing/IDs, not an automatic speed or Pareto grant. It exposes `timing_valid`, `same_schedule_numerical_prerequisite`, and `cross_schedule_token_equal` / `cross_schedule_differences` separately; different greedy trajectories do not make timing invalid or infer an arithmetic error.

- `.artifacts/m5-ultra-256gb/c8-batched-formal-cpu-v1`: 6/6 passed (`0b7257`), including strict derivation, source/history binding, original run-code reuse and full ID preservation.
- `.artifacts/m5-ultra-256gb/c8-batched-formal-audit-cpu-v1`: 3/3 passed (`e64a46`), including six-pair/count gates, descriptive token differences, drift rejection and exact agreement with the frozen E052 v2 preregistration identity.
- Every CPU invocation retained source snapshots, argv, stdout/stderr and exit code under a no-MLX/no-subprocess/no-signal hook. No actual prepare or GPU run occurred.

## Frozen source references

| Purpose | Repository path | SHA-256 |
|---|---|---|
| Frozen server | `bench/m5_ultra_256gb/c8_batched_http_v1/server.py` | `a3737f1b364e190e3fe45a1779e1be228c4f434701bf8875dbe18f1d519afc7b` |
| Batched integration | `bench/m5_ultra_256gb/gdn_batched_graph/integration.py` | `6e2e4d4350c36476292f2d0ff2bc7e095561f76a2c233b1a99575027378d50b7` |
| Batched GDN | `bench/m5_ultra_256gb/gdn_prework_batched/candidate.py` | `acd2a23802296b9e420108203cc215c3170cd1eccd9a859e1a787432f98a5353` |
| Native formal lifecycle/counts | `bench/m5_ultra_256gb/pareto_realistic/native_formal.py` | `ec9cdc924f7987ad46eac6820b6264495c95fb182c726545a6e3ce283df7e654` |
| Old paired factory | `bench/m5_ultra_256gb/gdn_graph_combined_formal/formal.py` | `4b567f8ed6090175f3639aa109584ca26c1426165c03de3167c3ebde74891a89` |
| Raw metric recomputation | `bench/m5_ultra_256gb/pareto_realistic/audit_native_formal.py` | `b57e4e9657f3dddb04069290bdfbab4c99b0d6cbfb1427e454d0217d5463c870` |
| Pause runner | `bench/m5_ultra_256gb/concurrency_config_v2/runner.py` | `dfb0b6f07e775249babab6e76c62dbf1ab8012027e3cb46f6c694b8b2b730577` |
| Pause driver | `bench/m5_ultra_256gb/concurrency_config_v2/execute.py` | `34a604c62f6de4eddcff2be53abd2d72716384ee67d13d1bf963869d35695a20` |


## Prepared after actual E053 PASS

Actual prepare13f1f9/session80732→c11d6d produced plan
`fc24144f4c0676ca6de3aa26f93bb6eabb2976d8addbc403ec35e228b484ed4b`.
The actual E053 audit file and saved resource review are bound. Owner labels were
aligned with TRIALS during source review before preparation. The original owner's
legacy scope receipt remains unchanged; separate `scope.json` records actual8/64.
No E055 device run has occurred at this entry.
