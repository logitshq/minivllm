# E053 actual candidate HTTP history replay

Source is prepared; the final 11 CPU contracts passed under backend, process,
signal and socket denial (`forbidden=[]`, source bytes unchanged). Device replay
has not run. Default invocation describes the experiment and loads no model or
service.

All actual CPU attempts are retained under `.artifacts/m5-ultra-256gb/`:
`c8-batched-http-replay-cpu-v1` ran 10 tests with 9 passes and one synthetic
`/var` versus `/private` tokenizer-path mismatch. Only the fixture was fixed;
the production exact-path gate remained unchanged. `...-cpu-v2` passed all 10;
`...-cpu-v3` added and passed the source-closure contract, 11/11 total. No actual
capture preparation, GPU, model load or service action was performed by these
checks.

After E052 v1 failed readiness, the unexecuted replay source was rebound to the
new `c8_batched_http_v2/trial.py` and explicit v2 capture/join schemas. Actual
`c8-batched-http-replay-cpu-v4` passed all 11 contracts again (tool `2c90a2`),
including rejection of the old v1 capture terminal. Attempts v1–v3 retain their
original source snapshots unchanged; no v1 input is relabelled successful.

`qualify.prepare(capture_child, spec_sha256)` accepts only the closed E052
`c8-batched-http-result-v2` capture, exact server/trial source closure and its
already saved `joined-input`. E052's validator rechecks the actual E051 full
numerical predecessor; E051 does not qualify this new HTTP history. The frozen v1 server remains the actual runtime; the trial is explicitly v2,
which removes the stale E047 diagnostic identity key. The outer
owner separately checks E052 resource observations and service restoration.

Preparation checks closed server/port receipts and the complete raw artifact
inventory, reconstructs the two drained eight-request groups, and runs the
original `http_join.join` against received SSE bytes and the actual schedule.
It checks the full prompts through the pinned tokenizer and compares the result
with the retained recording. No request or clock is fabricated. Steps, route
calls and raw/committed output counts come from these actual inputs; early EOS
is preserved. They are not fixed to the old 1030 steps, 1028 calls or 8192 tokens.

The device path keeps the original complete numerical replay, eager reference,
full logits and 64-layer state comparisons, ahead consumption, admission and
final request drain. It installs the actual batched GDN plus graph before the
original observer. The only private adapter substitutions are the existing
combined integration and the original `Handle.__init__`'s `TraceGraph` class,
which uses the already frozen E052 batched trace policy. `Handle.restore` and
all comparison math are unchanged. No direct-call prelude or prefix truncation
is injected.

The unchanged combined `CapturedRoutes` consumes the entire paired
`combined-graph-call` / `route-call` stream, compares each replay row with its
captured row, and requires EOF and the original SHA. Trace deltas count actual
Python branches including tracing. A compiled reuse may have zero new Python
branches; no trace counter is represented as a GPU execution count.

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/c8_batched_http_replay_v1/qualify.py \
  --prepare --capture-child ABSOLUTE_CLOSED_E052_CHILD --spec-sha256 ACTUAL_SPEC_SHA \
  --out FRESH_METADATA_DIRECTORY
# External owner only, after source/plan review, commit and admission:
.venv/bin/python -I -B bench/m5_ultra_256gb/c8_batched_http_replay_v1/qualify.py \
  --run --plan ABSOLUTE_PLAN --plan-sha256 ACTUAL_PLAN_SHA --out FRESH_NUMERICAL_DIRECTORY
```

Plan schema is `c8-batched-http-replay-plan-v1`; `prepare` returns the dictionary
without writing files. `validate(plan)` rebuilds its complete binding.
`run(plan, out)` returns 0 only after complete original numerical checks, exact
route EOF/count/SHA checks and final input/source revalidation.

Terminal schema `c8-batched-http-replay-result-v1` retains PID/PPID, actual base
receipt, case, coverage including combined statistics, actual steps/requests,
route comparison, emitted route SHA and failed evidence. Producer PASS sets
only `replay_numerical_checks_passed` and `source_input_pins_rechecked`.
`numerical_qualified`, `http_qualified`, `performance_qualified` and
`owner_resources_restore_qualified` remain false pending independent checks.
Successful tensor payloads are not retained by the original comparator; its
per-array device finite/bit receipts and first-failure payloads are preserved.

Keep the original 2 GiB allocator cache. The external owner's registered child
and system thresholds are sampled limits, not hard allocation caps. Do not
infer a safe 64 GiB threshold from a short prefix: the earlier complete replay
already measured a 68.55 GiB MLX peak. This replay is a numerical qualification,
not a throughput/latency point.
