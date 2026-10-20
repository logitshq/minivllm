# E057 independent shared MLP + head local-cost audit

Actual E057 device execution and independent saved-result audit now PASS (2026-10-20); see the E057 experiment README and local evidence manifest. This default-OFF script imports only small source modules and the standard library. It never loads a model, invokes a backend, probes a service, or changes resource policy.

The auditor retains the frozen E054 v2 raw tensor, qualification, canonical-state and timing mechanisms. It reads all 32 saved safetensors files from 16 qualification pairs, verifies original dtype, full shape, finite bits and every byte across 2,064 tensor pairs, including selected prefix state and native continuation. It independently recomputes all 400 ordered timing records and 20 groups, keeping negative break-even budgets. Timed calls retain output IDs; this does not establish 400 full-tensor comparisons.

E057 adds exact actual source and plan identity, the completed E056 canary and independent audit binding, six ordered qualification receipts containing 128 shared gate/up calls each, and six original shared-head receipts. Those call receipts describe source-bound Python calls, not GPU dispatch counts. All 1,349 original parameter hash receipts must match before qualification, immediately before timing and after timing, and match the audited E056 original values. Canonical storage uses the original double-root paths. Saved source copies, child/owner/reap chain, final observation acknowledgement and same-service restoration must close before any numerical file is read. Runtime/model payload hashes are accepted from the original source-bound verification receipt rather than recalculated.

The resulting `E057-shared-MLP-cost-independent-v1` record exposes `local_cost_audited` and `shared_mlp_cost_audited` only after PASS. It keeps the original `timings`, `timing_count=400`, `group_count=20`, `source_and_small_evidence` and `complete_sha256` fields. New fields include `qualified_mlp_calls=768`, `qualified_head_calls=6`, `timed_mlp_observer=false` and `parameter_receipt_sets=3`. Any failure is saved with partial counts; existing output files are refused.

## Invocation after completed restoration

Use the actual preregistered parent SHA and a fresh output path:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/rowwise_mlp_shared_cost_analysis_v1/audit.py \
  --run \
  --child .artifacts/m5-ultra-256gb/c1-mlp-shared-cost-v1-numerical \
  --parent-plan bench/m5_ultra_256gb/rowwise_mlp_shared_cost_device_v1/prereg-v1/plan.json \
  --parent-plan-sha256 ACTUAL_REGISTERED_PARENT_SHA \
  --out .artifacts/m5-ultra-256gb/c1-mlp-shared-cost-v1-independent.json
```

## CPU evidence

Actual guarded attempt v1 passed 9/9 tests, with no forbidden backend/process/network/signal operations and unchanged source hashes. Evidence and exact runner, argv, stdout/stderr, return code, and source snapshots are retained at `.artifacts/m5-ultra-256gb/rowwise-mlp-shared-cost-audit-cpu-v1`. There were no failed CPU attempts for this auditor.

The small tests cover ordered MLP records and altered late callbacks, original finite/signed-zero byte checks, the frozen 2,719 initial plus 655 post-timing records, all 400 synthetic timings including negative budgets and wrong IDs, saved closure and acknowledgement rejection, the pre-timing parameter gate, and preserved failed outputs. The synthetic coordinator test substitutes bounded raw counts; it does not create multi-GiB state fixtures or represent a device qualification. Original byte-reader behavior is exercised separately with tiny real safetensors files.

These are local full-verifier plus selected-state commit costs versus native AR(T). Draft generation, acceptance/rejection, rollback, scheduling and HTTP are excluded. An all-T-accepted remaining budget is not an end-to-end speedup. E054 head-only results are historical, not a paired third arm. No engine winner or Pareto qualification follows from this test alone.
