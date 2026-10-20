# E054 independent saved-result audit

Prepared source; actual E054 results have not been opened. Default OFF, standard
library only. No backend imports, model loading, service queries, inference,
weight/runtime payload rehash, or live timing. Run only after the owner has exited
and its lease is closed with the same service identity restored.

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/rowwise_native_verify_cost_analysis_v1/audit.py \
  --run --child .artifacts/m5-ultra-256gb/c1-verifier-cost-v1-numerical \
  --parent-plan bench/m5_ultra_256gb/rowwise_native_verify_cost_v1/prereg-v1/plan.json \
  --parent-plan-sha256 ACTUAL_REGISTERED_SHA \
  --out .artifacts/m5-ultra-256gb/c1-verifier-cost-v1-independent.json
```

The audit first checks the registered plan, retained source copies, E042 seed
metadata, direct parent/child wait chain, completed terminal acknowledgment and
final observation hashes. These are saved lifecycle receipts; it does not
independently probe current process absence or recompute the resource policy.

It reuses the pinned, bounded original-byte reader from
`long_decode_isolation/audit_raw.py`. All **32 safetensors, 16 complete pairs,
2064 tensor pairs** are read with full vocabulary and all 128 state arrays,
including KV padding. Shapes, original dtypes, whole-file and per-tensor hashes,
finite bits and original words must match. Signed zero differences fail.
Repeated oracle prefixes/continuations must agree, and saved full-vocabulary
argmax independently binds the six selected-prefix pending IDs and timed IDs.
Head dispatch, holder lens, seed preservation and 1349 before/after parameter
hashes remain explicitly described producer receipts. Parameters are not saved
payloads and are not rehashed here.

All **400 chronological durations, 200 AB/BA pairs and 20 groups** are checked
against the raw ledger. Ratios, median group ratios, p50/p90 and the remaining
`native AR(T) - verifier+selected-state commit` time budget are recomputed.
Both device-dependent and host-synchronous native baselines are retained.
Negative budgets and slow groups remain visible. This is an optimistic **all T
accepted** draft-plus-overhead budget; draft cost, rejection, scheduling and HTTP
are excluded. No engine throughput, speculative speedup or Pareto claim follows.

Only 16 qualification pairs have full saved tensors. The 400 timed calls have
output-ID/commit checks from the pinned producer; this audit does not claim 400
full-tensor comparisons. Failed audits preserve a fresh JSON failure receipt and
return a nonzero exit code.

CPU evidence: `.artifacts/m5-ultra-256gb/c1-verifier-cost-audit-cpu-v1`
retains the first attempt (12 tests, 10 passed, two fixture errors): the manually
expected p50 used integer JSON values although an even-count `median` emits a
float. Only that fixture expectation changed. The fresh
`c1-verifier-cost-audit-cpu-v2` attempt passed **12/12**, no skips, no forbidden
operations, all source hashes unchanged during execution. Both attempts include
argv, runner, stdout/stderr, exit code, before/after hashes and copied sources.
Actual E054 audit remains **NOT RUN**.
