# E054 auditor v2: exact canonical receipt path correction

The real E054 producer completed and restored. Audit v1 remains frozen and its
FAIL result is retained at
`.artifacts/m5-ultra-256gb/c1-verifier-cost-v1-independent.json` with execution
logs in `c1-verifier-cost-v1-independent-execution-attempt1`. It had already
validated all32 safetensors/2064 tensor pairs, then rejected producer receipt
names. No device rerun is needed for this auditor expectation error.

The frozen screen calls `support.snapshot(storage[arm])` for canonical
preservation. `mx.load` supplies a flat dictionary whose keys already contain
`root/state/...`, `root/tokens`, etc. The frozen snapshot visitor starts at
`root` and appends dictionary keys, yielding **`root/root/...`**. Ordinary
`{logits,state}` output snapshots remain `root/logits` and `root/state/...`.

V2 hash-loads the frozen v1 auditor and privately replaces only
`producer_checks`. It adds one `root/` prefix to the expected canonical maps for
both initial and post-timing preservation receipts. It never normalizes or
rewrites observed data. Counts remain2719 initial records (2064 output checks
plus655 canonical checks) and655 post records; all shapes, dtypes, finite and
bit_equal requirements remain exact. Raw reader/comparison, six selected-prefix
checks,400 timing rows, statistics, failure handling and source/lifecycle closure
are the **same original function code**. Actual full audit must reread all32
raw files; v1 partial success is not reused as numerical proof.

Focused CPU regression reads only the source-bound closed actual complete.json
(SHA `b719a1c25257e0b0fc93e80a8d49215e223d02f708ab5f879558ffe94968e9de`),
and extracts the actual frozen snapshot function AST for a tiny fake-array test.
No MLX/backend, model payload, service or GPU operation is imported/run.

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/rowwise_native_verify_cost_analysis_v2/audit.py \
  --run --child .artifacts/m5-ultra-256gb/c1-verifier-cost-v1-numerical \
  --parent-plan bench/m5_ultra_256gb/rowwise_native_verify_cost_v1/prereg-v1/plan.json \
  --parent-plan-sha256 f613b5df1494e9064ba84e2dd025b437177891b00bf5d98bcf18d03e5eb960ce \
  --out .artifacts/m5-ultra-256gb/c1-verifier-cost-v1-independent-v2.json
```

All original scope limitations remain: 16 complete qualification pairs, not400
full-tensor comparisons; local verifier+commit cost excludes draft/acceptance/
scheduling/HTTP overhead and does not establish an engine Pareto point.

Actual focused CPU result: **8/8 PASS**, first attempt, no forbidden operations,
no skips/failures/errors, source and closed metadata hashes unchanged. Evidence:
`.artifacts/m5-ultra-256gb/c1-verifier-cost-audit-v2-cpu-v1` (tool `4a999d`),
including copies of the actual closed complete.json and original v1 FAIL receipt,
all audited sources, argv, runner, stdout/stderr and exit result. Actual v2 full
raw audit has not yet been run by this preparation task.
