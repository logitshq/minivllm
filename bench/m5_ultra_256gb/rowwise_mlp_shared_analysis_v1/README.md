# Independent saved-byte audit for the E056 MLP canary

Source/CPU ready; **actual E056 audit NOT RUN**. `audit.py` is default OFF,
stdlib only, with no backend/model payload or live process/service operations.
The canary remains frozen at
`c1aeae682a5822a0483926fb588f3e184d307a1aeb7fc3a2380a0d16bf85ffd8`.

The auditor checks the explicit parent plan SHA, exact child plan/command and
E054 prerequisites, E055 closed predecessor receipts, retained/current source
pins, PID/reap chain, restored service receipt and final observation/exit-ack
hashes. These saved closure checks finish before numerical files are opened.
Owner source hashes come from the preregistered parent and must match retained
and current files; fixed numerical source hashes are also independently pinned.
It does not import or call the producer's validator.

It then independently reads all30 original BF16 tensor files:6 complete actual
normalized inputs and12 native/shadow gate/up pairs at T4/T8, layers0/32/63.
Header, exact shape, dtype, all payload bytes, finiteness, per-tensor/full-file
hashes and sidecars are checked. Pair comparison is exact original bits,
including signed zero; no numeric tolerance or sampling. Extra raw files,
missing pairs, aliases and truncated/changed payloads are rejected.

The full-logit/selected128-state provenance is **recorded live evidence only**:
4×129 expected producer checks, plus655 canonical-preservation checks and12
projection checks (1183 total). No successful full-state files were retained,
so this auditor cannot independently recompare their state bytes. It verifies
all exact receipt labels/shapes/dtypes/finite/equality flags and two native-
returning traversal/provenance records. Before/after1349 parameter hash receipts
must also equal the original receipt bound by E054's independent audit; model
payloads are not rehashed. The source guarantees shared outputs never feed later
layers. This cannot qualify shared MLP propagation, continuation, HTTP or speed.

```sh
.venv/bin/python -B bench/m5_ultra_256gb/rowwise_mlp_shared_analysis_v1/audit.py
# Only after actual device + owner restoration complete:
.venv/bin/python -B bench/m5_ultra_256gb/rowwise_mlp_shared_analysis_v1/audit.py \
  --run --child ABSOLUTE_NUMERICAL_OUTPUT \
  --parent-plan ABSOLUTE_PREREG_PLAN --parent-plan-sha256 EXACT_SHA \
  --out FRESH_INDEPENDENT_JSON
```

Output schema: `E056-MLP-canary-independent-v1`. PASS requires
`projection_bytes_audited=true`, raw_files30/input_files6/tensor_pairs12,
provenance_receipt_pairs4/provenance_receipt_arrays516,
canonical_receipt_arrays655/parameter_receipt_count1349, exact complete/parent
hashes and preserved source/evidence hashes. `raw_full_state_checked`, full-model,
HTTP, performance and Pareto flags stay false. All errors yield a saved FAIL
with partial counts; existing audit output is never overwritten.

First actual guarded CPU attempt **9/9 PASS**, tool `9a96cf`, evidence
`.artifacts/m5-ultra-256gb/rowwise-mlp-shared-audit-cpu-v1`. No failures, errors,
skips or forbidden operations; source hashes unchanged. The attempt includes
argv/stdout/stderr/returncode and source snapshots. Tiny synthetic evidence tests
cover end-to-end closed success, all30 shape-complete files, late-word/signed-zero
mismatch, NaN/Inf, truncation, missing late state receipts, canonical root/root
names, changed parameter hashes, terminal/source tampering, default inertness
and preserved FAIL output. No actual E056 data or active E055 result was read.
