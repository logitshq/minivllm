# Optional rowwise verifier with shared MLP gate/up

Source prepared; device correctness and performance **NOT QUALIFIED**. Default
OFF. No GPU/backend imports, weight packing, model loading, owner, scheduler or
speculative acceptance machinery. Frozen E042/E054 sources remain unchanged.

`candidate.verify(..., enabled=True, mlp_input_project=project,
head_project=existing_head)` enables the new branch. The external qualification
child supplies the actual `nvfp4_multirow_shared/candidate.py::build(mx,
enabled=True)` callable and adds **`runtime.silu = mlx.nn.silu`** from the original
model's backend. The child must pin both sources and original frozen model. The
candidate's constants record those source identities; injected function identity
and actual device dispatch still require the external qualification receipt.

Without `mlp_input_project`, execution delegates to the hash-bound frozen
`rowwise_native_verify/candidate.py::verify`. With the switch enabled, a small
explicit layer-loop adapter retains its admission, embedding, contexts, owned
prefix states and optional head path. Per layer it executes all native B1 mixers
in token order and all original row post-attention norms, shares **only** gate/up
projections over T2/4/8, then runs SiLU, multiply, native K17408 down and residual
addition separately per row. The original GDN `_projection` argument is never
passed: that would select its distinct fused recurrence.

All64 gate/up/down metadata are checked before private state construction. Gate
and up require exact NVFP4 K5120/N17408/no-bias/live U32+U8 arrays; down stays native
K17408/N5120. The new arm has128 shared MLP calls per verifier call, independently
of the optional one head call. An explicitly enabled projector rejection/error
propagates once without fallback/retry; canonical input caches are not assigned.
Weights are read from live Linear modules on each invocation, with no derived
copies or cached parameter handles.

The 44.5263% logical packed-byte share is a prioritization measure, not measured
DRAM/latency savings. See
`bench/reviews/2026-10-19-c1-shared-backbone-projections.md` for shapes, source
identities, attention/GDN follow-ups and the smallest real-activation canary.
First compare actual layer0/layer32/layer63 gate/up outputs and complete MLPs at T4/T8.
Then repeat full logits/selected prefix/continuation qualification before any
new E054 cost measurement. No device work has been run for this candidate.

CPU tests use scalar fingerprints and original fake cache structures, not model
arrays. They check all prefix dependency/ownership results against a separate
native token-major traversal, exact operation order, default delegation,
pre-admission rejection, no retry, live parameter replacement and the shared-head
composition. They cannot establish MLX arithmetic, slice ownership, Metal
execution, numerical equivalence or speed. Every actual test attempt will be
retained with source snapshots and a forbidden-backend/process/socket guard.

Actual guarded CPU attempt: **12/12 PASS**, no failures/errors/skips, no forbidden
operations, source hashes unchanged. Evidence is retained at
`.artifacts/m5-ultra-256gb/rowwise-mlp-shared-cpu-v1` (tool `0b9d5e`), including
runner/argv, stdout/stderr, rc, copied sources and before/after hashes. This was
the first attempt; no failed CPU attempt preceded it.

## Prepared actual-activation canary (device NOT RUN)

`canary.py` is default OFF and imports only stdlib at module scope. It binds the
actual independently audited E054 child plan/result/audit, then reuses its
source/model/loading/canonical helpers. The new child plan has exactly these keys:

```json
{
  "schema": "C1-MLP-shared-activation-plan-v1",
  "canary_sha256": "<actual source SHA>",
  "e054_plan": {"path": "<absolute child-plan.json>", "sha256": "3d10d60dc72eff7e2c70444239c61be09c865b6e15109c7f43b8cdf8a6a66c2c"},
  "e054_result": {"path": "<absolute complete.json>", "sha256": "b719a1c25257e0b0fc93e80a8d49215e223d02f708ab5f879558ffe94968e9de"},
  "e054_audit": {"path": "<absolute independent-v2.json>", "sha256": "7fe77b62600f36b27823b68b80e54ee32815e9b830178870679540b5e3835972"},
  "deadline_s": 900
}
```

The owner supplies the lifecycle/resource policy and later E055 closed
prerequisite. No plan, owner or device execution has been created here.

```sh
.venv/bin/python -B bench/m5_ultra_256gb/rowwise_mlp_shared/canary.py
# Only under the later external owner:
.venv/bin/python -B bench/m5_ultra_256gb/rowwise_mlp_shared/canary.py \
  --run --plan ABSOLUTE_CHILD_PLAN --plan-sha256 EXACT_SHA --out FRESH_OUTPUT
```

From the original E042 used1023/position1022 canonical state, native causal
execution generates nine tokens and must reproduce the original eight teacher
inputs. T4/T8 get independent evaluated seed storage. At every layer the callback
returns concatenated **original M1 Linear outputs**. At layers0/32/63 only, the
pinned shared kernel also consumes the same actual post-norm X/live weights;
its output is saved and compared but never returned to the decoder. The head
also remains original M1, and runtime.silu must be the original model's
`mlx.nn.silu` function. Norms, mixers, native down and row residuals retain their
original arithmetic.

There are12 projection pairs,24 BF16 output files and6 input files (roughly
5.13MiB tensor payloads). Every raw file has a full original-dtype shape/hash/
finiteness descriptor. Before success, both new native-returning traversals must
match the **full vocabulary logits and complete selected k=T state** of both
frozen original verifier and native causal oracle. This produces4 live
comparisons of129 arrays, including full padded KV state. These provenance
checks are producer receipts, not saved raw state evidence. Canonical storage
and every original parameter's bytes must remain unchanged. Failure preserves
all partial outputs and cannot produce a successful canary receipt.

`complete.json` schema is `C1-MLP-shared-activation-canary-result-v1`; success is
`PASS_PROJECTION_CANARY_ONLY`. It records pid/ppid/plan_sha256,6 `captures`,
2 `traversals`,2 `provenance` receipts (native+frozen comparisons each),
projection_pairs12/raw_files30/callback_count256/native_m1_projection_calls1536/
shadow_projection_calls12/shared_outputs_returned0. Required terminal flags:
`projection_canary_complete`, `native_returning_provenance_complete`,
`device_cleanup_completed`, `source_model_pins_rechecked`. HTTP, full-model,
performance and Pareto qualification remain false. No timing samples are taken.
A later canary which actually returns shared MLP values and checks complete
logits/prefix/continuation is still required before cost measurements or use.

Guarded canary CPU v1 (tool `9e7300`) ran10 tests:9 passed,1 fixture error because
a fake immutable tuple was assigned in place. The failed attempt and exact
source/argv/output are retained in
`.artifacts/m5-ultra-256gb/rowwise-mlp-shared-canary-cpu-v1`. Only that test fixture
was corrected. Fresh v2 (tool `572749`) **10/10 PASS**, no errors/failures/skips,
forbidden operations empty, source hashes unchanged; evidence is in
`.artifacts/m5-ultra-256gb/rowwise-mlp-shared-canary-cpu-v2`. These scalar tests
include deliberately wrong shadow values which must never affect later model
results, and rejected corrupt native-state/frozen-logit provenance. They do not
establish device numerical equivalence. No GPU/model payload work was run.
