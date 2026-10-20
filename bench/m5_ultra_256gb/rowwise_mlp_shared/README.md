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

## Minimal next actual-activation canary

Reuse E042 canonical used1023 state and T4/T8 original teacher tokens inside the
existing owned model lifecycle. Supply this candidate with a `probe_project`
callback which maps the actual live `(weight,scales)` objects to the original
layer gate/up Linear. For every layer, its returned value is concatenated
**original M1 Linear outputs**, not unqualified shared outputs. At layers0/32/63
only, also run the pinned shared projector from the same actual post-norm input
and compare/save all BF16 words plus finiteness. Keeping native returned values
prevents a failed local candidate from affecting later captured inputs. Keep the
already-qualified shared-head route or original M1 head explicitly identified.

This is12 projection pairs (3layers×2widths×2projections),24 output files, plus6
shared gate/up input files: roughly5.13MiB of raw tensor payloads, excluding
metadata. They are live normalized activations collected by the new callback,
not E040 synthetic X or previously saved E042 hidden arrays. Bind the original
model's `mlx.nn.silu` into `runtime.silu`; preserve original per-row norms,
mixers, native down and independent state reset per width. Actual kernel
dispatch and captured inputs require source-bound receipts. This first canary
is projection correctness only; a subsequent full logits/prefix/continuation
qualification must validate that the collector/new scheduling preserves the
whole model before treating shared MLP outputs as production values. No new
owner or formal performance claim is needed.
