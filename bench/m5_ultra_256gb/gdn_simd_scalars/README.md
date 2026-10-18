# C1 GDN scalar sharing — source-only candidate

2026-10-17. Default OFF, no model integration, no device compilation, no CPU
tests executed, no numerical qualification or performance result.

Native `tutorial_code/miniengine/kernels/gdn_metal.py:33–38` recomputes the same
serial 128-element q/k norms, two scale factors, exp(g) and beta in every
thread. With the existing `(32,4,1)` threadgroup, each 32-lane SIMD group has
the same head and those same scalar inputs. This candidate computes the exact
original scalar expressions on lane 0 and broadcasts four FP32 values. The
state products, SPLIT=4 reduction order, barriers, allocation, casts and launch
geometry remain literal copies. It adds no launch or threadgroup barrier.

This is a reduction in source-level duplicate work, not a claimed speedup.
The compiler may already hoist uniform expressions; lane divergence and four
broadcasts may offset any saving. Only actual compiled-code/device measurement
can establish a benefit. No speed estimate follows from diagnostic walltime.

The installed oMLX backend provides the useful design principle, not our
numerical oracle: `mlx_vlm/models/qwen3_5/language.py:1038–1103` prepares q/k
before recurrence; the installed `omlx/patches/qwen35_gdn_prework.py` combines
selected preparation paths. `omlx/engine/vlm.py:2618–2627` attempts to install
that patch. Its eligibility depends on the actual model; source presence alone
does not prove every qualified request used that branch. Its normalization,
state layout and reduction trees differ from native and are not copied here.

The root owner subsequently reported the real oMLX C1 warm log
`Qwen B1/T1 fused GDN prework engaged` at 17:19:51. That message maps to
`qwen35_gdn_prework.py:2050–2098`: the Qwen3.5 prework branch, not the separate
Qwen4 fused whole-step branch. It keeps its original qkv/z/b/a projections,
recurrence, output norm and output projection, replacing the preparation chain.
The warm generation-rate log is not a formal POST-to-DONE point.

This identifies a higher-priority integration candidate before this lane-sharing
variant: decouple native's already-present `gdn_decode_fused.fused(raw,conv,w,g,
beta,S)` from `CompiledDecode`'s used<=256 / packed-projection entry. Feed it
the ordinary unchanged Linear projections and use the ordinary output Linear.
That would leave attention shape/dispatch, projection arithmetic and E024
unchanged while exposing native's conv/activation/norm preparation to long
decode. It is not implemented here and still needs full-state exact qualification
on long-request data. Existing short-path results cannot certify that new scope.
Do not stack both changes in the first experiment: test prework alone against
the native eager/gates-ON/norm-OFF route, then this kernel as a separate arm.

## Small API and independent scope

`make_step(mx_module, baseline_module, backend_supported, enabled=False)` returns
a callable `(q,k,v,g,beta,S) -> (o,S_new)`. Import needs no MLX. The future harness
passes the pinned native module and actual M5 Ultra / MLX 0.32.3 / applegpu_g17d
backend predicate. ON accepts only B1/H48/D128 BF16 q/k/v and FP32 gates/state;
all other shapes/types/backend routes preserve the original helper. An enabled
source or geometry mismatch rejects construction. Compile/dispatch exceptions
propagate instead of being hidden by a fallback. No array or request state is
retained. `calls` and `last_path` witness only actual Python dispatch, not GPU
completion; all claimed successes must evaluate the outputs independently.

This does not turn on existing `norm_reuse`, change E024, or alter packed short
decode. `CompiledDecode` short/fused route is limited to used<=256; the fixed
D512 tasks begin above that limit and ordinary GDN `_decode` calls the original
helper. A later scoped per-instance integration must intercept precisely that
call without patching the production kernel module. Do not add a global hook.

## Falsifiable next experiment / tutorial sequence

1. Freeze this source and the original helper/backend. CPU mocks check lazy OFF,
   rejection, unchanged recurrence source and explicit-state lifetime only.
2. Root's existing bounded owner runs a small B1 kernel screen: finite zero,
   signed-zero, cancellation, noncontiguous inputs (same original FP32 casts),
   finite extremes, and real decode tensors from representative early/middle/
   late GDN layers. Evaluate and save all original-dtype output and S bytes.
   Keep every scalar/array first difference. Any bit difference rejects this
   candidate; no tolerance relaxation. NaN pattern diagnostics are separate
   from the finite model gate and must not be labeled finite qualification.
3. Single-step exact is insufficient: carry each arm's own state through a
   fixed multi-step chain, check inputs unchanged and all output/S bits at each
   step. Shape/backend fallbacks must really call the original helper. Confirm
   device SIMD execution width=32 before using the lane mapping assumption.
4. Only after those gates, scoped default-OFF integration reuses the existing
   37-request comparator/Observer with full logits, 128 cache arrays and ahead
   pairing. Keep gates ON, norm OFF and record B1 hits, B8 fallback, ordinary
   prefill/capture exclusions and partial-retire/refill coverage. No new generic
   validator is needed. Actual HTTP C1/C8 capture/replay follows before formal
   observer-OFF throughput vs complete-latency comparison.
5. Compare evaluated full-forward and actual HTTP points in paired fresh runs;
   do not promote a kernel-only win into a Pareto claim. Retain all failures and
   negative points in the tutorial. E024 combination needs its own gate.
