# Native GDN preparation independent of the short-context graph

Default-OFF experiment. The44 candidate/canary/raw-auditor/serving CPU tests pass in the recorded preparation; no device execution has been performed.
No core/frozen source changed. No numerical or speed claim.

The real oMLX C1 warm log reported `Qwen B1/T1 fused GDN prework engaged`. Its
installed `qwen35_gdn_prework.py:2050–2098` keeps normal projections/recurrence/
norm/output and substitutes one preparation pass. Native already contains
`kernels/gdn_decode_fused.py:fused` for the same broad purpose, but only the
packed `_decode_fused` route exposes it. `CompiledDecode` limits that route to
used<=256. The D512 prompts start at 886–978 tokens, so this is a concrete route
difference. No oMLX timing log is used as a formal throughput measurement.

The proposed change exposes only the existing native preparation/state kernels
through ordinary B1 `_decode`. qkv, z, b, a and out remain the five original
Linear calls, with current original parameter arrays. Gates keep their actual
ON/OFF branch. Attention, context bound, scheduler, packing and output norm are
unchanged. It does not enable the separate `norm_reuse` switch or E024.

## Exactness boundary

Normal `_decode` makes a BF16 convolution product then a BF16 sum, BF16 SiLU,
repeated q/k, and the serial-normalizing recurrence. The reused preparation
kernel explicitly rounds all four BF16 products/additions, retains native BF16
activation and serial 128-term q/k normalization, and supplies compact q/k to
the existing state kernel. It writes the same logical three-row convolution
tail but owns a compact buffer rather than a view of the four-row concatenate.
Physical ownership/lifetime differs and must be observed separately from bits.

This is **not a source proof of device bit equality**. The reused state kernel
uses `math_mode='safe'` while the ordinary helper uses default compiler options.
MLX reduction lowering, contraction, input strides and real hidden values can
matter. Short fused-path qualification does not certify this newly exposed
long-request scope. Do not copy oMLX's FP32 convolution or SIMD reduction tree.

## Minimal API

`make_decode(module, fused, enabled=False)` returns an injectable unbound method
for CPU contracts. `install(layer, gdn_module, enabled=False)` is the real entry:
OFF is untouched/None; ON verifies seven native source files, loads their real
existing fused helper, and returns an exception-safe cleanup handle. It replaces
only that instance's `_decode` using MethodType, not a class/module global.
Restore observers first, then the scope; a foreign wrapper blocks restoration
instead of being overwritten. Closed handles release layer references.

Install on a fresh idle target-only model and remove after requests drain or
engine teardown. The frozen model propagates `_norm_reuse_normal=True` only for
ordinary exact-context decode without capture. The adapter requires that marker,
the allow-compile marker, norm OFF, M5 Ultra and original fixed B1/H48/D128
BF16/FP32/NVFP4 geometry. Other routes use the original method before mutation.
The short packed `_decode_fused` entry bypasses the hook entirely. No generic
tracing detector, live model hot reload or retained compiled scope is supported.

No projection arrays are cached: idle same-shape parameter replacement is read
from the current layer on every call. The fused result commits both states in
the same place as native, before norm/out. Failures after commit propagate; they
must never retry the original method on already advanced state. Counters prove
Python calls only; graph tracing is not device invocation evidence.

## Qualification and tutorial plan

1. First run the seven CPU fake tests under the root-owned CPU window. They
   check routing, unchanged five projections, actual parameter object lookup,
   original fallback, exception cleanup and references. They prove no GPU bits.
2. Reuse the real-layer input capture/checker. Compare ordinary vs unbound
   prework on B1 decode tensors from at least early/middle/late GDN layers and
   fixed finite cancellation/zero/strided cases. Keep full returned output,
   conv and S raw bytes, original inputs and parameter before/after hashes.
   Fail on any changed bit; retain the first difference. Carry each arm's own
   state through multiple steps, never repeatedly feed the reference state.
3. For the model gate, reuse original 37-request same-schedule Observer and
   comparator: all full-vocabulary logits, 128 cache arrays, committed/staging/
   ahead states and IDs; observe B1 hits, B8 fallback, refill/retirement. Norm OFF,
   gates ON. Confirm prefill/capture/spec exclusions and independent idle
   changed-value/restored parameters. Unobserved invalidation/EOS paths remain
   explicit limits. This factory itself grants none of these qualifications.
4. Only after the exact-state gate, actual C1/C8 HTTP capture/replay and fresh
   observer-OFF paired throughput-vs-POST-to-DONE latency. Keep E024, lane0
   broadcast, E028 and this candidate separate for attribution. No kernel or
   diagnostic walltime is promoted to a Pareto point. Record each attempt and
   reject negative or nonrepeatable results rather than widening the gate.

[Recorded CPU attempts](../../../docs/experiments/m5-ultra-256gb/next-modules-prepared-cpu-v1/README.md). The separate route-restoration fixture failure and focused repair remain preserved.
