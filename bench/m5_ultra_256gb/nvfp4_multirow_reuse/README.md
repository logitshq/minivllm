# T2/4/8 NVFP4 weight reuse — source-only, default OFF

This independent prototype has not been Metal-compiled, device-run, numerically
qualified or benchmarked. No existing engine, verifier, frozen source, model or service changes.
Importing candidate.py creates only source strings and functions.
`build(mx)` returns None without inspecting mx. Only explicit
`build(mx, enabled=True)` checks the pinned MLX 0.32.3/M5 Ultra/g17d metadata
and constructs the lazy Metal callable. An unsupported shape returns None;
there is no fallback to native M=T QMM. Errors after admission propagate.

The scope is one bias-free NVFP4 projection, not a decoder. Input is BF16
[T,K], T in {2,4,8}, positive K divisible by 512; weight is U32 [N,K/8],
scale is U8 [N,K/16], positive N divisible by 4. Conservative uint index
bounds are enforced. There is no affine bias or global-scale input.

## Arithmetic and intended reuse

The existing `tutorial_code/miniengine/kernels/nvfp4_qmv.py` is the B1 source
reference. It adapts MLX 0.32.3 `fp_qmv_fast_impl/qdot` (MIT, Apple Inc.;
license at ../../../THIRD_PARTY_NOTICES/MLX.txt).

One SIMD group still computes one output row, with 32 lanes, two SIMD groups
per threadgroup and k increasing by 512. For each output row/k/lane, load its
scale once. For j=0..3, load one packed ushort and decode four weights once,
then apply them to each token's separate accumulator. Keep the original four
product left-associated sum, the four j accumulations, scale multiplication
after the full 16-value group, k accumulation, 32-lane simd_sum and final BF16
cast. FP contraction is explicitly OFF and math_mode is safe. Do not absorb
scale into each weight or introduce split-K, FMA, matrix tiles or reassociation.

This places j outside the token loop: no xx[T][16] array. Main source-level
live values are 2T accumulators plus four decoded weights and each token's
inputs; register allocation, spills, instruction count and occupancy are
unmeasured. The packed weight/scale stream is logically read once per block
instead of T times: 9NK/16 bytes. This is not measured DRAM traffic or a
claimed T-fold speedup; input traffic, compute, output and reduction remain.

## Layout and cost

The installed MLX API source
`.venv/lib/python3.13/site-packages/mlx/core/fast.pyi:184,201` documents
`ensure_row_contiguous=True`. This candidate sets it explicitly for x, w
and scales before flat-pointer indexing. Noncontiguous inputs are therefore
admitted through that API's staging semantics, not through a guessed stride
check. Any allocation, staging copy and lazy evaluation must remain inside
end-to-end timing and memory measurements. Actual sliced/strided input cases
must be device-qualified; CPU fakes only check that the option is passed.

No parameter packing, replacement, eval, model import, acceptance or rollback
is added. A per-call GPU-default guard avoids dispatch after switching to CPU.
The runtime metadata guard does not replace source/model provenance.

## Required qualification before integration

1. Keep T2, T4 and T8 separate candidates. Freeze actual candidate, B1 custom
   source, native MLX/compiler/device, weights and activation provenance.
   First compare every candidate row with BOTH an independent actual native
   MLX B1 call on x[t:t+1] and the existing custom B1 call. Native M=T is not
   the oracle. Use original BF16 bytes, including signed zeros; a converted
   float digest, tolerance or matching argmax is insufficient.
2. Cover every intended actual projection shape, actual model weights and
   captured model activations, plus diverse finite magnitudes, cancellation
   and scale encodings, and contiguous/sliced x/w/scales. Any mismatch is a
   failed domain, not a reason to silently replace the oracle.
3. The smallest future integration is the vocabulary head after original
   per-row norms. Larger integration needs projection stages around the
   original per-token attention/GDN recurrence. Do not simply call the whole
   layer with T rows. Before any integration qualifies, compare every full
   vocabulary logit and every complete KV/conv/S prefix against original
   sequential C1, including 257/1023/1024 positions and untouched input/model
   bytes. Compiled integration requires its own repeated qualification.
4. Only after numerical PASS, measure warmed repeated paired trials comparing
   one eval of T native-B1 outputs, one eval of T custom-B1 outputs, and one
   eval of the candidate. Materialize the same [T,N] output. Avoid artificial
   per-row eval synchronization and repeated-identical-input graph reuse.
   Exclude JIT/provenance hashing; include necessary concatenation/layout
   staging and publication of the equivalent outputs. Report raw block
   latency, cost per verified token, distribution, memory and any available
   compiler register/spill evidence. Record matrix/cache regime and order.
   A useful verifier must beat T sequential C1 cost with room for drafter,
   acceptance and publication; a projection micro-win is not an HTTP/Pareto
   point and provides no end-to-end claim.

test_cpu.py contains five tiny source/dispatch mechanism tests with an injected
fake backend. They intentionally do not simulate Metal arithmetic or prove
numerical equality. After the independent profile driver restored its service, the guarded CPU run
passed **5/5**, with no skips, forbidden operations or source changes
(0.00398849998600781 s). Raw command, stdout/stderr, source-before/source-after
pins and frozen test-time sources are retained in
`.artifacts/m5-ultra-256gb/nvfp4-multirow-reuse-cpu-v1/` (tool `ce543f`).
That archive retains the earlier README stating tests had not yet run; this
paragraph records the subsequent result. Candidate/test source is unchanged.
The guard denied backend imports, mmap, subprocess, process signals and network.
No actual Metal arithmetic, device qualification or speed measurement occurred.
