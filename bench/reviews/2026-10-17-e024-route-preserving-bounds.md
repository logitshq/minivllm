# E024: keep the native attention reduction route while bounding graphs

2026-10-17. Source review and proposed next experiment only. No new candidate,
GPU execution, timing, or correctness qualification. The independently audited four-arm isolation
found A–B/A–D different and A–C/B–D exact at B1/used257; see the
[result archive](../../docs/experiments/m5-ultra-256gb/E024-dispatch-isolation-v1/README.md). That single point does not qualify the proposal below.

## Fixed source evidence

MLX 0.32.3, Apple M5 Ultra `applegpu_g17d`, the pinned Qwen3.8-27B NVFP4 model,
BF16 attention, head dimension256, one query per row, and the original explicit
boolean length mask are the scope. Model state and arithmetic are unchanged.

- [Official host dispatch source](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/scaled_dot_product_attention.cpp#L444-L714),
  [saved copy](sources/mlx-v0.32.3-scaled_dot_product_attention.cpp), SHA256
  `4eaca28cbe1d192a3137a970b2f4d7cdecaba67a1f448b0b0b5f0309b6b3961e`.
  Lines951–1019 select vector mode for query length<=8 and two-pass on `d`/`s`
  devices at K>=1024. Lines444–498 keep single-pass launch/reduction geometry
  independent of N. Lines569–606 keep **128 blocks** on `d` throughout
  N1024..1536; batch size is a grid dimension, not this block-count condition.
  The separate GQA kernel requires no mask, restricted head dimensions, and
  N>=8192 (lines546–556), so it is excluded here. `MLX_SDPA_BLOCKS` must remain
  absent under the already pinned clean environment.
- [Official shader](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/sdpa_vector.h),
  [saved copy](sources/mlx-v0.32.3-sdpa_vector.h), SHA256
  `f52c8da21f2c31f3f77168d326a49815fc29bde4b573883896a8eb3298f11508`.
  Single-pass fixes **BN=32** (line43) and assigns keys by SIMD group modulo32
  (lines99–147). Boolean-false entries skip score, exponential and accumulator
  updates. Two-pass assigns keys by block index modulo `blocks` (lines263–306),
  with the same conditional skip. Its final reduction depends on `blocks`, not
  N (lines498–532). The distinct GQA shader partitions using N-dependent chunks
  (lines362–371); that excluded route must not be conflated with this one.

Line numbers above refer to complete downloaded raw files, not the browser's
compressed source view. Raw URLs are the tagged GitHub paths with
`raw.githubusercontent.com/ml-explore/mlx/v0.32.3/` prefix. Original copyright
headers and the [MIT license](sources/mlx-v0.32.3-LICENSE) are retained;
license SHA256 `ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01`.
Matching tagged source does not prove the installed binary's runtime kernel
trace; actual runtime/source pins remain required.

## Small proposed change, still default OFF

For the existing eligibility domain 257<=used<=1536 only, choose bound1023 when
`ctx.max_len<1024`, otherwise bound1536. Preserve logical positions/lens and
original boolean mask; keep physical capacity1536. Outside the domain keep the
original eager fallback. Do not edit the already rejected frozen candidate.

The source-based hypothesis is that padding within either bucket adds only
masked iterations: valid keys retain their lane/block and accumulator order.
The original eager forward and proposed bound then use the same 32-way or
128-block reduction. This is stronger than merely choosing the same kernel
name, but remains a prediction until full tensors are tested. Additional mask
iterations are not free and no speedup is projected.

There are **two bound callable families**, not necessarily two machine graphs.
Batch and explicit-state shapes can cause separate `mx.compile` specializations.
Keep real invocation/build witnesses and report what actually compiled.

## Mandatory next numerical coverage

1. **40 direct points:** B1..8 crossed with used257/1023/1024/1280/1536, compared
   to original eager `max_len=actual max(lens)`. Start with B1/used257 as the
   cheapest canary. Each point starts from independent copies of the same
   verified real prefill state; preserve all input bytes and parameter values.
2. Reuse the same candidate through actual transitions **1023→1024→1025** and
   **B1→B8→B1**. Include nonuniform row lengths on both sides of1024. The bucket
   is selected from batch max length, just as native dispatch uses that shared
   K slice; do not select a separate route per row. At least one mixed batch
   must retain short rows while its longest row crosses the boundary.
3. Compare full logits and all128 K/V/conv/S arrays in original unsigned bits,
   finite, shape/dtype/metadata, greedy IDs, input before/after and parameter
   hashes. Preserve first failure and raw tensors; no tolerance or token-only
   substitute. Check actual compiled invocation at every required point;
   fallback cannot count as candidate coverage.

Passing these points would still leave the original parameter replacement and
actual ahead/retirement lifecycle gates, then current gates-ON full D512 and
HTTP schedule replay, before observer-OFF throughput versus complete-request
latency measurements. The old gates-OFF diagnostic does not qualify production
gates-ON behavior, future arrivals, or any Pareto point.
