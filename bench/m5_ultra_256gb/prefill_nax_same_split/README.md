# E025: same-split NVFP4 partial diagnostic

CPU/static preparation only. No Metal compilation, device numerical qualification or performance result exists yet. The candidate is **OFF by default**, lives entirely in this directory and does not patch Linear, MLX, the environment or any server. A rejected bit gate must remain a rejection.

This implements the narrowly scoped hypothesis from [the NAX source review](../../reviews/m5-nax-prefill-opportunities.md): keep the original split-K partition and BF16 intermediate rounding points, and try the official MLX NAX matrix helper for each partial. Different internal accumulation order may still prevent exactness.

The first screen registers only T19 and these three shapes:

| Projection | K | N | Original split count | K per partition |
| --- | ---: | ---: | ---: | ---: |
| GDN z | 5120 | 6144 | 2 | 2560 |
| MLP down | 17408 | 5120 | 2 | 8704 |
| Attention/GDN out | 6144 | 5120 | 3 | 2048 |

M1/11/12/32/33/128 and every unregistered shape use the original `mx.quantized_matmul`. M12/32 were proposed as later diagnostic extensions in the research note; this first implementation does not enable them. Only BF16 activations, UINT32 packed NVFP4 weights, UINT8 group16 scales, MLX 0.32.3 and the previously qualified Apple M5 Ultra / `applegpu_g17d` identity qualify. Quantization is fixed to NVFP4/group16/bits4, transpose true, no bias/global scale. No dequantization/requantization of the checkpoint occurs.

## Implementation and compiler feasibility

`candidate.py` reads 15 SHA-pinned installed MLX headers; `installed-sources.json` binds their exact identities. It expands only registered quoted includes, retains Apple's copyright notices, and keeps SIMD and NAX definitions in **separate libraries** because their helper names otherwise collide. It truncates each floating-quantization source after the first `fp_qmm_t_impl`, before unrelated kernel entry points. The required library preambles are roughly 91 KiB SIMD and 46 KiB NAX, reconstructed from the installed headers rather than vendoring a framework. See `LICENSE.MLX` for the upstream license.

The SIMD reference calls the unmodified official helper with BM/BK/BN=32. The NAX helper has precisely three textual changes: its function name; an additional constant `K_eff` reference; and the contraction-loop limit `K`→`K_eff`. K remains the **full original row stride** for activations, packed weight rows and scales. Partition offsets are `x+k_start`, `uint32_weight+k_start/8`, `uint8_scales+k_start/16`; output offset is `partition*M*N`. Only the loop stops at K_eff. Simply passing K_eff as K would silently read the wrong matrix rows and is explicitly avoided.

NAX uses BM32/BK64/BN64 and WM/WN2. Both arms use threadgroups `(32,2,2)`. Custom-kernel grids are thread counts, so they are the native group-grid multiplied componentwise by this threadgroup shape. Every output cell in `[split,M,N]` is written exactly once; partial inputs and output are not aliased. The unchanged native NAX loader and FP32 accumulator remain in the helper. Expanded headers use fixed global `constant int` geometry so the helper's constant-reference parameters keep the correct Metal address space.

The actual MLX 0.32.3 `fast.metal_kernel` chain is `CustomKernel::eval_gpu` → `Device.get_library` → `build_library_` → `get_metal_version`: it is not limited to Metal 3. On this host macOS 27.0.1 the reviewed code selects Metal 4.1; macOS 26 selects 4.0. Its standard builder prepends `metal::utils()`, supplying the same MLX scalar helpers. The public compile-options argument accepts `math_mode`; this candidate explicitly uses `safe`, matching the native default. No language-version option or private exported C++ function is required. These are static feasibility facts; actual compilation can still fail and will be recorded as FAIL without an eager fallback.

Official source anchors: [custom kernel](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/custom_kernel.cpp#L51), [source generator and builtins](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/common/metal_kernel.cpp), [library compiler](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/device.cpp#L653), [native split-K](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp#L1152). Small downloaded source copies used for this review are in `.artifacts/m5-ultra-256gb/nax-review-v2/`; runtime execution only requires the pinned installed headers.

## Concrete commands and the numerical contract

CPU-safe preparation:

```sh
.venv/bin/python -B bench/m5_ultra_256gb/prefill_nax_same_split/screen.py --describe
.venv/bin/python -B -m unittest discover -s bench/m5_ultra_256gb/prefill_nax_same_split -p 'test_cpu*.py'
```

Only the root GPU owner may later execute this diagnostic, inside its existing service lease, resource monitor and hard process timeout:

```sh
.venv/bin/python -B bench/m5_ultra_256gb/prefill_nax_same_split/screen.py \
  --run --out .artifacts/m5-ultra-256gb/prefill-nax-same-split-synthetic-v1
```

This first command uses seed 20261017 and materializes synthetic BF16 activations, full-range packed FP4 bits and finite FP8 scale encodings. It is an **operator diagnostic**, not a Qwen model-input qualification. Frozen complete input files and their SHA-256 values are retained. It loads no model. No timing is performed.

The same executable accepts genuine captured projection tensors with `--fixtures PATH --fixtures-sha256 SHA`. The manifest must have exactly `schema=E025-projection-inputs-v1` and `cases`, with exactly `gdn_z`, `mlp_down`, `out`; each case is `{path,sha256}`. Each safetensors file must contain exactly `activation: BF16[128,K]`, `weight: UINT32[N,K/8]`, `scales: UINT8[N,K/16]`. Registered T19 uses its first 19 rows; all declared fallback lengths use the same first-row prefixes. Captures must record their model/layer provenance externally; this CLI validates tensor bytes, not the truth of an unprovided capture history.

For each projection the gates are sequential and fail immediately:

1. OFF output equals original official quantized matmul bytes.
2. SIMD reconstructed partials, reduced with `mx.sum(axis=0)`, equal original official QMM output bytes. Without this, the reconstructed reference is invalid.
3. Every NAX partial equals its SIMD counterpart byte for byte, with finite values and matching dtype/shape.
4. NAX partial reduction and the actual enabled callable both equal original QMM bytes.
5. All six unregistered M values return original output, without generating new custom-kernel receipts. All input tensor bytes remain unchanged, including an exit snapshot when a numerical gate fails.

Both reductions consume a row-contiguous BF16 `[split,M,N]` array and use axis0 outside `mx.compile`. Official `common/reduce.cpp` maps that layout to `ContiguousStridedReduce(shape={split},stride={M*N})`; `metal/reduce.cpp` calls the same `strided_reduce_general_dispatch` used by native QMM. For split2/3 this reaches the same small column reduction. The reference-vs-original byte gate is still mandatory; static dispatch reasoning does not replace it.

`first-difference.json` identifies the first failed gate and differing byte offset; `first-difference.safetensors` retains both original-dtype arrays and its SHA. `events.jsonl` is flushed after each gate, `kernel-source-receipts.json` identifies generated headers/bodies and installed pins, and `complete.json` distinguishes PASS/FAIL. Passing output arrays and full before/after inputs are saved. Source/fixture pins are rechecked after a successful run. A process killed by the external owner may have no final summary; incomplete files must not be treated as PASS.

There is no allclose escape, automatic precision change, new runtime, model-serving code or performance claim. A future captured-input PASS would still need complete layer/cache/output qualification and then a separate unobserved throughput-versus-complete-request-latency experiment before adoption. Source-level use of MPP also does not measure NAX hardware utilization.
