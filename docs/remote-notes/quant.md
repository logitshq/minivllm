# A4 — FP8 / NVFP4 linear layers and compressed-tensors scheme detection

Files: `minivllm/kernels/quant.py`, `minivllm/model/quant_linear.py`, `minivllm/model/quant_schemes.py`,
`tests/test_quant.py`. Box: RTX 5090 (sm_120, 31.4 GiB), shared with other agents.

## 1. What the checkpoint actually contains (format reverse-engineering)

`unsloth/Qwen3.8-27B-NVFP4` is a compressed-tensors checkpoint, `format: mixed-precision`,
compressed-tensors `version 0.17.2.a20260716`. Two config groups (config.json → `quantization_config`):

| group | format | weights | input activations | targets |
|---|---|---|---|---|
| group_0 | `float-quantized` | fp8 e4m3, `channel` (per output row), symmetric, static | fp8, `token`, dynamic | `re:.*self_attn\.(q\|k\|v\|o)_proj$`, `re:.*linear_attn\.(in_proj_qkv\|in_proj_z\|out_proj)$`, `re:.*lm_head`, `re:.*layers\.(56..63)\.mlp\.(gate\|up\|down)_proj$` |
| group_1 | `nvfp4-pack-quantized` | fp4 e2m1, `tensor_group` 16, scale_dtype e4m3, actorder static | fp4, `tensor_group` 16, `dynamic: "local"` (group scales dynamic, global scale static) | `re:.*mlp\.(gate\|up\|down)_proj$` |

`ignore` has 303 entries: every `model.visual.*` block, `visual.merger`, all `layers.N.linear_attn`,
`linear_attn.norm`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b` for every GDN layer N, and `re:^mtp.*`.
`kv_cache_scheme` is fp8 per-tensor static (the `k_scale`/`v_scale` tensors); unused while KV is bf16.

Rules of compressed-tensors matching (`compressed_tensors/utils/match.py`): an entry is either an exact
module name or `re:<pattern>` matched with `re.match` (anchored at the start only; the targets carry their
own `$`). `ignore` is checked before `config_groups`; the group order is the dict order. Hence
`get_scheme()` in `quant_schemes.py` is: no quantization_config → bf16; any ignore match → bf16; first
group whose targets match → scheme from its `weights` (`num_bits=4, group_size=16` → nvfp4;
`num_bits=8` → fp8); else bf16. The mlp targets of group_1 also match layers 56..63, but group_0 is
listed first and claims them — order matters, and `test_get_scheme_against_safetensors_header` checks every
2-D language-model weight in the real header (`ref/nvfp4_safetensors_header.json`) against the function:
168 nvfp4, 24 fp8 mlp, 64 attention fp8, 144 GDN fp8, 96 GDN bf16, embed bf16 + lm_head fp8.

Tensor dtypes (from the header): nvfp4 modules store `weight_packed U8 [M, K/2]`, `weight_scale F8_E4M3
[M, K/16]`, `weight_global_scale F32 [1]`, `input_global_scale F32 [1]`; fp8 modules store `weight F8_E4M3
[M, K]`, `weight_scale BF16 [M, 1]`.

### The exact (de)quantisation definitions

Read from compressed-tensors 0.19.0 source (installed `--no-deps` into a throwaway venv `~/scratch_a4/ctread`
on the box — never into the engine venv — and deleted afterwards). Relevant files:
`compressors/nvfp4/helpers.py` (pack/unpack), `quantization/utils/fp4_utils.py` (`cast_to_fp4`),
`quantization/utils/helpers.py` (`calculate_qparams`, `generate_gparam`, `compute_dynamic_scales_and_zp`),
`quantization/lifecycle/forward_helpers.py` (`_quantize`, `_dequantize`, `_process_group`).

NVFP4 weights:
- E2M1 code → value table `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`, bit 3 = sign (so 0x8 = -0, 0xF = -6).
- Packing: `packed = idx[2i] | (idx[2i+1] << 4)` — the **low nibble is the even element**. Unpack is
  `stack((p & 0xF, p >> 4), -1).flatten()`. (vLLM's `break_fp4_bytes` is identical.)
- `weight_global_scale = 448 * 6 / amax(|W|)` (`generate_gparam`, fp32).
- group scale: `s = e4m3(clamp(amax(|W_group|) / 6 * weight_global_scale, max=448))`, zeros replaced by
  0.125 (`_get_dtype_eps` for e4m3). The stored `weight_scale` is this e4m3 value, i.e. it **already includes**
  the global scale.
- quantise: `W / (s / global_scale)` → clamp ±6 → `cast_to_fp4` (round to nearest with ties to the even
  code: 0.25→0, 0.75→1, 1.25→1, 1.75→2, 2.5→2, 3.5→4, 5→4 — the same as hardware `cvt.rn` e2m1).
- dequantise (`_dequantize`): `W = fp4 * (float(s) / weight_global_scale)` in fp32 then cast to bf16.
  So `W[m,k] = e2m1[m,k] * weight_scale[m,k//16] / weight_global_scale` — the ARCHITECTURE §2 formula is right.

NVFP4 activations (`dynamic: local` + static `input_global_scale`): `compute_dynamic_scales_and_zp` with
`tensor_group`: reshape the last dim into groups of 16, `s_blk = e4m3(min(amax/6 * input_global_scale, 448))`,
then `x_q = round_e2m1(clamp(x * input_global_scale / s_blk, ±6))`, dequant `x_q * s_blk / input_global_scale`.
Therefore the GEMM is `y = Σ_k x_q w_q · s_x[t,k//16] s_w[m,k//16] / (input_global_scale · weight_global_scale)`;
vLLM's `alpha = 1 / (input_global_scale * weight_global_scale)` is this constant. (ModelOpt checkpoints store
the reciprocals; the compressed-tensors converter inverts them — `entrypoints/convert/converters/modelopt_nvfp4.py`.)

FP8: weight `scale = amax_row / 448` (bf16 because the weight is bf16; stored as BF16 [M,1]),
`w_q = e4m3(clamp(W / scale, ±448))`; dequant `w_q.to(scale.dtype) * scale`. compressed-tensors does that product in
bf16; `dequant_fp8` here does it in fp32 (one bf16 rounding less, 2^-9 relative — irrelevant at the 2e-2 test bar).
Activations per token: `s_x = amax(|x_t|)/448`, `x_q = e4m3(clamp(x/s_x, ±448))`; compressed-tensors' own
`TOKEN` reduction on a 2-D input degenerates to per-tensor (it keeps dims {0,1}), vLLM uses per-row — we follow
vLLM (per row of `[T, K]`) which is what the model is calibrated for in serving.

Pitfall: `torch.float8_e4m3fn` → float conversion of the e4m3 scales is exact in bf16 (3-bit mantissa fits), so
reading `weight_scale` as bf16 or fp32 gives the same numbers; the global-scale division is where fp32 matters.
