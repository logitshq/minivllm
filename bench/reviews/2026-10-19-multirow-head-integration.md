# Multirow NVFP4 → rowwise verifier: smallest actual-model entry

Read-only review, 2026-10-19. No source implementation, backend import, GPU execution, service action or commit was performed. Model inspection read only `config.json` and the three safetensors headers, not weight payloads. The only written file is this review.

**The actual vocabulary head fits the kernel.** Replacing only its independent M1 projections after the existing per-row norms is the smallest real-model numerical integration. It preserves the present prefix-state construction and full-vocabulary comparison contract. **Head-only reuse covers about 4.96% of logical full-projection weight/scale bytes; it is a numerical entry point, not the large-gain target.** Actual DRAM traffic share and latency share are unknown.

## Actual model and shape admission

Snapshot directory (`M` below):
`/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`

`M/config.json` SHA256: `7380cfc6eb77891381ee24e944e6bd0caed7a957707f2512ddd6effc82c7e706`.
It reports `model_type=qwen3_5`, `hidden_size=5120`, `vocab_size=248320`, 64 decoder layers, `tie_word_embeddings=false`, and quantization exactly `{group_size:16,bits:4,mode:nvfp4}`. The external model name is Qwen3.8; the stored architecture/source names remain qwen3_5.

Actual head entries in `M/model-00003-of-00003.safetensors`:

| Tensor | Original dtype | Shape | Data offsets relative to tensor payload | Bytes |
|---|---|---|---|---:|
| `language_model.lm_head.weight` | U32 | `[248320,640]` | `[2854265824,3489965024]` | 635,699,200 |
| `language_model.lm_head.scales` | U8 | `[248320,320]` | `[4994506016,5073968416]` | 79,462,400 |

For `nvfp4_multirow_reuse.candidate.supported`: `K=5120`, `N=248320`, `K%512=0`, `N%4=0`, `N*K=1,271,398,400 < 2^32`; for T8, `T*K=40960 < 2^32`. Weight shape is exactly `[N,K/8]`; scale shape is exactly `[N,K/16]`. Original one-row norms produce BF16, so concatenating their results gives the required BF16 `[T,K]`, T2/4/8. T8 output is `[8,248320]`, 3,973,120 bytes. No head packing or weight duplication is required.

The embedding has separate U32/U8 entries of the same shapes in shard 1. Because this model is untied, the actual target is `model.lm_head`, not `embed_tokens.as_linear`. The frozen model constructs distinct modules and the strict loader maps `language_model.lm_head.*` to `lm_head.*`; it does not manufacture a head/embedding alias. Do not replace or rebind either parameter. Read the live `head.weight` and `head.scales` at each call; do not retain stale packed handles. Actual runtime identity/layout admission still belongs to the device child.

## Scaling and oracle scope

The frozen `model/layers.py::Linear.__call__` passes x, packed weight and scales directly to `mx.quantized_matmul(..., transpose=True, group_size=16, bits=4, mode='nvfp4')`. This head has no affine bias, global scale, activation scale or post-projection rescale in that implementation. Its `Embedding.as_linear` has the same primitive shape, but is not the selected route here. The loader leaves U32 and U8 payload dtypes unchanged and does not fold a head scale.

All three snapshot headers were searched for `global_scale`, `input_scale`, `activation_scale` and `weight_scale` tensor names; none exists. Together with this exact frozen Linear/loader and config, this establishes that **the reviewed model path supplies no separate external global-scale tensor or multiplication**. It does not claim that all NVFP4 formats/models lack global scaling, that scaling is absent from quantization, or that an arbitrary external loader has identical semantics. Per-group U8 scales remain essential and the candidate consumes them once per group without moving their multiplication into the four products.

The qualifier's reference calls `engine.runner.decode_model` at used positions 257/1023/1024. Frozen `CompiledDecode.__call__` admits only `max_len<=256`, so these references take the original eager model's independent M1 `Linear` path; the final assertion also requires `_functions=={}` and `_packed is None`. Do not replace this oracle with native M=T QMM or with the custom M1 kernel.

The completed 36-case synthetic GPU check is not actual-head qualification: its largest N is 6,144, versus the actual head N of 248,320, and it used constructed weights/activations. Full vocabulary original-byte comparison remains required even though dimensions satisfy admission.

## Minimal edits, no new lifecycle framework

1. **`rowwise_native_verify/candidate.py::verify` (line 21, final projection at 79–83):** add an optional injected `head_project=None` argument. Preserve the entire embedding, layer-major B1 loop, contexts and prefix ownership. Preserve each `model.norm(x[t:t+1])` call at M1; concatenate only those normalized rows. With the explicit head-reuse arm, call the injected projector once with the live `model.lm_head.weight/scales`, returning `[T,vocab]` with the unchanged `prefixes`. Without it, keep the current per-row head route. Admit the actual untied NVFP4/no-bias head metadata before running the arm. A rejected/None projector result in the explicitly enabled qualification arm is a failure, not silent baseline fallback. Keep default OFF inert. Do not batch the norm or whole decoder layer.
2. **`rowwise_native_verify/qualify.py::{validate,device,checked_call,compare_case}` (73/270/186/197):** pin and load the existing `nvfp4_multirow_reuse/candidate.py`, construct `build(mx,enabled=True)` once inside the already owned device child, require a non-None callable, and pass it only to candidate calls. A small wrapper should record actual T/K/N/dtypes and one successful multirow head call for each of the nine valid cases; assert nine calls and no rejection-domain call. Keep reference seed generation, original row norms, all 42 prefix comparisons, parameter-value receipts, original input/canonical preservation and exception cleanup unchanged. This is on the order of 20–35 lines of candidate change plus 25–45 lines of qualifier plumbing, before focused test updates—not another owner/controller.
3. **Source sealing and offline binding:** the existing qualifier still has `BRIDGE_SHA='UNSEALED'` and `CANDIDATE_SHA='UNSEALED'`. It cannot currently launch. Freeze the reviewed modified candidate/qualifier, pin the unchanged multirow kernel, and generate a fresh explicit plan that identifies the head arm/kernel (a seventh `head_projection` source ref is sufficient). Update `rowwise_native_verify/audit.py::identities` (line 82) to require that new plan/source binding and actual nine-call receipts. Its byte/finite/prefix/input checks and raw artifact counts need no redesign: 96 raw files, 42 candidate prefixes, 5,418 output tensor pairs remain applicable if the existing dump layout is retained. Re-seal the auditor and retain previous source/CPU evidence; existing digests do not qualify changed code.
4. **Existing narrow CPU tests only:** extend candidate tests to check original per-row norms plus exactly one projector invocation, genuine live head arrays, OFF behavior, and failure on enabled-projector rejection; extend qualifier/auditor source binding and call-receipt checks. No new framework. The existing full-model prefix qualifier is the decisive next GPU test, under the root owner's existing load/resource/recovery lifecycle.

This route preserves all prefix logits and states structurally; numerical equivalence is still an empirical gate. The head is after the final decoder layer and writes only its output, so it need not modify any cache wrapper/state. Keeping the existing per-row norm is necessary to avoid introducing another rounding change. `ensure_row_contiguous=True` may stage inputs; any resulting staging/allocation must remain in later performance timing. This review makes no Metal occupancy, head latency or end-to-end speedup claim.

## Exact logical byte denominator

Safetensors header enumeration found **994** language-model tensor entries whose dtype is U32/U8, excluding `embed_tokens.*` and the loader-excluded visual/MTP prefixes. All 994 are respectively `.weight` or `.scales`: the complete decoder projections plus the untied vocabulary head. Their payload-byte sum is:

`D = 14,412,349,440 bytes`.

This is a metadata-derived **one logical sweep of packed full-projection weight and scale arrays**, not measured memory traffic. Embedding lookup is excluded because it reads selected token rows rather than the entire vocabulary matrix. Norm/conv/other floating parameters, activations, KV/recurrent state, intermediate storage, cache effects, compiler duplication and staging are excluded from this denominator.

Head bytes are:

`H = N*K/2 + N*K/16 = 9*N*K/16 = 715,161,600 bytes = 682.03125 MiB`.

`H/D = 0.0496214446491`, or **4.962144465%**.

With T independent full-projection sweeps, head-only register reuse logically changes `T*H` to `H`. The model-wide logical projection-byte reduction is `(T-1)*H/(T*D) = (1-1/T)*H/D`:

| T | Logical head bytes avoided per block | Reduction of T complete logical projection sweeps |
|---:|---:|---:|
| 2 | 715,161,600 | 2.481072232% |
| 4 | 2,145,484,800 | 3.721608349% |
| 8 | 5,006,131,200 | 4.341876407% |

These percentages are **neither total DRAM reductions nor latency reductions**. An independent M1 sweep may already reuse cached weights; the multirow kernel may change occupancy, arithmetic scheduling, activation traffic and staging. Actual speed and memory traffic remain unknown. The remaining ~95.04% of this logical projection-byte denominator lies in decoder projections, where extending shared-weight execution around the exact per-token attention/GDN recurrence is the larger opportunity.

## Reproducible source/header identities

Repository root (`R`): `/Users/ryanli/code/minivllm`.
Frozen native root (`F`): `/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine`.
The paths below are relative to these explicitly stated absolute roots.

| Source | SHA256 of full source bytes |
|---|---|
| `R/bench/m5_ultra_256gb/rowwise_native_verify/candidate.py` | `5903f60ceef9ff5429e21c0c3d38b6ac67a6a25423d71a6cafc47715ac0f6b34` |
| `R/bench/m5_ultra_256gb/rowwise_native_verify/qualify.py` | `d93ee4a906bb1f879326c04cfe9ca6ec3c889ef42817a4300cf07adb2ca01017` |
| `R/bench/m5_ultra_256gb/rowwise_native_verify/audit.py` | `34c4404af515ad6a859f25a1bc0df5aaa4c27a164d30534f11c0ec53f23b873d` |
| `R/bench/m5_ultra_256gb/nvfp4_multirow_reuse/candidate.py` | `ee9c5eef0e1282125310ee6661be1ec5ce64dc93a82add48a6d6c17f8958816f` |
| `R/bench/m5_ultra_256gb/nvfp4_multirow_reuse/smoke.py` | `cac89f54ce7ded33e09b7b8e6bfa61caa03af1ae80f2c36b4b2e658afa0e8213` |
| `F/model/layers.py` | `c2c9936df2a34877edb5e1702152583e54549f69b37d7c93e9996af1e7c20908` |
| `F/model/loader.py` | `e5170e97f1fdc747ae1d8485b7f1bdf086e4aa55930c0cbedfed90e343a13a2a` |
| `F/model/qwen3_5.py` | `dae9694d717a0ebe80c9c54317c65f5d206e4d5695bc898fe0e93eb04a8590ce` |
| `F/engine/compiled_decode.py` | `22111101109b9d9c69e8fd114aa114533fe708e462303d5e72bedd6d217da1bf` |

The next hashes cover **header JSON bytes only**, exactly the length given by the initial 8-byte little-endian safetensors length. They are deliberately not whole-shard payload hashes; the existing model manifest binds payloads for the real qualifier.

| Header path under M | JSON header bytes | Header JSON SHA256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 86,450 | `5ee4fe36fedff37be5707fae8d414da5f4fbf11f23c74dd0c3a97d7384f497e7` |
| `model-00002-of-00003.safetensors` | 68,557 | `8f93becf8c7619733122018894d054a4c3e7decf382ce8cf4df86237ed2988ab` |
| `model-00003-of-00003.safetensors` | 58,133 | `910ede76e14d0da40deb3bca75126c4a4989641e2f664c927adf58473e2cc249` |
