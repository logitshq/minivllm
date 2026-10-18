# GLM5 Next native support: minimum boundaries from installed source

Read-only source analysis, 2026-10-18. No package import, model load, GPU call,
test or measured performance result. The target remains the exact archived
GLM-5.3-Flash mixed 4/5/6-bit snapshot, revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e`. The existing controlled official load
probe is still the prerequisite; this note does not claim a successful load.

Sources below are the installed compatibility vendor, not an assumed generic
upstream model API:

- [Model configuration](../m5_ultra_256gb/glm_flash/metadata-v1/config.json): 45
  layers, hidden4096, vocab154880, 34 KDA layers and 11 sparse-attention layers;
  first three FFNs dense, subsequent FFNs MoE, 288 routed experts/top8 and one
  shared expert. KDA has 64×128 heads, conv4; MLA uses 512 latent KV, 64 heads,
  q-LoRA1536, NoPE256; index pool4/topk2048; mHC multiplier4.
- Installed source root:
  `.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/`.
  Relative line references below are within that root. These source paths are
  covered by the existing explicit probe inventory only to the extent listed in
  `glm_flash/load_probe.py:63–72`; transitive imports must not be called a complete
  independently verified closure without a new inventory.

| Minimum module boundary | Concrete existing source / required semantics |
| --- | --- |
| **GLM-specific loader/config/quantization** | `language.py:2020–2087` drops MTP, applies DeepSeek sanitization, remaps mHC/forget-gate keys, concatenates q/k/v conv weights, permutes conv axes and preserves sensitive FP32 parameters. `:231–241`/`:2094–2106` keep router/mHC/A_log/dt_bias FP32; router gates are not quantized and indexers use 8-bit. Config has per-path 4/5/6-bit affine quantization. Current Qwen `model/loader.py:37–45` casts most floats to BF16 and subtracts zero-centered norms: it cannot be reused unchanged. |
| **Explicit mixed cache contract** | `language.py:2108–2123` creates KDA `ArraysCache(2)` versus sparse `CacheList(KVCache(), PoolingCache(4))`. KDA advances offset and stores conv+recurrent state; sparse attention also owns pooled index state. Qwen's K/V-or-conv/S cache snapshot/drop/rebuild methods do not cover this schema. Establish full offset/length/pool snapshots before adding batching/ahead. |
| **KDA attention** | `language.py:397–430,554–729`: q/k/v plus two-stage per-channel forget/output gates, conv4, lower bound−5 and recurrent state update. Fused path is B1, S1–8, no mask/lengths, supported quant groups. B>1 has different official fallback. This is not Qwen's scalar gate/48-head GDN; existing GDN kernels/qualification cannot be inherited by changing dimensions. |
| **Sparse MLA + indexer** | `language.py:731–1080,1083–1501`: projected latent KV, pooled index scores/topk, dense-prefix bypass and gathered sparse attention. NoPE is explicitly enforced (`:1095–1104`). Preserve pool tail, index ordering and mask semantics; test the dense→sparse/topk boundary independently. |
| **MoE + mHC residual block** | `language.py:1555–1753,1755–1941`: expert selection, shared expert contribution, clamped SwiGLU and multi-stream residual mixing are essential model arithmetic. The B1 short-block router uses a specific FP32 GEMV route (`:1574–1576` explains why block GEMM can select different experts). Changing batch/history may change routing before final logits. |
| **Text-only model/runner facade** | `glm5_next.py:10–20,82–94,127–145` separates vision from text embeddings; `language.py:1988–2018` exposes `LanguageModel(...,cache=...)→LanguageModelOutput(logits)`. Start with one sequence, explicit normal-EOS greedy tokens and untouched official cache classes. This is an isolated official-reference adapter, not yet a native reimplementation. |

The smallest first implementation should be a **text-only reference facade +
GLM cache snapshot contract**, with one complete prefill and three true greedy
steps. Record full logits, selected experts/weights, all KDA conv/recurrent
states, sparse latent KV/index-pool state and offsets. Compare independently
continued states; do not feed reference token IDs into a purported independent
candidate rollout. The actual config includes a vision section; text-only requests
do not prove that the official wrapper skipped constructing/loading its vision
parameters. Any omission needs an explicit loader-scope qualification. Only after
loader/cache equality should one module be ported,
keeping official quantized linear arithmetic first. MoE routing and mHC are
higher correctness risks than ordinary affine projection plumbing.

A further resource/scheduling boundary matters before any whole-model compiler:
`language.py:1902–1938` uses a bounded `LayerPipeline` for prefill≥256, clears the
allocator cache after completed layers, and issues existing `mx.async_eval`
every eight layers for B1 short decode/verify. `_HCDeferred` can carry part of
one layer's residual work into the next (`:166–218,1833–1863`). Removing these
boundaries can change load/prefill peak memory and overlap even if equations
look identical. A native adapter must preserve or explicitly requalify them;
no new eval/sync should be inserted merely for profiling.

Recommendation: complete the strict dual-monitor official cold-load diagnostic
first. Then build the small reference facade/state observer, not a full serving
engine. Native scheduler/continuous batching, speculation, compiled graphs,
quant conversion and expert offload are separate later qualifications. This
source analysis predicts neither successful residency nor a speedup.
