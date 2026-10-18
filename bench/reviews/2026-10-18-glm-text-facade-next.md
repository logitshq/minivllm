# GLM-5.3-Flash: smallest text reference facade and next C2 boundary

2026-10-18, source-only. No runtime import, tests, model payload reads, hashes,
service query or device operation. This is an official-vendor reference design,
not mini engine architecture support or a mini engine performance result.

## Minimal useful entry

After the existing controlled load probe succeeds, borrow the **already loaded
objects** at `load_probe.py:441–455`: `engine._vlm_model`, its
`language_model`, `engine._engine.engine.scheduler.model` (the actual
`VLMModelAdapter`), and the original `_mlx_executor`. Do not instantiate another
LanguageModel or reload the checkpoint. Run all tensor/cache calls through that
single executor and the scheduler's actual MLX stream, after its queues/UID maps
have drained. A facade must release its references before the existing owner
shuts down the pool; it does not own or upgrade the runtime.

The useful API fits one small module:

```text
bind_loaded(engine, expected_identity) -> Reference
new_request(prompt_ids) -> Session(cache=actual_adapter.make_cache(), consumed=0)
prefill_prefix(session, ids_without_last_prompt_token) -> updated session
step(session, pending_ids[B,1]) -> full_logits[B,1,154880], updated session
snapshot(session) -> explicit typed cache fields + full backing tensors
retire(session) -> drop only that request's cache/references
```

`step` calls the actual adapter `scheduler.model(ids, cache=cache)`; its text
branch calls `language_model(ids, cache=cache)` and unwraps `.logits`
(`omlx/models/vlm.py:585–724`). `LanguageModel.__call__` is the original
embedding → 45 layers → mHC mean → RMSNorm → original LM projection
(`vendor/.../glm5_next/language.py:1880–1942,1998–2018`). No custom layer loop,
quantized-linear replacement, mHC materialization hook or compiler is needed.

Match the **actual call boundary**, not merely the prompt string: official
external prefill leaves the last prompt ID for generation
(`omlx/scheduler.py:4001–4012`), calls `_omlx_prefill` and evaluates cache state
(`:4166–4181`). Prefix logits can remain unused/lazy; the pending last prompt ID
is processed by the first true step. Record the actual cache classes at this
handoff: `_patched_merge_caches` can preserve supported singleton caches and
merge other leaves (`:936–994`). Do not blindly turn every C1 cache into a batch
cache or call the whole prompt as one block and label it the same oracle.

For the first short reference keep the probe's plain completion tokens,
`encode(add_special_tokens=False)`, cap8 then cap32, normal EOS
`[154820,154827,154829]`, target-only/cacheOFF and context≤2048. The independent
reference supplies its own greedy next ID. When EOS is sampled, retain it as raw
terminal output and exclude it from committed count; do not feed it into another
step. A token sampled from logits is pending, not yet part of committed cache.
An extra observer that evaluates/saves full logits/cache is a diagnostic, never
a timing run. Match adapter/sampler/suppression policy to the actual engine
before claiming that direct-facade greedy IDs reproduce the EnginePool request.

## Text execution is not text-only loading

The current wrapper constructs `VisionModel` whenever `vision_config` exists
(`glm5_next.py:10–20`), remaps/loads vision parameters (`:143–178`), and the
normal loader runs strict `model.load_weights` followed by
`mx.eval(model.parameters())` (`mlx_vlm/utils.py:1180–1192`). The GLM ordinary
route supplies neither lazy nor a vision-removal option
(`omlx/engine/vlm.py:2154–2171`). `pixel_values=None` only bypasses image
encoding (`glm5_next.py:82–94`). The facade should retain the same loaded wrapper
and call its text adapter. It does **not** save the roughly 1.05 GiB vision
payload, establish safe physical peak, or qualify a text-only loader.

## Cache fields that cannot be dropped

The actual cache family is mixed: KDA/latent-KV/CacheList come from
`mlx_vlm.models.cache`; PoolingCache is injected into `mlx_lm.models.cache` from
**`omlx/patches/deepseek_v4/cache_extras.py`**
(`deepseek_v4/__init__.py:50–74`). The latter even rewrites `__module__`.
Record actual source paths/class identity, not only the displayed module name.

| Component | Keep at a request boundary |
| --- | --- |
| 34 KDA leaves | `cache[0]` convolution tail and `cache[1]` recurrent state, original shape/dtype; actual `_left_padding`, `_left_padding_advance`, `_lengths`, `_lengths_advance`, `metadata_revision`, plus explicit inactive speculation status. `ArraysCache.state` only returns the two tensors (`mlx_vlm/models/cache.py:838–876,1193–1208,1264–1284`). `advance(N)` advances mask bookkeeping, **not a persistent scalar token offset**. Track consumed token counts separately. |
| 11 DSA latent-KV leaves | Full `keys` backing, zero-width `values`, logical `offset`, capacity/step. A batch adds `_idx`, per-row `offset`, `left_padding` and transient `_right_padding`. `KVCache.state` and `BatchKVCache.state` slice to the logical prefix and therefore are not full-backing snapshots (`:500–550,1601–1641`). |
| 11 DSA pooling leaves | `ratio`, `buf_kv`, `buf_gate`, `remainder`, `_pool_buf`, `_pool_len` (batch: `_pool_extent`, `_pool_lengths`, `_processed`, `_lengths`), previous-window tensors and batch `_prev_valid`; record undo/undo-chain policy separately. `state`/`meta_state` are persistence APIs, not an exact live clone. Batch state restore explicitly clears previous-window carry (`cache_extras.py:35–68,281–319,431–465,723–749`). No rollback qualification is implied. |
| mHC | No persistent per-request mHC cache exists in this model. `_HCDeferred` is **within one forward**: `y,residual,post,comb,mm` (`language.py:166–193`). Let the original layer loop consume it; do not store a fabricated cross-token mHC state or force materialization between layers. Model-owned compiled FFN/projection caches are separate shared model state and must not be reset per request. |

For the first implementation avoid restore/clone altogether: run fresh request
caches sequentially and save diagnostic snapshots. When cloning becomes needed,
copy explicit tensors and metadata under a fixed typed adapter; neither
`CacheList.state` nor generic `deepcopy` is evidence of full state preservation.

## Preserve existing asynchronous prefill policy

The vendor model uses `LayerPipeline(on_evaluated=mx.clear_cache,lazy_last=True)`
for S≥256. Depth1 queues at most two layers, waits the older result and clears
completed allocator cache; its last held output stays lazy
(`language.py:1902–1939`, `omlx/utils/layer_pipeline.py:35–82`). For B1/S≤8 it
starts existing async evaluation every eight layers and uses the deferred mHC
route only for B1/S1. Keep these calls and the engine stream unchanged. Do not
replace them with whole-model compile, per-layer diagnostic eval, or a guessed
prefill chunk size. Scheduler adaptive GLM prefill geometry is an additional
runtime-selected policy (`scheduler.py:1629–1654,2010–2026`); the initial ≤128-ID
probe does not exercise long-prefill memory behavior.

## C1 first, then a genuine C2 reference

1. After official load success, collect one actual engine C1 prompt/step call
   sequence, full logits and typed cache snapshots at the prefix handoff plus
   three greedy steps. Replay the **same call/cache layout** through the facade
   with fresh caches on the same immutable model. Compare full original-dtype
   logits and all meaningful state/metadata; archive first differences. This
   only validates the facade against the installed reference.
2. C2 begins with two separately prefetched, short equal-length prompts, merged
   using the official cache methods, two real rows and three steps. Then test
   unequal histories and C2→C1 retirement separately. `BatchPoolingCache` rejects
   nonzero left padding (`cache_extras.py:436–440,506–510`), so a Qwen-style generic
   left-padded prefill is not a legal shortcut. Unequal KV histories can use the
   vendor's merge metadata; pool history is tracked by its own per-row lengths.
3. C2 is a separate numerical domain: B1 fused KDA requires B==1/no lengths or
   mask (`language.py:554–625`); B1 mHC/compiled FFN and FP32 MoE-router routes
   also change at B2 (`:1566–1598,1783–1819,1917–1937`). Its reference is the
   **actual official B2 path**. A rowwise C1 comparison is a useful diagnostic,
   not a promised bit-equivalence gate that the vendor has already passed.

Small additional source closure needed for this facade: actual VLM cache,
`cache_extras`, LayerPipeline, vendor `linear/gated_delta/hc_prefill`, its
decode kernels and imported MoE/DSA helpers, plus any scheduler merge/sampling
seam actually invoked. The probe's current REQUIRED_SOURCES is not that whole
transitive arithmetic closure. No new lifecycle is needed: the same external
cold/physical/reap owner admits the fresh diagnostic child. A successful facade
remains an official reference, not a mini engine or cross-model Pareto point.
