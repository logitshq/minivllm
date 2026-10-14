# minivllm — architecture contract

> 这份文档是所有组件的接口契约（英文，给并行开发的 agent 看）。教程正文在 `docs/tutorial/`（中文）。
> 改接口必须先改这里。

## 0. Goal and ground truth

- A vLLM-free inference engine (pure PyTorch + Triton + `flash-linear-attention` kernels) that serves
  `unsloth/Qwen3.8-27B-NVFP4` on ONE RTX 5090 (31.4 GiB, sm_120) and matches or beats vLLM on:
  ```
  vllm bench serve --model <same ckpt> --dataset-name random --random-input-len 1024 \
      --random-output-len 1024 --num-prompts 100 --max-concurrency 32
  ```
  vLLM baseline recipe for one 5090: `vllm serve unsloth/Qwen3.8-27B-NVFP4 --enforce-eager --max-model-len 32768 --kv-cache-dtype fp8`.
- Then add DFlash2 speculative decoding (`incoai/Qwen3.8-27B-DFlash2`).
- Design follows vLLM **V1**: unified scheduler (no prefill/decode distinction, token budget per step, chunked prefill,
  mixed batches), paged KV cache with a block pool, request-level recurrent-state slots for linear attention,
  one attention kernel for every query length, CUDA graphs for decode steps.
- Dev model with the same architecture: `Qwen/Qwen3.5-0.8B` (bf16, 24 layers, hidden 1024). Every GPU test runs
  on it first; the 27B is only loaded for end-to-end and benchmarks.

Hard rules:
- No `import vllm` anywhere in `minivllm/`. Allowed deps: torch, triton, fla (`flash-linear-attention`),
  transformers (config + tokenizer only), safetensors, numpy, xxhash, fastapi, uvicorn.
- Every module ships with a test in `tests/` that passes on the GPU box via
  `ssh surf@surf-gpu 'cd ~/minivllm && .venv/bin/pytest tests/test_<x>.py -q'`.
- Every component author writes `docs/notes/<component>.md`: what was built, why this design, the alternatives
  rejected, numbers measured, pitfalls hit. This is the raw material for the tutorial.
- Minimal code. No feature without a consumer in this plan. No abstract base classes "for later".

## 1. Target model (Qwen3.5 architecture; Qwen3.8-27B uses it)

From `config.json` → `text_config` (HF `Qwen3_5TextConfig`):

| field | 27B | 0.8B |
|---|---|---|
| hidden_size | 5120 | 1024 |
| num_hidden_layers | 64 | 24 |
| layer_types | `full_attention_interval=4`: layers 3,7,11,… are `full_attention`, others `linear_attention` | same |
| full attn: num_attention_heads / num_key_value_heads / head_dim | 24 / 4 / 256 | 8 / 2 / 256 |
| attn_output_gate | True (q_proj outputs `[heads, head_dim*2]` = q and gate, chunked per head) | True |
| rope | theta 1e7, `partial_rotary_factor=0.25` → rotary_dim = 64 of 256; `mrope_section=[11,11,10]`, `mrope_interleaved=True`; text-only ⇒ all three position streams are equal ⇒ plain RoPE on the first 64 dims with `inv_freq = 1/theta^(arange(0,64,2)/64)` (verify numerically vs HF!) | same |
| GDN: linear_num_key_heads / linear_num_value_heads / key_head_dim / value_head_dim / conv kernel | 16 / 48 / 128 / 128 / 4 | 16 / 16 / 128 / 128 / 4 |
| intermediate_size | 17408 | 3584 |
| vocab_size | 248320 | 248320 |
| tie_word_embeddings | False | True |
| rms_norm_eps | 1e-6 | 1e-6 |

Layer structure (HF `Qwen3_5DecoderLayer`): `x = x + mixer(input_layernorm(x)); x = x + mlp(post_attention_layernorm(x))`.
RMSNorm is **zero-centered**: `y = x * rsqrt(mean(x²)+eps) * (1 + weight)` computed in fp32, cast back.
MLP: `down(silu(gate(x)) * up(x))`.

Full attention (`Qwen3_5Attention`): `q, gate = q_proj(x).view(T, H, 2*D).chunk(2, -1)`; `q = q_norm(q)`, `k = k_norm(k)` (per-head RMSNorm, zero-centered, dim 256); RoPE on first 64 dims of q and k; attention (GQA, causal, scale D^-0.5); `o = o * sigmoid(gate)`; `o_proj`.

Gated DeltaNet (`Qwen3_5GatedDeltaNet`), per token:
```
qkv = in_proj_qkv(x)            # [T, 2*key_dim + value_dim] = [T, 2*2048 + 6144] (27B)
z   = in_proj_z(x)              # [T, value_dim]
b   = in_proj_b(x); a = in_proj_a(x)      # [T, num_v_heads]
qkv = silu(causal_conv1d(qkv, conv1d.weight[conv_dim,1,4]))   # depthwise, kernel 4, carries conv state of last 3 tokens
q, k, v = split(qkv) -> [T, Hk, 128], [T, Hk, 128], [T, Hv, 128]
beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)    # fp32, per (token, v-head)
q, k repeated to Hv heads (repeat_interleave by Hv/Hk); q, k L2-normalised inside the kernel (use_qk_l2norm_in_kernel=True)
S_t = S_{t-1} * exp(g_t);  v_t' = v_t - k_t^T S_{t-1} ... (gated delta rule; state S [Hv, 128, 128] fp32 per sequence)
o = norm(o, z) = rmsnorm(o) * weight * silu(z)   # RMSNormGated (weight is plain, NOT zero-centered), per head of 128
out_proj(o)
```
Reference implementations: HF `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`
(`transformers/models/qwen3_5/modeling_qwen3_5.py`, a copy is in `ref/hf/` on the dev machine) and
`fla.ops.gated_delta_rule.{chunk_gated_delta_rule, fused_recurrent_gated_delta_rule}`.

Multimodal parts (`model.visual.*`, `mtp.*`) are never loaded. Input is text only.

## 2. Checkpoint formats

Weight names (27B, compressed-tensors; the 0.8B is plain bf16 with the same names minus scales):
```
model.language_model.embed_tokens.weight                      BF16 [V, hidden]
model.language_model.layers.N.input_layernorm.weight          BF16
model.language_model.layers.N.post_attention_layernorm.weight BF16
model.language_model.layers.N.self_attn.{q,k,v,o}_proj.weight F8_E4M3 + .weight_scale BF16 [M,1]   (per-output-channel)
model.language_model.layers.N.self_attn.{q,k}_norm.weight     BF16 [256]
model.language_model.layers.N.self_attn.{k,v}_scale           BF16 [1]        (fp8 KV cache scales; unused while KV is bf16)
model.language_model.layers.N.linear_attn.in_proj_{qkv,z}.weight, out_proj.weight   F8_E4M3 + weight_scale [M,1]
model.language_model.layers.N.linear_attn.in_proj_{a,b}.weight  BF16 [Hv, hidden]
model.language_model.layers.N.linear_attn.conv1d.weight       BF16 [conv_dim, 1, 4]
model.language_model.layers.N.linear_attn.{A_log,dt_bias}     BF16 [Hv]
model.language_model.layers.N.linear_attn.norm.weight         BF16 [128]
model.language_model.layers.N.mlp.{gate,up,down}_proj:
    layers 0..55: weight_packed U8 [M, K/2] (two e2m1 per byte, low nibble first — verify), weight_scale F8_E4M3 [M, K/16],
                  weight_global_scale F32 [1], input_global_scale F32 [1]       (NVFP4, group 16, W4A4)
    layers 56..63: weight F8_E4M3 + weight_scale BF16 [M,1]                     (FP8 per-channel)
model.language_model.norm.weight                              BF16
lm_head.weight F8_E4M3 [V, hidden] + lm_head.weight_scale BF16 [V,1]
```
Dequant conventions (compressed-tensors; the quant author MUST confirm against `compressed_tensors` source and a
numeric check): FP8 per-channel: `W = w_fp8.float() * weight_scale`; activations dynamic per-token:
`x_fp8 = clamp(x / s_x)`, `s_x = amax(|x|, dim=-1)/448`. NVFP4: `W[m,k] = fp4[m,k] * weight_scale[m,k//16].float() / weight_global_scale`;
activations: `s_x_blk = amax(|x_blk|)/6 * input_global_scale` stored as e4m3, `x_fp4 = x / s_x_blk * ...` — follow the
compressed-tensors `NVFP4` definition exactly.

Quant kernels, in order of preference (the quant author tests which works on sm_120 and records it in notes):
1. `torch._scaled_mm` (fp8 rowwise; nvfp4 with `torch.float4_e2m1fn_x2` + `ScalingType.BlockWise1x16` if supported on sm_120),
2. own Triton kernel (`tl.dot_scaled` for fp4; plain fp8 `tl.dot` for fp8),
3. W4A16 fallback: Triton kernel that dequantises fp4 → bf16 tiles on the fly (memory-bound decode is still fine).

## 3. Package layout

```
minivllm/
  config.py              EngineConfig, ModelConfig (parsed from HF config.json; no transformers model classes)
  sampling_params.py     SamplingParams(temperature=1.0, top_p=1.0, top_k=0, max_tokens=..., ignore_eos=False, stop_token_ids=())
  forward_context.py     ForwardContext dataclass (§4) + get/set helpers
  model/
    layers.py            RMSNorm (zero-centered), GatedRMSNorm, RotaryEmbedding (partial), Embedding, LMHead, SiluAndMul
    linear.py            LinearBF16, LinearFP8 (per-channel W, dynamic per-token A), LinearNVFP4 (group 16); `make_linear(scheme, ...)`
    attention.py         FullAttention layer (merged q/k/v GEMM optional; q_norm/k_norm; rope; output gate; kernels.attention)
    gdn.py               GatedDeltaNet layer (conv + delta rule via kernels.gdn / fla chunk; GatedRMSNorm; out_proj)
    qwen3_5.py           Qwen3_5ForCausalLM: forward(input_ids, positions) -> hidden [T, hidden]; compute_logits(hidden) -> [N, V]
    loader.py            load_weights(model, ckpt_dir): safetensors streaming, name mapping, scheme detection per module
  kernels/
    attention.py         store_kvcache(), paged_attention()  (Triton; §5)
    gdn.py               gdn_recurrent(), causal_conv1d_update(), causal_conv1d_varlen() (Triton; §6); chunk path wraps fla
    quant.py             fp8 / nvfp4 activation quant + GEMM wrappers (§2)
  engine/
    request.py           Request, RequestStatus
    kv_cache_manager.py  BlockPool, KVCacheManager (prefix caching optional)
    scheduler.py         Scheduler, SchedulerOutput
    model_runner.py      ModelRunner: allocates caches, builds ForwardContext, runs model + sampler, CUDA graphs
    llm_engine.py        LLMEngine (sync): add_request / step / generate
    async_engine.py      AsyncLLMEngine: background step loop, per-request asyncio.Queue of output tokens
  sampler.py             Sampler(logits, SamplingMetadata) -> token ids (greedy / temperature+top_p+top_k)
  server.py              FastAPI OpenAI-compatible /v1/completions, /v1/chat/completions (SSE streaming), /v1/models
  spec/dflash2.py        (phase D) DFlash2 drafter + verify loop
bench/                   baseline + ours scripts, result logs (json)
tests/                   pytest; GPU tests skip if no CUDA
docs/ARCHITECTURE.md     this file
docs/notes/*.md          per-component design notes (raw material)
docs/tutorial/*.md       the tutorial (Chinese)
```

## 4. ForwardContext — the per-step metadata (the one contract everything shares)

Built once per step by `ModelRunner`, read by attention and GDN layers through `forward_context.get_forward_context()`.
All tensors live on GPU, int32 unless noted. `T` = total scheduled tokens this step, `B` = scheduled requests.

```python
@dataclass
class ForwardContext:
    # ---- token-level (length T, flattened in request order) ----
    positions: Tensor           # int64 [T]  absolute position of each token
    slot_mapping: Tensor        # int32 [T]  physical KV slot = block_id * block_size + offset; -1 → do not store (warmup)
    # ---- request-level (length B). Requests are ordered DECODE-LIKE FIRST, then PREFILL-LIKE ----
    query_start_loc: Tensor     # int32 [B+1] prefix sum of query lengths (q tokens per request this step)
    seq_lens: Tensor            # int32 [B]   context length INCLUDING this step's tokens
    block_table: Tensor         # int32 [B, max_blocks_per_req]  padded with 0 (never read past seq_len)
    state_slots: Tensor         # int32 [B]   GDN state slot per request (conv_state / ssm_state index)
    has_initial_state: Tensor   # bool  [B]   False → treat slot state as zeros (first chunk of a request)
    num_decodes: int            # requests 0..num_decodes-1 are "decode-like": query_len <= DECODE_MAX_Q (1 now, 1+num_spec later)
    num_decode_tokens: int      # = query_start_loc[num_decodes]
    max_query_len: int
    max_seq_len: int
    # ---- sampling ----
    logits_indices: Tensor      # int64 [B]  row in [T] whose hidden state produces this request's logits (last scheduled token)
```
Invariants: `query_start_loc[-1] == T`; `seq_lens[i] - query_len[i] == num_computed_tokens[i]` (tokens already in cache);
decode-like requests always come first so CUDA-graph decode steps are a contiguous prefix and the GDN layer can
split `[0:num_decode_tokens]` (recurrent kernel) from `[num_decode_tokens:]` (chunk kernel).

Chunked prefill: a request with a 1024-token prompt and a 512-token budget appears in two consecutive steps with
query_len 512 each; `seq_lens` = 512 then 1024; `has_initial_state` = False then True (GDN state written at the end of
chunk 1 is read by chunk 2). The attention layer always writes K/V for the new tokens via `slot_mapping` and reads
`0..seq_len` through `block_table`.

## 5. Paged attention kernel (`kernels/attention.py`)

KV cache tensor owned by `ModelRunner`:
`kv_cache = torch.empty(2, num_full_attn_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype=bf16)`;
layer `i` of full attention gets `k_cache = kv_cache[0, i]`, `v_cache = kv_cache[1, i]`. `block_size` default 64
(configurable, kernel must support 16/32/64). Slot `s` addresses `k_cache.view(-1, Hkv, D)[s]`.

```python
def store_kvcache(k: Tensor[T, Hkv, D], v: Tensor[T, Hkv, D], k_cache, v_cache, slot_mapping: Tensor[T]) -> None
def paged_attention(q: Tensor[T, Hq, D], k_cache, v_cache, query_start_loc, seq_lens, block_table,
                    scale: float, max_query_len: int) -> Tensor[T, Hq, D]
```
One Triton kernel for every query length (vLLM "unified attention"): program per (request, q-tile, kv-head);
the GQA group (Hq/Hkv query heads) is packed into the M dimension together with the q tokens of the tile;
online softmax over KV blocks fetched through `block_table`; causal mask with query position
`seq_len - query_len + i`. Must be correct for query_len ∈ {1, 2..8, 512, 1024} and seq_len up to 32768, head_dim 256,
and must not read past `seq_len`. Accuracy test: vs `torch.nn.functional.scaled_dot_product_attention` with an explicit
mask, bf16, atol 2e-2 relative to fp32 reference. Perf test: report µs for (B=32, q_len=1, seq 2048) and
(B=1, q_len=1024, seq 1024) on the 5090; target ≥ 50% of the memory-bandwidth bound for the decode case.
fp8 KV cache is a later option, not in scope now.

## 6. GDN kernels (`kernels/gdn.py`)

State pools owned by `ModelRunner` (per GDN layer index `j`):
```
conv_state[j] : bf16 [num_state_slots, conv_dim, conv_kernel-1]   # last 3 post-projection, pre-conv inputs
ssm_state[j]  : fp32 [num_state_slots, Hv, Dk, Dv]                # 48×128×128×4 B = 3 MiB per slot per layer (27B)
```
`num_state_slots = max_num_seqs` (+ extra for spec decode later). Slot indices come from `ForwardContext.state_slots`.

```python
def causal_conv1d_update(x: Tensor[Td, C], conv_state, state_slots, weight: Tensor[C, K], has_initial_state) -> Tensor[Td, C]
    # decode-like tokens (query_len==1 per request now; generalise to <= DECODE_MAX_Q): shift state, write state, silu(conv)
def causal_conv1d_varlen(x: Tensor[Tp, C], query_start_loc_p, conv_state, state_slots_p, weight, has_initial_state_p) -> Tensor[Tp, C]
    # prefill-like tokens: uses initial state when has_initial_state, writes the final state (last K-1 inputs) per request
def gdn_recurrent(q, k, v: Tensor[Td, Hv, 128], g: Tensor[Td, Hv] fp32, beta: Tensor[Td, Hv] fp32,
                  ssm_state, state_slots, query_start_loc_d, has_initial_state) -> o: Tensor[Td, Hv, 128]
    # Triton: program per (request, head): load S from slot (or zeros), loop over the request's tokens
    # (q,k L2-normalised in-kernel), write o per token, store S back to the slot. Varlen so a request may have 1..8 tokens.
def gdn_chunk(q, k, v, g, beta, query_start_loc_p, initial_state: Tensor[Bp, Hv, Dk, Dv] | None) -> (o, final_state)
    # wraps fla.ops.gated_delta_rule.chunk_gated_delta_rule(cu_seqlens=..., initial_state=..., output_final_state=True,
    # use_qk_l2norm_in_kernel=True); the layer gathers/scatters slot states around it.
```
Tests: vs HF `torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule` (copy the two functions into the test,
they are pure torch) for random inputs; chunk-then-recurrent continuation must equal one chunk call over the whole
sequence; state in slot must match. Report µs for (B=32 decode) on the 5090 with 27B shapes (Hv=48).

## 7. Linear layers (`model/linear.py`, `kernels/quant.py`)

```python
class LinearBase(nn.Module):     # no bias anywhere in this model
    def forward(self, x: Tensor[T, K]) -> Tensor[T, M]
    def load(self, tensors: dict[str, Tensor])   # receives this module's tensors keyed by suffix: "weight", "weight_scale", ...
```
`LinearFP8`: weight fp8 [M,K], scale fp32 [M]; forward = dynamic per-token fp8 quant of x + `torch._scaled_mm` rowwise.
`LinearNVFP4`: packed weight [M,K/2] u8, scales e4m3 [M,K/16], global scales; forward per §2.
Fused projections are allowed when they are a measured win (e.g. q/k/v into one GEMM, gate/up into one GEMM) but the
three MLP projections have different `input_global_scale` values in NVFP4 — only fuse what shares a scheme and input scale.
Tests: each scheme vs dequantised bf16 matmul (rel err < 2e-2 for fp8, < 6e-2 for nvfp4 with random data); perf table for
M ∈ {1, 32, 256, 2048} on the 27B shapes (K=5120, N=17408) → µs and achieved TB/s or TFLOPS, recorded in notes.

## 8. Engine (`engine/`)

`Request`: `req_id: str`, `prompt_token_ids: list[int]`, `output_token_ids: list[int]`, `num_computed_tokens: int`,
`block_ids: list[int]`, `state_slot: int | None`, `sampling: SamplingParams`, `status: WAITING|RUNNING|FINISHED_*`,
properties `num_tokens`, `all_token_ids` (prompt + output), `is_finished`.

`KVCacheManager(num_blocks, block_size, enable_prefix_caching)`: V1 block pool: `free_block_queue` (LRU order; freed
blocks keep their hash until reused), `cached_block_hash_to_block`, ref counts; API
`get_computed_blocks(req) -> (blocks, num_computed_tokens)`, `allocate_slots(req, num_new_tokens) -> list[int] | None`,
`free(req)`, `cache_blocks(req, num_computed_tokens)` after a step. Block hash = xxhash64(parent_hash, token_ids of the block).
Prefix caching is forced off for hybrid (GDN) models — the recurrent state cannot be reused by a prefix hit — and the
engine says so at startup; the tutorial explains why.

`Scheduler(config, kv_cache_manager, num_state_slots)`: V1 unified policy per step:
1. `token_budget = max_num_batched_tokens`; iterate `running` (FCFS): each gets `min(remaining prompt/1 for decode, budget)`
   tokens if blocks can be allocated, else preempt the LAST running request (free its blocks + slot, back to `waiting`
   head, recompute from scratch) and retry.
2. Then `waiting` while `len(running) < max_num_seqs` and budget > 0: needs a free state slot + blocks for the first chunk.
3. Output `SchedulerOutput(scheduled: list[(Request, num_tokens)], finished_req_ids: set[str])` with decode-like requests
   (num_tokens == 1 and prompt fully computed) FIRST. Also a `policy` knob: `mixed` (default, V1) or `separate`
   (never mix prefill with decode in one step — used to keep decode steps CUDA-graphable; benchmark both).
`Scheduler.update(scheduler_output, sampled_token_ids: dict[req_id, int])`: advance `num_computed_tokens`, append tokens
only for requests whose prompt is fully computed, finish on EOS (unless `ignore_eos`), `max_tokens`, `stop_token_ids`,
free resources. CPU-only unit tests: budget respected, chunking, preemption + recompute, slot reuse, finish conditions,
decode-first ordering, a 200-request simulation with random lengths finishes with correct token counts.

`ModelRunner(config, model)`: allocates `kv_cache` and state pools from free memory after loading + a warmup forward
with `max_num_batched_tokens` (mirror nano-vllm/vLLM: `gpu_memory_utilization * total - used - peak`), builds
`ForwardContext` from `SchedulerOutput` (pinned CPU tensors → non_blocking H2D), runs `model.forward` → `compute_logits`
on `logits_indices` → `Sampler`; **CUDA graphs**: capture decode-only steps for batch sizes {1,2,4,8,16,24,32,48,64}
(padded; static input buffers; one shared mempool), replay when `num_decodes == B` and `T == B`; eager otherwise.
`enforce_eager` disables capture.

`LLMEngine`: `add_request(prompt | token_ids, sampling) -> req_id`, `step() -> list[RequestOutput]` (new tokens per
request + finished flag), `generate(prompts, sampling) -> list[str]` convenience for tests.
`AsyncLLMEngine`: one background task running `step()` in a loop (only when there is work; `await asyncio.sleep(0)` between
steps), per-request `asyncio.Queue` delivering `(token_id, finished)`; the step loop must not block the event loop for more
than one model step (run `step()` in the loop thread is fine because it is GPU-bound and releases the GIL inside kernels;
measure, and move to a thread if TTFT suffers).

## 9. Server (`server.py`)

OpenAI-compatible, enough for `vllm bench serve` (backend `vllm`/`openai`, which hits `POST /v1/completions` with
`stream=true`, `stream_options={"include_usage": true}`, `max_tokens`, `temperature`, `ignore_eos` (vLLM extension — honour it),
`model`) and for a chat client (`/v1/chat/completions` applies `chat_template.jinja` via the HF tokenizer). SSE chunks
with `text`/`delta.content`, final `usage` with `prompt_tokens`/`completion_tokens`. `GET /v1/models`. `GET /health`.
Tokenizer: `transformers.AutoTokenizer`; detokenise incrementally (handle byte-fallback / partial unicode: buffer until the
decode output does not end with the replacement char).

## 10. Benchmark protocol (`bench/`)

Same box, nothing else on the GPU (the jupyter kernel must be gone), same checkpoint, same client:
`~/vllm-baseline/.venv/bin/vllm bench serve --backend vllm --model unsloth/Qwen3.8-27B-NVFP4 --host localhost --port <p>
--dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 100 --max-concurrency 32 --save-result`.
Record for each run: request throughput, output token throughput, mean/median/p99 TTFT, TPOT, ITL, plus server-side
GPU memory split (weights / KV / state / graphs). Baseline runs: vLLM eager (recipe), vLLM eager + DFlash2 (if it fits).
Parity = output token throughput ≥ vLLM's within noise (3 runs each). Sanity before any benchmark: greedy outputs of
ours vs vLLM on 5 fixed prompts (first 64 tokens identical or nearly: quantised kernels may differ after a few tokens —
report the match length), and vs HF transformers on the 0.8B (logits max abs diff).

## 11. Phase plan

- A (parallel, independent files): A1 model+loader (bf16 path on 0.8B, eager full-sequence attention + HF torch GDN
  reference for the first test, then switch to kernels) · A2 attention kernel · A3 GDN kernels · A4 quant linears + loader
  for the 27B scheme · A5 engine (request/kv manager/scheduler, CPU tests) · A6 server + async engine against a fake
  engine, tested with `vllm bench serve` from the baseline venv.
- B (sequential): ModelRunner + LLMEngine integration; 0.8B greedy == HF generate; 27B loads within memory; CUDA graphs.
- C: baseline measurements, ours, profile, iterate until parity (loop).
- D: DFlash2 (drafter forward with target hidden states from layers [5,19,33,47,61], block 8, candidate selector top-16,
  two-tap dynamic conv; verify + GDN state commit = recompute recurrent over accepted tokens from the saved state).
- E: tutorial in `docs/tutorial/` (Chinese): one chapter per step, written from code + notes.
