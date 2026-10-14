# Engine core (A5): Request, KVCacheManager, Scheduler — CPU only

Files: `minivllm/sampling_params.py` (23 non-blank lines), `minivllm/engine/request.py` (46),
`minivllm/engine/kv_cache_manager.py` (137), `minivllm/engine/scheduler.py` (183) — 389 non-blank lines of
pure Python, no torch. Tests: `tests/test_engine_core.py` (431 lines, 15 tests, 0.15 s).
Run: `ssh surf@surf-gpu 'cd ~/minivllm && .venv/bin/pytest tests/test_engine_core.py -q'`.

## 1. What it is

The engine core is the part of an inference engine that never touches the GPU: it decides, once per step,
*which tokens of which requests the model computes next*, and where in the paged KV cache they go. The
design is vLLM V1's, which collapsed the old "prefill phase vs decode phase" into a single rule:

> Every running request has `num_tokens - num_computed_tokens` tokens that are not in the cache yet. Give each
> one `min(that, remaining budget)` tokens, oldest first. A decode is just the case where that number is 1.

Everything else (chunked prefill, mixed batches, preemption) falls out of that one rule.

### Request (`request.py`)

A plain dataclass: `prompt_token_ids`, `output_token_ids`, `num_computed_tokens`, `block_ids`, `state_slot`,
`status`. Prompt + output are **one sequence**: after a preemption `num_computed_tokens` goes back to 0 and the
next "prefill" embeds prompt *and* the outputs generated so far. `RequestStatus` values double as the
OpenAI `finish_reason` strings (`"stop"`, `"length"`, `"abort"`) so the engine front door needs no mapping.

`token_ids_slice(start, end)` returns `all_token_ids[start:end]` without concatenating the two lists; the
decode hot path asks for exactly one token per request per step and must not copy a 2k-token list to get it.

### KVCacheManager (`kv_cache_manager.py`)

`BlockPool`: `num_blocks` `KVCacheBlock(block_id, ref_cnt, block_hash)` objects plus

- `free_queue`: an `OrderedDict[block_id, block]` used as an LRU queue. Head = least recently freed = evicted
  first. vLLM hand-rolls a doubly linked list because a prefix-cache hit must remove a block *from the middle*
  of the free queue in O(1); an `OrderedDict` gives `popitem(last=False)`, `del d[k]` and append, all O(1),
  in one line.
- `cached_block_hash_to_block`: hash → block for every block that still carries a hash.

**Lazy eviction** (the V1 trick): `free()` only decrements `ref_cnt` and pushes the block to the tail of the
free queue; the hash stays. The content is only lost when `get_new_blocks()` pops that block for someone else
(`del cached[hash]`, `hash = None`). So a finished request's prefix stays hittable for exactly as long as
the memory is not needed — no explicit cache size, no eviction policy beyond LRU-of-free-blocks.

Block hash = `xxhash64(parent_hash || token_ids_of_block)` with `parent_hash = 0` for the first block. The
chain means a hit on block *k* implies blocks *0..k-1* are identical too, so a lookup stops at the first miss.
Only **full** blocks are hashed (a partial block's content changes as tokens arrive).

API as in ARCHITECTURE §8:
- `get_computed_blocks(req) → (block_ids, num_tokens)`: longest hashed prefix present in the cache. If *all*
  blocks of the sequence hit, the last one is dropped: at least one token must be computed to produce logits,
  and `num_computed_tokens` must stay a multiple of the block size (V1 does the same).
- `allocate_slots(req, num_new_tokens, new_computed_blocks=())`: make `req.block_ids` cover
  `req.num_computed_tokens + len(hits)*block_size + num_new_tokens` tokens. Returns the new block ids (possibly
  `[]`: a decode token that still fits in the last block) or `None` when the pool cannot satisfy it — and then
  **nothing changes**, so the scheduler can preempt and retry. Hit blocks that are currently free (ref 0) are
  about to leave the free queue, so they are subtracted from the free count before the check.
- `cache_blocks(req, num_computed_tokens)` after a step hashes the newly full blocks. Two requests computing
  the same block in one step produce one duplicate; the first registration wins, the other block just never
  gets a hash (vLLM keeps a dict-of-dicts for this; not worth it).
- `free(req)`: ref-- on every block, pushed to the free queue in **reverse** order so the tail block of a
  request (least likely to be anyone's prefix) is evicted first.

**Prefix caching is forced off for hybrid models** (`is_hybrid=True` logs a warning and sets
`enable_prefix_caching=False`). Reason: a prefix hit restores the K/V of the full-attention layers only. The
48 GDN layers keep a recurrent state `S [Hv,128,128]` *per request* that summarises the whole prefix; it is
not stored per block, so nothing can be reused and the hit would start the request with
`has_initial_state=True` on a slot that holds another request's state. The scheduler snapshot shows the
exact failure: `has_initial_state = num_computed_tokens > 0`, which a cache hit makes true without any state
having been written. (vLLM gates prefix caching on hybrid models the same way; its "mamba prefix caching"
work stores state per block boundary — out of scope.) The cache code still ships because the tutorial
chapter needs it, and it is ~60 lines.

### Scheduler (`scheduler.py`)

`Scheduler(config, kv, num_state_slots, eos_token_id=...)`. `config` is `EngineConfig`
(`max_num_batched_tokens`, `max_num_seqs`, `max_model_len`, `scheduler_policy`); `eos_token_id` comes from
`ModelConfig` and is an int, a list (Qwen's `generation_config` lists `<|im_end|>` and `<|endoftext|>`) or
None.

`schedule()` per step:

1. **running, FCFS**: `remaining = num_tokens - num_computed`; `num_new = min(remaining, budget)`;
   `kv.allocate_slots(req, num_new)`. On `None`: pop the **last** running request (the youngest), free its
   blocks and state slot, `num_computed_tokens = 0`, push it to the **head** of `waiting`; retry. If the
   victim is the request itself, stop — nothing younger is left to take blocks from.
2. **waiting, FCFS**: while `len(running) < max_num_seqs`, a state slot is free, budget > 0 and the step did
   not preempt anything: `get_computed_blocks` → `allocate_slots(req, min(num_tokens - hits, budget), hits)`;
   failure = stop (head-of-line blocking is deliberate; it is what makes FCFS starvation-free).
3. Stable-sort the step so **decode-like requests come first** (`num_scheduled_tokens == 1 and
   samples_token`), FCFS inside each group. ForwardContext needs that contiguity: the GDN layer splits
   `[0:num_decode_tokens]` for the recurrent kernel, and a CUDA-graph decode step is `num_decodes == B`.

`SchedulerOutput.scheduled` is a list of `ScheduledRequest` **snapshots** (`req_id, num_scheduled_tokens,
num_computed_tokens, block_ids, state_slot, has_initial_state, token_ids, sampling, samples_token`), not
`Request` objects: `update()` mutates the requests, and the runner may still be reading the snapshot (async
engine). Everything in §4 ForwardContext derives from the snapshot: positions =
`range(num_computed, num_computed + num_scheduled)`, `seq_len = num_computed + num_scheduled`,
`slot_mapping[i] = block_ids[pos // bs] * bs + pos % bs`, `logits_indices` = last token of each request with
`samples_token`. `finished_req_ids` are the ids finished in the previous `update()` (vLLM semantics: the
runner learns one step later, in the same message it needs anyway).

`update(out, sampled_token_ids)`: for every scheduled request `num_computed += num_scheduled`;
`cache_blocks`; if `samples_token`, append the token and check, in this order: EOS (unless `ignore_eos`) or a
stop token → `FINISHED_STOPPED`; `len(output) >= max_tokens` or `num_tokens >= max_model_len` →
`FINISHED_LENGTH_CAPPED`. Finished requests are released (blocks + slot) and removed from `requests`.
Returns the requests that sampled this step — exactly the set `LLMEngine.step()` must report.

`abort(req_id)` exists because A6's `LLMEngine.abort` docstring needs it (client disconnect). `update()`
skips a request aborted while its step was in flight.

**`scheduler_policy`**: `"mixed"` is the V1 rule above. `"separate"` runs the same `_schedule()` twice: first
with decodes masked out (chunked prefills of running requests + new admissions); if that scheduled nothing,
again with prefills masked out. So a step is either all-prefill or all-decode, and prefill has priority
(vLLM V0 / nano-vllm behaviour). The point is phase B: a decode-only step with `T == B` replays a CUDA graph;
one prefill chunk in the batch forces the eager path for all 32 decoders. Phase C benchmarks both.

## 2. Alternatives rejected

- **Allocate all blocks of a prompt at admission** (nano-vllm `can_allocate`): no chunked prefill, and a
  4k-token prompt would block for the whole pool at once. V1 allocates per chunk; the cost is the thrash case
  in pitfall 1, which only appears when the pool is nearly full.
- **`Request` holding its block hashes** (vLLM does): keeps the request dataclass to the §8 fields; the hash
  chain lives in the manager, keyed by `req_id`, and is dropped on `free()`. Rehashing after a preemption is
  0.7 µs per block.
- **Hand-rolled doubly linked free list** (vLLM `FreeKVCacheBlockQueue`): `OrderedDict` does the same.
- **Reserving block 0 as the null block** (vLLM pads block tables with it): §4 says the block table is padded
  with 0 and never read past `seq_len`, so block 0 can be a normal block.
- **`dict[hash, dict[block_id, block]]`** for duplicate blocks: first-wins is enough; the duplicate is just
  never hashed.
- **Alternating prefill/decode steps** for `separate`: prefill-priority is what V0 did and is simpler;
  starvation is bounded because `max_num_seqs` caps how many prefills can queue up.
- **Returning `(Request, num_tokens)` tuples** (the literal §8 text): the snapshot carries the same `Request`
  data plus what the runner needs, and is safe under concurrent `update()`.

## 3. Numbers

All measured on the Mac (Python 3.12, `../a5-venv`) with `../a5_bench.py` in the scratchpad; the box numbers
are in the last line of each entry once the venv finished installing (same script, `.venv/bin/python`).

| what | command | result |
|---|---|---|
| `schedule()+update()`, 32 decode requests, 1024-token context, bs 64, no prefix caching | `a5_bench.py` | median **44 µs**, p99 59 µs (Mac) |
| same with prefix caching on (hash per newly-full block) | `a5_bench.py` | median 52 µs, p99 70 µs (Mac) |
| prefill step: two 1024-token chunks, 2048 budget | `a5_bench.py` | median 57 µs (Mac) |
| `hash_block_tokens` on a 64-token block | `a5_bench.py` | 0.73 µs (Mac) |
| 200-request simulation, `mixed` | `pytest -s -k simulation` | 422 steps, 36 078 scheduled tokens for 34 045 needed (6% recompute), 14 preemptions, 177 mixed steps |
| same, `separate` | | 603 steps (+43%), 38 538 tokens, 26 preemptions, 0 mixed steps |
| same, `mixed` + prefix caching (30% of prompts share a 48-token prefix) | | 415 steps, **31 459** scheduled tokens (7.6% below the 34 045 the model must see without a cache), 4 preemptions |
| whole test file | `pytest tests/test_engine_core.py -q` | 15 passed in 0.15 s |

For scale: one 27B decode step for 32 requests is on the order of 10–20 ms on the 5090 (memory-bound:
~13.5 GB of NVFP4/FP8 weights plus up to 4 GB of KV reads at 1.8 TB/s), so 44 µs of scheduling is well under
1% — no need to optimise this further in Python.

Simulation parameters: 200 prompts of 1–300 tokens, `max_tokens` 1–120, 30% `ignore_eos`, 30% with a stop
token, 0–5 arrivals per step, pool 200 blocks × 16 = 3 200 token slots for up to 16 × 420 tokens of demand
(so preemption must happen), budget 256, `max_model_len` 512.

## 4. Pitfalls hit

1. **Livelock in `separate` mode (and a latent one in `mixed`).** The 200-request simulation hung at
   20 000 steps. A per-step trace (`a5_debug.py`: print `[(req_id, num_scheduled, num_computed, is_decode)]`,
   free blocks and the preemption counter) showed the same line every step:
   `[('r34', 256, 0, False)], running 15, waiting 165, free 1, preemptions +1`. r34 (300-token prompt) was
   admitted with its first 256-token chunk, then next step needed 16 more blocks, got none, preempted *itself*
   (it was the youngest), its blocks went back to the pool — and the waiting loop of the **same pass**
   re-admitted it from scratch with those very blocks. The pass was never empty, so the decode pass never ran,
   so the 15 decoders never finished, so the pool never drained. Fix (it is a V1 rule I had skipped): no
   waiting admission in a step that preempted. After the fix: 603 steps, 26 preemptions.
   The thrash itself (admit chunk 1 → fail on chunk 2 → self-preempt) remains possible in any policy when the
   pool is nearly full; it costs one wasted chunk per cycle but always progresses because the older requests
   keep decoding. The benchmark config (27B, ~300k token slots, 32 × 2k) never gets near it.
2. **`finished_req_ids` lost in `separate` mode.** `_schedule()` originally moved `_finished_req_ids` into
   its output; `separate` calls `_schedule()` twice and discarded the first output. Caught by asserting
   that every request id appears in exactly one `finished_req_ids` over a run; moved the hand-off to
   `schedule()`.
3. **Fake-model off-by-one.** `fake_next_token(seq)` emits EOS when `len(seq) % 37 == 0`, where `seq` is the
   sequence *before* the new token — a 36-token prompt does not give EOS first, a 37-token one does. Three
   assertions in `test_finish_conditions` were written for the wrong length; the engine was right every time.
   Lesson: derive expectations from the reference generator (`expected_output`) instead of hand-counting.
4. **EOS beats `max_tokens`.** With `max_tokens=1` and EOS as the first token the status is `stop`, not
   `length` (vLLM reports the same). The test now sets `ignore_eos` on the `max_tokens` case.
5. **Ref counts must be inspected between `schedule()` and `update()`.** A test that checked `ref_cnt == 2`
   on shared blocks after `run_step()` saw 0, because both one-token requests had finished and freed inside
   `update()`.
6. **`allocate_slots` reads `req.num_computed_tokens`.** A unit test called it three times on one request
   without advancing `num_computed_tokens` and got `[]` where it expected a new block. In the engine the
   scheduler advances the count in `update()`; the manager never guesses.
7. **Shared tree.** `sampling_params.py` was overwritten mid-task by the server agent (same fields and
   defaults, no validation) — the test for `SamplingParams(temperature=-1)` suddenly stopped raising. Merged.
   `config.py` appeared later with `scheduler_policy` (not `policy`) and `eos_token_id` on `ModelConfig`;
   the scheduler was renamed to match and takes `eos_token_id` explicitly.

## 5. Integration notes for phase B

```python
mc, cfg = ModelConfig.from_pretrained(cfg.model), EngineConfig(...)
kv = KVCacheManager(num_blocks, cfg.block_size, cfg.enable_prefix_caching, is_hybrid=bool(mc.gdn_layers))
sched = Scheduler(cfg, kv, num_state_slots=cfg.max_num_seqs, eos_token_id=mc.eos_token_id)

def step():
    out = sched.schedule()                       # [] scheduled → nothing to do
    sampled = runner.execute(out)                # dict[req_id, token] for every s with s.samples_token
    return [RequestOutput(r.req_id, [r.output_token_ids[-1]], r.is_finished,
                          r.status.value if r.is_finished else None) for r in sched.update(out, sampled)]
```
- `num_blocks * block_size` must be ≥ `max_model_len`; `add_request` rejects prompts longer than
  `max_model_len` but a request that can never fit the pool would sit at the head of `waiting` forever.
- `ScheduledRequest.block_ids` covers `num_computed + num_scheduled` tokens; pad the block table with 0.
- `ScheduledRequest.state_slot` is reused as soon as a request finishes; `has_initial_state=False` on the
  first chunk tells the GDN kernels to ignore whatever the slot holds.
- `SchedulerOutput.num_decodes` / `total_tokens` are the `num_decodes` and `T` of ForwardContext.
