# A2 — Triton paged attention (`minivllm/kernels/attention.py`)

Status: DRAFT — numbers section pending (environment still installing at time of writing).

## What was built

Two Triton kernels behind the two functions of ARCHITECTURE.md §5:

- `store_kvcache(k, v, k_cache, v_cache, slot_mapping)` — one program per token; copies `k[t]`/`v[t]`
  (`[Hkv, D]`, any row/head stride, D contiguous) into slot `slot_mapping[t]` of the flat
  `[num_slots, Hkv, D]` view of each cache. Slot `-1` returns immediately (warmup / padded rows).
- `paged_attention(q, k_cache, v_cache, query_start_loc, seq_lens, block_table, scale, max_query_len)` —
  one kernel for every query length. Grid `(B, cdiv(max_query_len, BLOCK_Q), Hkv)`; a program owns one
  request, one tile of `BLOCK_Q` query tokens and one KV head. Its M dimension holds
  `BLOCK_Q * GROUP` rows: row `m` is query token `tile_q0 + m // GROUP`, query head
  `kv_head * GROUP + m % GROUP` (GROUP = Hq/Hkv = 6 on the 27B, 4 on the 0.8B). The loop walks keys
  `0 .. kv_end` in steps of `BLOCK_N = block_size`, reads the block id from `block_table[req, kv0 // block_size]`,
  loads K as `[D, BLOCK_N]` and V as `[BLOCK_N, D]` straight from the cache (no gather copy), does
  `tl.dot` in bf16 with fp32 accumulators, and runs the usual online softmax in the log2 domain
  (`scale * log2(e)` folded into the QK product, `exp2`). Causal mask: key `j` is visible to row `m`
  iff `j <= seq_len - query_len + q_local(m)`; keys `>= seq_len` are masked in the load (`other=0`) and
  in the score, so nothing past `seq_len` is ever read. `kv_end = min(context_len + tile_q0 + BLOCK_Q, seq_len)`
  stops the loop at the last key any row of the tile can see.

The GQA packing is the whole point of the "unified" design: a decode step (`query_len = 1`) still gives the
tensor cores an M of 6 (or 4) real rows per KV head, and K/V are read from HBM once per KV head instead of
once per query head.

## Design decisions and why

- **Program per (request, q-tile, kv-head), grid sized by `max_query_len`.** That is what §5 asks for and it
  keeps the launch free of host syncs: the grid comes from `B` (a shape) and the Python int `max_query_len`.
  Programs whose tile starts past their request's `query_len` exit after one load (mixed batches with a
  1024-token chunk and 31 decodes launch 31 × 102 empty programs; that costs a few µs, measured below).
  vLLM's kernel instead flattens tiles over all requests and binary-searches `query_start_loc` per program;
  that saves the empty launches but needs `num_q_tiles` on the host, i.e. a `.tolist()` of query lengths.
- **`BLOCK_N = block_size`.** One `block_table` scalar load per inner iteration and no per-key gather of
  block ids; the contract allows 16/32/64 and all three are legal `tl.dot` N sizes.
- **Config chosen by `(GROUP, max_query_len)`, not autotuned.** `_tile_config` maps the number of real rows
  (`max_query_len * GROUP`) to `BLOCK_M ∈ {16, 32, 64}`; decode steps always have `max_query_len = 1` (or
  `1 + num_spec` later), so every CUDA-graph decode step hits the same compiled binary and no re-tune can ever
  happen on replay. `triton.autotune` keyed on head_dim/block_size only would also have picked the config
  from whichever shape ran first, which is wrong for a kernel that serves both q=1 and q=1024.
- **K loaded transposed by pointer arithmetic** (`offs_d[:, None] + slot_off[None, :]`) rather than
  `tl.trans` on a `[BLOCK_N, D]` tile: same coalesced access (D is the contiguous axis), no layout
  conversion through shared memory.
- **fp32 everything except the two dots.** Scores, running max, running sum and the accumulator are fp32;
  `p` is cast to bf16 only as the second dot's A operand. This is what gives the 2e-2 bf16 accuracy.
- **Fully-masked-row guard.** `m_safe = where(m_new == -inf, 0, m_new)` avoids `inf - inf = NaN` for padded
  rows; real rows always see key 0 so it is only hygiene, but NaNs in a masked store are still NaNs in the
  register file and cost nothing to avoid.
- **int64 pointer math on the block id and token index.** `blk * stride_cache_block` overflows int32 at
  ~2 GiB of cache per layer; the cast is a scalar op per iteration.

## Alternatives rejected

- **Split-KV (flash-decoding) for decode.** It is the standard fix for small batches: B=1 decode has only
  `Hkv = 4` programs on 170 SMs. The contract explicitly asks for one kernel and the target is the B=32 shape,
  where `B * Hkv = 128` programs are enough to keep HBM busy (numbers below). If B=1 latency matters later,
  the kernel can take a KV-split index in a 4th grid dimension plus a 10-line reduce; not built now.
- **Separate decode kernel with `BLOCK_M = GROUP` and no tensor cores.** Would avoid the 10 wasted rows of a
  16-row tile at q=1, but the decode step is bandwidth bound: the 16-row MMA costs nothing measurable and
  the code stays one kernel.
- **Gathering K/V into a contiguous buffer before attention.** Doubles the traffic of the one thing the kernel
  is bounded by.
- **`tl.make_block_ptr` / TMA descriptors.** sm_120 has no TMA in the Hopper/B200 sense and the block table
  indirection makes descriptors awkward; plain pointer tiles are what vLLM uses too.

## Numbers

Pending — see the sweep table and perf tests once the box is up.

## Pitfalls

Pending.
