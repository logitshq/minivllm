# A3 — Gated DeltaNet kernels with slot-indexed state

Files: `minivllm/kernels/gdn.py`, `tests/test_gdn_kernel.py`. Contract: ARCHITECTURE.md §6.

## What was built

Four entry points, three Triton kernels (two of them are one kernel with a constexpr switch) and one wrapper:

| entry point | kernel | consumer |
|---|---|---|
| `causal_conv1d_update(x, conv_state, state_slots, weight, has_initial_state)` | `_causal_conv1d_kernel[SINGLE_TOKEN=True]` | GDN layer, decode-like tokens (1 per request) |
| `causal_conv1d_varlen(x, query_start_loc_p, conv_state, state_slots_p, weight, has_initial_state_p, max_query_len=None)` | `_causal_conv1d_kernel[SINGLE_TOKEN=False]` | GDN layer, prefill-like tokens (and 1..8-token spec-verify batches) |
| `gdn_recurrent(q, k, v, g, beta, ssm_state, state_slots, query_start_loc_d, has_initial_state)` | `_gdn_recurrent_kernel` | GDN layer, decode-like tokens, 1..8 per request |
| `gdn_chunk(q, k, v, g, beta, query_start_loc_p, initial_state, query_start_loc_cpu=None)` | `fla.ops.gated_delta_rule.chunk_gated_delta_rule` | GDN layer, prefill-like tokens |

State pools are exactly the contract's: `conv_state[j]` bf16 `[slots, C, K-1]` (oldest input first, the HF
`conv_state` layout) and `ssm_state[j]` fp32 `[slots, Hv, Dk, Dv]`. Every kernel takes the slot index per request
and reads/writes the slot in place; nothing is gathered or copied for the decode path.

## The math, and which orientation the state has

Both references agree (HF `torch_recurrent_gated_delta_rule` and fla `fused_recurrent_gated_delta_rule_fwd_kernel`
with `STATE_V_FIRST=False`, the default). Per token, per v-head, with `S` of shape `[Dk, Dv]`:

```
q = q / sqrt(sum(q^2) + 1e-6);  k = k / sqrt(sum(k^2) + 1e-6)     # L2 norm, eps INSIDE the sqrt
q = q * Dk^-0.5
S = S * exp(g)                        # g <= 0, per (token, v-head)
u = beta * (v - k^T S)                # k^T S = sum over Dk  -> [Dv]
S = S + k u^T                         # outer product, S[Dk, Dv]
o = q^T S                             # [Dv]
```

So `ssm_state[slot, h]` is `S[Dk, Dv]`, `o = q^T S`, and the state read is a column reduction over Dk. That matches
HF's `(S * k[:, None]).sum(-2)` and fla's `tl.sum(b_h * b_k[:, None], 0)` / `b_h += b_k[:, None] * b_v[None, :]`.
fla also has a transposed variant (`state_v_first=True`, `S[Dv, Dk]`) — we do not use it; the chunk path is called with
the default so the state fla returns is directly the layout our recurrent kernel reads.

L2-norm epsilon: HF's `l2norm` is `x * rsqrt(sum(x^2) + eps)`, the installed fla fused-recurrent kernel is
`x / sqrt(sum(x^2) + 1e-6)`, and fla's `l2norm_fwd` kernel (used by the chunk path) is `x * (1 / sqrt(sum(x^2) + eps))`
with eps 1e-6. All three put eps inside the sqrt; the kernel does the same. (An earlier fla version had
`x / (sqrt(sum(x^2)) + eps)`; with |x|² ≈ 128 for bf16 activations the difference is ~1e-8 relative, far below any
tolerance here, but matching the installed chunk kernel exactly is what keeps chunk→recurrent continuation bit-close.)

## Design: `gdn_recurrent`

- Grid `(Dv / BV, Hv, B)`: program per (Dv tile, v-head, request). The recurrence is independent per output column
  (every column of `S` only ever interacts with the same column of `v`, `u`, `o`), so tiling Dv needs no communication.
  q and k are re-read by every tile (1 KiB per token vs 64 KiB of state per head), which is free.
- A full `[128,128]` fp32 state is 16 384 registers per program (64 per thread at 8 warps before any temporaries) —
  it spills. fla uses `BV=8, num_warps=1`. We swept BV ∈ {8..128} × warps ∈ {1..8} (table below) and picked
  `BV=__BV__, num_warps=__NW__`.
- Varlen: `T = query_start_loc[i+1] - query_start_loc[i]`, token loop with a runtime trip count; pointers advance by the
  token stride, which is a kernel argument so q/k/v can be column slices of the fused `[T, 2*key_dim+value_dim]` conv
  output (no `.contiguous()` copies; the layer passes views).
- GQA in-kernel: q/k may have `H` heads with `Hv % H == 0`; v-head `i_hv` uses k-head `i_hv // (Hv // H)`, exactly
  `repeat_interleave(Hv // H, dim=1)`. This is fla's convention for both its recurrent and chunk kernels, so the
  layer never materialises repeated q/k. With `H == Hv` it is the contract's `[Td, Hv, 128]` signature unchanged.
- `has_initial_state` False → `S = 0`; the slot is still written at the end (first chunk of a request lands in the
  slot). The flag is read as `uint8` (`bool_tensor.view(torch.uint8)`, free) because Triton's handling of `int1`
  pointers has been unreliable across versions.
- Output dtype = v dtype (bf16); state stays fp32 in the slot. Scale `Dk^-0.5` is passed in, like fla.

## Design: `causal_conv1d_update` / `causal_conv1d_varlen`

One kernel, grid `(B, C / BLOCK_C, ceil(max_query_len / BLOCK_T))`, constexpr `SINGLE_TOKEN` picks `bos = request
index, T = 1` (decode) vs `query_start_loc` (varlen). Unlike the delta rule, the conv has **no sequential dependency**:
output `t` only needs inputs `t-3..t`, so prefill is tiled over tokens too (a 1024-token prompt is 32 token tiles ×
40 channel blocks, not one program looping 1024 times). Per tap `j` (static-unrolled, K=4) the input index is
`t + j - 3`; a negative index reads `conv_state[slot, c, idx+3]` when `has_initial_state`, else 0. Accumulate in fp32,
`silu`, cast to the x dtype. The tile-0 program alone writes the new state — the last K-1 inputs of
`[old state, x_request]`, which for requests shorter than K-1 tokens means part of the old (shifted) state. A
`tl.debug_barrier()` sits between the reads of the old state and the writes: same addresses, possibly different threads.

`max_query_len` is an optional kwarg (the contract signature has none) so the layer can size the grid from
`ForwardContext.max_query_len` instead of `Tp` (which would launch `Tp/BLOCK_T` token tiles per request, nearly all
empty, for a batch of 32 × 8-token verify requests). The default is `Tp`, always correct.

## Design: `gdn_chunk`

Thin wrapper. The installed fla (`flash-linear-attention` __FLA_VERSION__) signature is
`chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, state_v_first=False,
cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None, **kwargs)`; layout `[B, T, H, D]` only (`head_first` was removed
and raises), `B == 1` when `cu_seqlens` is given, `initial_state [N, Hv, Dk, Dv]` with `N = len(cu_seqlens) - 1`,
returns `o.to(q.dtype)` and the fp32 final state `[N, Hv, Dk, Dv]`. Things worth knowing:

- GVA is supported (`Hv % H == 0`) — same mapping as our recurrent kernel; no repeat needed on either path.
- `prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu)` runs on `cu_seqlens_cpu` when given, otherwise on
  the GPU tensor, which forces a device→host sync per call (one per GDN layer per prefill step = 48 syncs on the 27B).
  `gdn_chunk` exposes `query_start_loc_cpu` for the model runner, which already holds the pinned CPU copy.
- It is an autograd `Function`; call it under `torch.inference_mode()`/`no_grad` or it keeps q/k/v/g/beta/A alive
  for a backward that never comes.
- fla's `use_gate_in_kernel`/`use_beta_sigmoid_in_kernel` could fold `g = -exp(A_log) * softplus(a + dt_bias)` and
  `beta = sigmoid(b)` into the kernel; not used — the contract hands both paths the same precomputed fp32 g/beta, and
  the recurrent kernel would have to grow the same options for parity.

## Alternatives rejected

- **Full-state program (no Dv tiling).** 16k fp32 registers per program; spills, and only `B × Hv` programs — fewer,
  fatter programs are the wrong shape for a bandwidth-bound kernel. Tiling Dv is free (see above).
- **Gather slot states into a dense `[B, Hv, Dk, Dv]` tensor and call fla's `fused_recurrent_gated_delta_rule`.**
  That is a 96 MiB gather + 96 MiB scatter per layer per step on top of the 192 MiB the kernel itself must move —
  2× the traffic of a slot-indexed kernel. (fla's kernel is also the design template for ours; the only real
  differences are the slot indirection, the `has_initial_state` flag and the tile shape.)
- **One program per request looping over all channels and tokens for the conv.** Simple but serial over tokens;
  a 1024-token prefill would be 1024 sequential iterations on ~40 programs. Tiling tokens costs nothing because
  the conv has no recurrence.
- **Sliding window in registers for the conv (keep x_{t-3..t-1} between iterations).** Saves re-reading 3 inputs per
  tap from L1/L2; needs K-1 extra named tensors and a prologue. The K re-reads are L1 hits; measured bandwidth
  (below) is already near the roofline for a kernel this small, so not worth the code.
- **Separate `update` kernel.** It is the varlen kernel with `BLOCK_T=1` and the request index as the token index; one
  constexpr branch instead of a second kernel.

## Numbers (RTX 5090, torch __TORCH__, triton __TRITON__; `cd ~/minivllm && .venv/bin/pytest tests/test_gdn_kernel.py -q -s`)

__NUMBERS__

## Pitfalls

__PITFALLS__
