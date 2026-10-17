# E028: skip identity gathers during decode refill

Status: preparation, default OFF. No model or performance qualification yet.

The current DecodeBatch rebuild gathers every surviving old row before appending
new requests, including when every old row survives in its original order. The
candidate passes those old arrays to the unchanged slice/pad/concatenate path.
It retains the required combined output allocation and all model arithmetic.
Only nonempty identity refill with unchanged K/V capacity is eligible. Reorder,
retirement, initial batch construction and capacity growth use the original path.
There is no process-wide patch or default production behavior change.

The actual E023 workload contains two B4 to B6 refills at L1536. The old full-model
state selected by those expressions is 971.25 MiB, but actual Metal traffic and
saved time are unknown. See the [source review](../../reviews/2026-10-17-identity-refill-gather.md).

The first device gate uses one actual-shape attention layer plus one GDN layer,
with independent OFF/ON inputs, full original BF16/F32 snapshots and exact-byte
comparisons. It includes identity refills, fallback controls, strided/offset
inputs, delayed evaluation and independent subsequent writes. The complete
64-layer state, model logits/tokens, scheduling and HTTP remain later gates;
this diagnostic cannot substitute for them.

The bounded numerical policy reuses the exercised E026 guards: cache 0, observed
MLX active/cache/peak at most 1 GiB, observed physical current/peak at most 2 GiB,
120-second child, external owner with cleanup, stable existing pressure 2 allowed
only for this separately registered small diagnostic. It does not admit full-model
or performance work at pressure 2. The outer authorized SIGSTOP/SIGCONT lease alone
pauses/restores the original oMLX service. No cold restart is authorized here.

The fixed device contract and all source/runtime identities will be committed
before execution. First numerical mismatch stops the screen; retain the full
successful prefix and failed evidence. A passing screen is module-level evidence,
not a throughput–latency point. Paired whole-model HTTP measurements are required
before adopting this path or changing the Pareto plot.
