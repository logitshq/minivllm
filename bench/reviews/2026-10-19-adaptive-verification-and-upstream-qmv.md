# Adaptive verification costs and existing upstream small-M reuse

Read on 2026-10-19. Source review only: no backend imports, GPU work, model
payload reads or new benchmark. External claims below are not miniVLLM results.
Each primary-source summary is under150 words; no source code is reproduced.

## MLX already implements small-M weight reuse

[MLX PR3764](https://github.com/ml-explore/mlx/pull/3764), merged June26, introduces
`qmv_wide` for the small-M interval between matrix-vector and matrix-matrix
kernels, including NVFP4. It dequantizes a weight group once and reuses it across
input rows. The merged follow-up moves scale multiplication after the group dot
product. **Our inference:** changing operation grouping can change floating-point
rounding; algebraic equivalence does not establish our original-M1 bit contract.
Its kernel benchmarks are specific shapes/devices, not our engine frontier.
Generic reuse is therefore existing upstream work, not our novelty. Our proposed
specialization must demonstrate a useful arithmetic/scheduling tradeoff against
the installed route, with original-byte qualification before timing.

Local confirmation: installed
[fp_quantized.h](/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/fp_quantized.h:570)
contains `fp_qmv_wide_impl`; scale application is at line651. Header SHA256:
`359a1ea9008f74cc15c61e9d39c6300dcffb9cfb9cc0ebd59ec7c00b789d719f`.
Header presence alone does not establish dispatch for our exact shape/config.

## A recent affine kernel proposal has narrower evidence

[MLX-LM PR1922](https://github.com/ml-explore/mlx-lm/pull/1922), opened September24,
remains **OPEN** when checked. Its proposal reuses decoded weights across4–8 rows
and wraps linear/embedding projections, restricted to affine4-bit models. The
author reports +22.8% aggregate throughput at batch8 with128 prompt/32 generation
tokens, and batched projection relative error up to2.3e-3 against MLX. These are
author-reported short-workload results, not batched bit equality. Our Qwen
checkpoint is NVFP4, so neither its encoding assumptions nor its speedup transfers
directly. In particular, its broad statement about per-row MLX traffic cannot
replace checking the installed NVFP4 dispatch, given the already-merged PR3764.

## EVICT makes verification cost an explicit decision input

[EVICT v2](https://arxiv.org/abs/2605.00342v2), revised September5;
[method sections3.1–3.3](https://arxiv.org/html/2605.00342v2#S3), estimates candidate
benefit from products of draft probabilities. It selects an ancestor-closed tree
prefix using expected accepted length divided by a pre-profiled step cost, then
uses the original target verification/sampling logic. The table is an average
for a fixed runtime configuration, not a prediction of the exact activated expert
set each step. It primarily targets expert-unsaturated, memory-bound MoE: adding
branches can expand the expert union and make verification expensive. The authors
call it lossless; that does not independently qualify our implementation. For
GLM, first establish a working loader/target baseline, then measure cost and
expert-union growth before choosing tree breadth or predicting a speedup.

## DSpark budgets across requests and current load

The [vLLM adaptive-verification post](https://raw.githubusercontent.com/vllm-project/vllm-project.github.io/main/_posts/2026-08-14-dspark-adaptive-verification.md)
(August14) describes a learned confidence head whose cumulative survival scores
rank draft positions across requests. It chooses a budget using expected useful
tokens divided by separately profiled draft and verification costs. The budget
therefore responds to request count and already-scheduled tokens. Graph padding
creates cost steps, so “one more token” need not have a uniform cost. The reported
implementation depends on specific varlen CUDA graphs/attention support; it is
not a ready-made Metal path. The useful transferable design is a small measured
cost table plus a qualified selection policy, not an assumption that verification
is free at low concurrency or that one draft length wins at every load.

## Concrete next experiment and acceptance rule

Keep a table keyed by model/revision, arithmetic/config switches, concurrency,
context-length bin and actual verification width/compiled shape. Store raw paired
native-AR, verifier+selected-state-commit, draft and scheduling durations, plus
accepted-token counts. Preserve negative outcomes and graph-boundary jumps;
retain measured values even if a later policy uses a smoothed estimate.

Our audited [E054 result](/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/c1-verifier-cost-v1-independent-v2.json)
only covers C1, used1023, k=T4/T8. Native-device AR medians are85.471/168.490ms;
verifier+commit medians are59.685/108.666ms. The corresponding audited median
paired savings25.724/59.655ms are optimistic full-acceptance budgets for draft and
other overhead, not engine speedups or free verification.

The smallest falsifiable next step is E056's actual layer0/32/63 T4/T8 gate/up
shadow comparison, followed only on PASS by E057's shared-output full
logits/selected-state/continuation qualification and the same local cost screen.
Reject a numerical failure regardless of kernel speed. After qualification,
compare the resulting cost table against the current verifier; then measure real
draft cost/acceptance before enabling adaptive T. C1 and C8 may choose different
qualified configurations. A kernel/local-cost win is insufficient: final engine
claims require identical-workload throughput/interactivity measurements versus
oMLX and Lithos at each tested concurrency.
