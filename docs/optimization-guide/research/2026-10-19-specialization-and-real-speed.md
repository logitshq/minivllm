# Specialization must remove measured work

Accessed 2026-10-19. These are external design references, not new M5 benchmark results.

## Recent result: acceptance is not completed-request speed

[Acceptance Is Not Speed](https://arxiv.org/abs/2610.23818v1), submitted 2026-10-16, evaluates relaxed verification on twelve target/drafter pairs. At the most relaxed evaluated settings, its abstract reports 1.34× accepted tokens per round but 1.54× longer completions on average. It therefore measures completion length, verification rounds, wall time and accuracy together. This is research on lossy verification; it supplies no justification for relaxing our exact correctness contract. Our decision: retain normal EOS and full-request time, report all committed tokens, and never rank a candidate by acceptance alone.

## Concrete native-runtime design, with limits

[BaseRT](https://arxiv.org/html/2607.00501v1), July 2026, describes preallocated decode scratch/KV, fused dequantization, dedicated decode/prefill kernels and reduced command-dispatch/synchronization overhead. The evaluated machines are M3 and M4 Pro, with different models and quantizations from ours. Its tables include near-parity and a loss against MLX on a large MoE, despite stronger small-model gains. Thus specialization is a mechanism to test, not a proof that every batch must already win. The useful design question is which allocations, weight reads, state copies and unoverlapped submissions we can actually eliminate.

## Decision for this repository

The user’s acceptance target remains a large, repeatable advantage at every target batch/concurrency for Qwen3.8-27B and GLM-5.3-Flash. Current E033 C1 improves about5.5–5.7% over its paired graph-only control but still trails the historical oMLX endpoint. Native’s best C8 advantage is only1.34%. Neither is completion of that target.

The immediate C1 profile measures submission/compiled-call/async-eval/delivery boundaries with OFF/runner/full overhead controls. Overlapping host spans must not be summed as GPU time. A large speculative gain also needs actual weight amortization: calling the original B1 layer T times in layer-major order preserves an arithmetic route but does not by itself prove each weight byte is read only once. The rowwise verifier must first pass prefix-state equivalence, then beat the cost of the same number of native steps before adding a draft model. Candidate source size, accepted length and a microkernel ratio cannot replace that measurement.

For GLM, E034 shows why nominal weight size and one footprint total do not establish residency. Inspect compression, allocator and load-stage state before another attempt. Any new diagnostic memory policy must be declared before execution; failed historical zero-swap evidence stays failed. No current GLM speed or fit claim is made here.
