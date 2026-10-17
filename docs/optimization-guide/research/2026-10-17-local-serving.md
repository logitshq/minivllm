# M5 inference research follow-up — 2026-10-17

Read-only research; no implementation, tests, model imports, downloads of large files, or service/GPU operations. Two relevant September originals were found; no need to relabel older work as October research. AgSpec, KVBuffer and DAMP were excluded as already read.

## 1. JustFit: Just-in-Time State Management for Local LLM Serving

**Date verified:** first submitted September 15; the inspected manuscript is **v3, September 30, 2026**. [Submission history](https://arxiv.org/abs/2609.17475v3), [original full text](https://arxiv.org/html/2609.17475v3). The paper links an [author reproduction tag](https://github.com/YuhuaBillChen/mlx-vlm/tree/justfit-repro-v4/examples/justfit); its GitHub contents could not be retrieved here, so no source-code reproduction claim follows.

**Actual evidence:** M4 Pro, 24 GiB, Qwen3.8-27B **MXFP4**, with lossy TQ4 KV; this is not our NVFP4/BF16-state configuration. Sections 3.1–3.3 separate bounded attention workspace, component lifetimes and request-state ownership. Prefill reconstructs two KV heads at a time, explicitly evaluates each group, and concatenates outputs. The runtime retains target KV/recurrent state when switching singleton speculation and batched AR. Its mixed cadence is four decode forwards followed by at most 64 prefill tokens. Section 4 distinguishes prefill work time, post-first-token decode time, concurrent common-active decode time, and end-to-end wall time. Long-context cohorts include prefix extensions. These are not our first-POST→last-DONE or request POST→DONE metrics. [Sections 3–4](https://arxiv.org/html/2609.17475v3#S3)

**Numerical boundary:** adopting TQ4 changes values; the paper does not prove equivalence to our full state. Splitting heads preserves real-number attention semantics, but a different MLX SDPA shape can select different kernels/reductions. Exact bits must be tested, not inferred.

**Our falsifiable candidate, low priority:** an isolated default-OFF two-head prefill workspace path, keeping current weights, BF16 KV, masks and context unchanged. Qualify every output/full state against the original path before timing; any bit difference rejects it under the current contract. Record peak and total prefill time including added evals. Arithmetic sanity check: at L1536, four KV heads of dimension256 in BF16 require only `2×4×1536×256×2 = 6 MiB` per row/layer for K/V; halving this operand is a 3 MiB saving before scratch. Therefore this is primarily a long-context/resource hypothesis, not the first D512 speed experiment. Do not import its quantizer, component swapping, or capacity ratios.

## 2. SiliconBench: Speed, Memory, and Fidelity for LLM Serving on Unified-Memory Desktops

**Date verified:** v1 submitted **September 12, 2026**; publication date is distinct from its earlier measurement dates. [Submission history](https://arxiv.org/abs/2609.19169v1), [original full text](https://arxiv.org/html/2609.19169v1), [authors' repository](https://github.com/WindChimeRan/SiliconBench).

**Actual evidence:** the main track uses M5 Pro 64 GB/macOS26.6, Qwen3-0.6B BF16, C1/8/16, 100 requests and three warmups. Larger Qwen3.8-27B/Qwen3.6-35B-A3B comparisons use nominal four-bit weights, C1/2/4, different GGUF/MLX formats and one observation per configuration. The paper associates packed mixed prefill/decode with better loaded TTFT, but does not isolate that feature causally. Throughput counts completed output tokens over sweep wall time; failed requests remain in the denominator. Token accounting can fall back to stream-delta counts. Fidelity is a separate classification task, not full-logit/cache equivalence. Historical engine-version/model metadata is incomplete. [Sections 4, 6.1 and Appendix A](https://arxiv.org/html/2609.19169v1#S4)

**Our falsifiable candidate, higher priority:** tune only the amount of prefill permitted during active decode, with a default-OFF `mixed_prefill_token_cap=64` versus the current policy. Keep FCFS, normal EOS, weights, request bodies, output cap, decode-first ordering and existing kernels. Do not simultaneously add packed mixed kernels, prompt sorting, N4 cadence or prefix caching. This is our proposed ablation, not a paper-reported result.

Local source rationale: `tutorial_code/miniengine/engine/scheduler.py:Scheduler.schedule` gives the remaining token budget to prefills. `engine/model_runner.py:ModelRunner.run` constructs decode then prefill and reaches a shared eval; a mixed step also disables its next-step ahead submission. Smaller chunks can shorten each interruption but add forwards and delay prompt completion. Either outcome is plausible; no speedup estimate is justified.

Qualification must use both engines under the same newly recorded policy/schedule, retaining full logits, all KV/recurrent state, ahead-consumption, refill and retirement checks. Do not force a C1 token oracle into a different batch schedule. Then compare fresh C1/C8 runs of unchanged D512/N64, with observer OFF, complete raw counting, normal EOS and resources as hard gates. The primary result remains committed throughput versus **POST→DONE p50**, including queueing; p95 and TTFT are secondary. Reject a TTFT-only improvement that worsens the relevant complete-latency/throughput frontier. C1 should exercise the unchanged no-active-decoder path.

## Decision

Finish the already prepared route-graph serving qualification and measured baseline first. Of these new directions, the single mixed-prefill budget knob is the smallest useful next performance question. The workspace split is conditional on measured temporary-memory pressure; TQ4 and heterogeneous compute are outside the current exact-numerics experiment.

The July-only BaseRT records were checked and excluded from this September–October selection: [2607.00501](https://arxiv.org/abs/2607.00501), [2607.19438](https://arxiv.org/abs/2607.19438). No paper's speed ratio is projected onto the 256 GB M5 Ultra, and no published number is added to our Pareto plot.

M4 coordination: mixed-prefill admission belongs to the already agreed M4 branch. Share this single-knob hypothesis there; the M5 branch continues graph/refill validation first and will independently test any accepted M4 implementation after pulling its commit. No duplicate scheduler implementation is started here.
