# Native dispatch and shared-bandwidth boundaries

Read 2026-10-17 during the GDN full-model numerical run. These July papers supplement the October sources already recorded; neither is a new M5 measurement.

## BaseRT: move work out of the token loop

[BaseRT, July 1 2026](https://arxiv.org/html/2607.00501) describes preallocated scratch/token/KV buffers, specialised Metal kernels and cheaper command dispatch. Its evaluation uses M3/M4 Pro, other models and quantisation formats, and decode128 rather than full HTTP D512. The paper's table also shows MLX ahead for one MoE configuration; its headline is not universal superiority.

Our inference: first measure the already-qualified route graph; then use allocation/dispatch diagnostics to choose the next replacement. Keep buffer ownership and state updates explicit. A native rewrite is justified by a measured remaining bottleneck, not by removing Python alone. This reading examined the paper and repository landing page, not its kernel implementation. No external speed number enters our plot.

## FusionML: idle compute does not add memory bandwidth

[FusionML, July 24 2026](https://arxiv.org/html/2607.22785) finds that a CPU branch consuming an unevaluated GPU result can serialize a lazy graph. Its prefill split uses an explicit materialization boundary and contention-aware calibration. The tested devices exclude M5 Ultra, and the full-model section reports matching tokens in five of six configurations, with one near-tie divergence; the abstract's token-identical wording is too broad for our exact-state requirement. Its validation tolerance also differs from bit equality.

For us, CPU/GPU row splitting is an unqualified prefill hypothesis, not an automatic decode improvement. Shared bandwidth and probe-induced working-set eviction make continuous online tuning risky under memory pressure. Send this mechanism to the M4 prefill track; keep the current M5 decode experiment isolated. Any trial needs matching precision, real dependent inputs, complete state checks and whole-request latency, not just a ready-input matmul microbenchmark.

## Current experiment connection

The GDN adapter changes only prework around the original five Linear calls. Its local real-state canary passed; the original 37-request whole-model comparison is now running. E024 has independently passing real HTTP histories, with formal OFF/ON timing still pending. These are separate candidates and remain default OFF.
