# 038 — From correct verifier cost to a serving decision

E057 feeds shared gate/up results into every MLP layer, unlike the E056 shadow canary. It preserves rowwise norm, mixer, SiLU/multiply, native down and residual order. Six full selected-prefix/continuation cases must pass before timings. Independent comparison of all32 raw files/2064 tensor pairs passed; all1349 parameter hashes match before qualification, before timing and afterward.

Measured device-baseline pooled p50: T4 native85.7917ms → verifier+commit58.9421ms; T8 native168.9082ms →107.8756ms. Median paired group ratios are1.45442× and1.56605×. E054's head-only costs were measured earlier, so their small difference from E057 is not a paired estimate of MLP benefit. This result does not justify stacking speedup factors.

A useful first-principles serving gate is `round_cost / useful_outputs < native_cost_per_output`, with round_cost including drafting, verification, acceptance/rejection, state repair and scheduling. Use actual useful output counts (including the algorithm's correction/bonus convention), not the configured draft width. Measure streaming gaps separately: a faster block average can still delay individual output tokens. C1 and C8 may select different widths or disable speculation.

Keep the design modular: an unchanged native path, an optional shared verifier, an explicit accepted-prefix state interface, and a small measured policy table indexed by concurrency and context/width. Start with fixed policies and paired runs; adaptive selection requires fresh evidence. The speculative state must retain exactly the accepted causal prefix and the drafter must advance consistently.

The newest [primary research notes](../../../bench/reviews/2026-10-20-load-adaptive-speculation.md) reinforce two constraints: account for temporary-memory lifetimes, and treat throughput loss from reduced active batch as part of speculation's cost. They do not establish gains for this M5.

[Audited result and plot](../../experiments/m5-ultra-256gb/E057-c1-mlp-shared-full-cost-v1/README.md). The complete experiment remains local-cost-only; there is no new oMLX/Lithos Pareto claim.
