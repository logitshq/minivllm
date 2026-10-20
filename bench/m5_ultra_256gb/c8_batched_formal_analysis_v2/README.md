# E055 independent reducer v2

The original v1 reducer failed on the standard Hugging Face tokenizer symlink even though its bytes matched the plan. v1 evidence, trial sources, registered specs, numerical prerequisite and performance gate remain unchanged.

This wrapper permits only the exact original tokenizer path, its fully resolved canonical blob, byte count and SHA256. Other paths retain the original strict resolver. Twelve CPU regressions pass, including same-size byte corruption, re-targeting, symlinked blob, extra reference fields and preservation of the frozen reduction code. Only the direct tokenizer-loading helper is wrapped; the original module is not patched.

The actual closed-evidence audit passes for 432 requests across six C8 processes. Median ON/OFF throughput ratio 1.0211242632, p50 ratio 0.9796914959, p95 ratio 0.9800594025; OFF rate spread 0.0013176326. The preregistered 3% speed gate FAILS. Greedy trajectories differ across schedules; the report neither labels this alone an arithmetic error nor claims all formal full states were replayed. No Pareto point or performance qualification is granted.

Evidence: `.artifacts/m5-ultra-256gb/c8-batched-formal-analysis-v2-cpu-v1/` and `c8-batched-formal-v1-independent-v2-execution-attempt1/`. The original failed v1 audit is retained beside v2 in the E055 documentation directory.
