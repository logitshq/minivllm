My first equation was wrong: it added unitless `(1 if verify)` to measured seconds, and called a speedup "throughput" without a baseline.

Correct formulas (all C terms are measured seconds):

`C_round = C_draft + C_verify + C_select_commit + C_scheduler`

Do not assume an extra correction/bonus native forward. A round may emit it as the next pending anchor, already counted as output and computed next round.

For a greedy round accepting `a_i` drafts, `n_i = a_i + 1` outputs before EOS/output-cap truncation; use actually emitted `n_i` after truncation.

`speculative_seconds_per_output = sum(C_i) / sum(n_i)`

Dimensionless speedup, same workload:

`speedup = baseline_seconds_per_output / speculative_seconds_per_output`

Local E058 verifier-only ratios omit the other C terms and prove no serving gain. Next experiment: measure the complete real-drafter loop.
