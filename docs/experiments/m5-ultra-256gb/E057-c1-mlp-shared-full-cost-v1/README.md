# E057: full shared MLP + head verifier, correctness and local cost

The actual frozen-plan run completed in 87.718875875 seconds total lifecycle and restored original oMLX PID52818. Independent audit PASS:32 raw tensor files,2064 tensor pairs,six selected-prefix/continuation cases,768 actual shared gate/up Python call receipts,three complete parameter-hash sets,400 timed calls in20 groups. Resource review separately PASS:27 samples,max gap3.320865s,max system197832212480B,zero swap growth,terminal ACK/reap/restoration verified. Sampled observations are not a hard allocation cap.

| Width | Native device AR(T) pooled p50 | Full MLP+head verifier + commit pooled p50 | Audited paired speed ratio |
|---|---:|---:|---:|
| T4 |85.7917 ms|58.9421 ms|1.45442×|
| T8 |168.9082 ms|107.8756 ms|1.56605×|

[Figure](chart/render-v2/local-verifier-cost.png) shows all five group medians, with paired lines and a separate median of group medians. Its labels can therefore differ slightly from this pooled p50 table. The figure rechecks all400 timing rows and plots the200 device-baseline rows. The remaining200 host-synchronous rows stay in the audit.

All T tokens are assumed useful for these local costs. Draft generation, rejection probability, rollback scheduling and HTTP costs are excluded. E054 head-only measurements are historical, not a paired third arm: its nearby costs do not establish the incremental benefit of MLP sharing. This is a prerequisite for a speculative serving experiment, not a new Pareto point or a win over oMLX/Lithos.

DefaultOFF and modular: shared MLP, shared head and C8 batchedGDN configurations remain independent. The next serving experiment must measure actual accepted tokens per round, complete draft+verify+commit/rejection cost, and streaming latency. Preserve native behavior for unsupported shapes and when the measured break-even condition fails.

The prepared figure v1 had an incorrect expected schema spelling; review found this before any render. v2 uses the actual frozen auditor schema and additionally requires the shared-MLP qualification fields. Both sources remain available. Both E055 and E057 rendered figures were visually inspected: labels, lines and qualification notes are readable without clipping.

`independent.json` and `resource-review.json` are compact checked summaries. `local-evidence-manifest.json` hashes the retained local evidence, including all32 tensor files; the large payloads remain under `.artifacts` on this M5. No failed device/audit attempt occurred for E057.
