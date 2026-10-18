# E029 GDN prework: actual HTTP qualification

2026-10-17 PDT. All four preregistered phases returned PASS. Separate independent restoration, C1 numerical, C8 numerical and complete-cycle audits passed. This is correctness/observed-resource evidence, not a timed Pareto point.

| Actual captured history | Requests | Scheduler steps | Committed tokens | Full-array bit checks |
|---|---:|---:|---:|---:|
| C1 | 8 | 4,096 | 4,096 | 5,251,008 |
| C8 | 16 | 1,028 | 8,192 | 1,322,478 |

Total: 5,124 steps, 12,288 committed tokens and 6,573,486 full-array bit-check receipts. The independent audit checks receipts, source bindings, complete recorded HTTP histories, scheduling/state/ahead/token comparisons and their retained evidence. Successful tensor bytes were not retained for an independent offline arithmetic recomputation. Both histories ended at cap512; zero terminal EOS were observed, so this run does not add EOS coverage.

C1 recorded4,088 B1 forward calls and196,224 per-layer candidate dispatches; capture and replay agree. C8 recorded1,027 multirow calls and49,296 per-layer original fallbacks, with zero candidate dispatches. C8 therefore establishes fallback/scheduling compatibility in this history, not batched-fusion correctness or a mechanism for higher C8 throughput. The GDN-only switch keeps E024 graphs and E028 refill OFF.

The cycle retained249 resource samples, pressure1, nominal thermal/AC gates and zero swap-size increase. Original oMLX PID73594 was replaced by PID90511; persisted-visible recovery independently passed. Warm KV and exact hidden runtime state were not restored. The four direct experiment children were reaped before recovery.

The complete archive has1,286 regular members,1,840,591,202 raw bytes and11,914,384 compressed bytes. Every member was reopened and checked against its source SHA/size; originals and tree membership were checked again. Archive SHA256: `43c862c46b41a4a3e736600f28dcdd0160ed67570ef6c980a7413e074ae8ef53`. The archival source review fixed regular-member preservation for repeated inodes and restricts diagnostic failure arrays to their exact first-failure path/SHA; no frozen experimental source or raw result changed.

Next: twelve fresh observer-OFF servers, three adjacent OFF/ON pairs per C1/C8,8 warm and64 formal requests per process. The new GDN-only results remain separate from E024-only Native ON. A combined configuration requires its own qualification. All formal outcomes, including regressions, will be retained before any new frontier is drawn.
