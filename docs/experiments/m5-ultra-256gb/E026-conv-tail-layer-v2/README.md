# E026 real-weight layer v2: exact output and state, smaller retained backing

Preregistered in `aba84a5`, after the separately archived [v1 observation failure](../E026-conv-tail-layer-v1/README.md). Plan SHA `313384a618fe1ddf4724ff2fafb8c5e07fbd6f923bb04fe5f5654fd4caf2eef2`. The root and independent auditor produced identical PASS receipts. This qualifies a single real-weight GDN layer with synthetic finite inputs; full-model and HTTP performance remain unqualified by this test.

Three fresh processes ran OFF, ON-lazy and ON-materialize across T=1,19,935,978 with fresh and synthetic continuing state. All **24 cases / 204 phases** completed. All 252 saved original-dtype arrays (983,424,768 bytes) passed independent SHA, shape and finite checks. Both ON arms matched OFF across 84 full arrays each: **168 comparisons / 655,616,512 compared bytes**, including complete output, convolution output, recurrent state, convolution tail, inputs and loaded parameters. Each case also preserved the actual original convolution tail exactly. No tolerance, quantization change or arithmetic rewrite was introduced.

After dropping observer/input aliases, the following allocated backing remains in the state group. Fresh and continuing cases give the same values; both ON variants give the same values.

| Input tokens | OFF state bytes | Either ON state bytes | State / active / unique bytes saved |
|---|---:|---:|---:|
| 935 | 22,364,160 | 3,211,264 | 19,152,896 |
| 978 | 23,248,896 | 3,211,264 | 20,037,632 |

At T978, active bytes fall from 98,531,976 to 78,494,344. The weight and output groups are unchanged; the state group includes recurrent S plus the convolution tail. These are deduplicated allocated-buffer observations after common validation evaluation, not process footprint or throughput. They establish that the original frozen computation can retain the exact small tail while releasing its larger parent in this diagnostic. They do not establish the cause of E023's resource abort or a measured 48-layer saving.

The final layer weak reference is dead in all arms. Active arrays retain 8 bytes, including a known 4-byte normalization epsilon global; this is not a claim that all allocator memory returned to zero.

All 608 source/runtime/bundle pins rechecked. Fourteen system snapshots and six external physical probes passed, with no terminal probe races. Pressure remained 2 under the separately registered numerical-only policy; swap used and Swapouts did not grow, and minimum real free pages represented 67,712,024,576 bytes. Across all 204 child phases, maximum MLX peak was 970,521,466 bytes (below 1 GiB), and physical peak 1,001,554,856 bytes (below 2 GiB). Sparse external sampling has a lower maximum and must not replace the complete phase evidence. The original budgets were not increased.

All three children exited zero, were waited/reaped and proved absent. The owner exited zero and was waited by the service lease. Original PID74384 returned T→S with its identity unchanged; the additional read-only proof finds all four experiment processes absent. Operation durations are deadline evidence only and are not performance results.

[Independent audit](independent-audit.json) · [Identical root audit](root-audit.json) · [Memory-only comparison](independent-memory-comparison.json) · [Final process proof](final-readonly.json) · [Frozen plan](../../../../bench/m5_ultra_256gb/conv_tail_owned_copy/prereg-layer-v2/README.md).

`raw-evidence.tar.xz` preserves the complete owner output, frozen sources, arrays, raw observations, plans and service lease. Identical size/SHA payloads use standard tar hardlinks to avoid copying the same real weights and arrays repeatedly; no original pathname or contents is discarded. `manifest.json` records every original member and the verified archive SHA. Runtime binaries and the original full model shard remain external pinned inputs; the real per-arm loaded parameter bytes are included.

The throughput–complete-request-latency [v5 plot](../../../optimization-guide/progress/2026-10-17-v5/README.md) remains unchanged. Root redrew it from all 217 source files in this continuation and reproduced chart-data SHA `ad365ee1d3e9858ba3f8890db2b3a6f0ce3304c12b8f1247a8a58e18ad9fe5cc`. Only full-model-correct, separately measured HTTP results may add a Pareto point. Next work is engine-level qualification and evaluating copied tails at an existing shared boundary; this single-layer observer is not a performance implementation.
