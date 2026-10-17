# E020 external speculative HTTP independent audit

Audit: **PASS_RECONCILED**. Experiment: **COMPLETE_VALID_CONTRACT**.

Started/finished/valid servers: 15 / 15 / 15 of 15.
Warm/formal requests: 45 / 45; valid requests 90.

This audit reconstructs text, counts, clocks, E2E, TTFT, C1 aggregate throughput and process medians from raw SSE; stock logs are separately checked for declared/executed routes. All negative, partial and invalid rows remain in summary.json. A failed experiment can still be faithfully reconciled; audit PASS is not a speed/adoption claim.

| Engine | Pair | Valid contract | Throughput ratio (on/off) | E2E speedup (off/on) | TTFT speedup (off/on) |
|---|---:|---|---:|---:|---:|
| omlx | 1 | True | 0.903280 | 0.903280 | 1.132072 |
| omlx | 2 | True | 0.911512 | 0.911512 | 1.138305 |
| omlx | 3 | True | 0.901819 | 0.901819 | 1.135623 |
| lithos | 1 | True | 1.077841 | 1.077841 | 1.020003 |
| lithos | 2 | True | 1.099826 | 1.099826 | 1.115700 |
| lithos | 3 | True | 1.080440 | 1.080440 | 1.058104 |

Cross-framework latency/throughput may be compared with explicit output contracts: native/oMLX and Lithos do not share an identical qualified continuation. External visible text is not a new live engine raw-ID capture. MTP log denominators are nominal proposal capacity; Lithos verify width/steps do not constitute a fresh acceptance trace.

The archive retains frozen source/config, model tokenizer metadata, all raw run/lease/jobs receipts, original official logs and recorded package/payload hashes. It excludes weight/pack payloads and runnable virtual environments. Full payload checks were performed by the runner before/after; this CPU audit independently hashes frozen source/metadata and every archive member without rereading large weights.

Verified archive: 9131 regular members, 29444554 compressed bytes. SHA256 `ba9c668ce0d98760f5413307af86400e4129d43974ab7d3d64cdfcfe0bdfddec`.

Use manifest.json and tar.verify.json to verify the evidence. For offline CPU replay, extract the archive and pass the extracted inputs/, run/, lease/, jobs.json plus --tokenizer model-metadata/target/tokenizer.json to the archived audit script; choose a new --out. No server or GPU is required.

Nonzero Swapins/other paging counters and sampled RSS limits are retained per process. Service restoration is established only from the outer lease identity/stop-to-running receipt. No zero-paging, GPU allocation-peak or global Pareto claim is made.

## Audit corruption challenges

Six in-memory corruptions were independently rejected after four unmodified positive baselines passed: changed SSE text/visible IDs, reversed SSE timestamps, impossible MTP acceptance, foreign listener PID, wrong parent PID, and wrong service restoration identity. See [negative-challenges.json](negative-challenges.json), [challenge source](external_spec_http_audit_challenges.py), and [auditor source](external_spec_http_audit.py). These review sidecars were added after the 9,131-member archive was verified; the archive and its manifest were not rewritten. The challenge receipt pins both scripts and the original evidence files.
