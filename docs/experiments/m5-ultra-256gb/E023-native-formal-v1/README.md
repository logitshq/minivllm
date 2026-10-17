# Native D512 formal baseline v1: two audited operating points

Completed 2026-10-17 on M5 Ultra 256 GiB. Qwen3.8-27B NVFP4 revision
`5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`; fixed eight coding fixtures,
886–978 prompt tokens, greedy normal EOS, cap 512. Baseline compile/fuse/gates ON,
norm reuse and experimental E024/E028 OFF. No qualification observers in timing.

Each C has one fresh process, eight warm and 64 measured requests. These are
single-run observations, not confidence intervals. All 128 formal requests
reached cap 512; actual early EOS coverage is zero. Numerical eligibility binds
the prior full-logit/cache-state replay of representative C1/C8 HTTP histories;
this is not a full-state replay of every new formal arrival schedule.

| Concurrency | Requests | Committed tokens | Common window (s) | Output tok/s | p50 (s) | p95 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 64 | 32768 | 722.205195 | 45.372147 | 11.283287 | 11.348184 |
| C8 | 64 | 32768 | 195.747992 | 167.398907 | 24.431814 | 25.979063 |

Throughput uses the sum of actual committed output IDs divided by the common
first-POST to last-DONE window. Latency is full per-request POST→DONE, including
server queueing. Waiting for a client concurrency slot occurs before POST.
Both points express a throughput/latency tradeoff; neither dominates the other.
oMLX/Lithos D512 comparison is pending, and old cap 32 measurements are separate.

The independent final audit reparses all 144 warm+formal raw SSE request records,
fixture prompts, UUIDs, committed text, EOS/usage, clocks, cleanup and observed
HTTP concurrency. It also verifies the fixed source/runtime and model pre/post
receipts, all 216 raw resource/absence samples, both child lifetimes and recovery.
Pressure remained 1; swap delta −8 MiB, Swapouts 0, Pageouts 0, Swapins +140. These
process observations cannot exclude an unknown runtime or GUI GPU activity.
Runtime/source bytes are independently hashed; large model payloads are checked
through full producer pre/post hashes plus unchanged file stats, not hashed a
third time by the final auditor. All original point files retain performance=false;
only the separate completed-cycle audit awards performance eligibility.

Original oMLX PID 90204 was stopped, and experiment children 92555/97229 were
terminated, waited/reaped and absent before restoration. Recovery completed
16:12:21 UTC as PID 98702 with the original Qwen3.8-Flash-Next-oQ6e-mtp model and
fresh MTP/resident PLE log evidence. Warm KV and exact hidden runtime state are
not restored. The independent recovery audit checks 19 physical samples and the
retained 11,301-byte log window; final HTTP/job/listener observations remain
source-pinned controller receipts. The maximum recorded system footprint during
recovery was 175,831,891,968 bytes.

Root: 10 formal-audit CPU regressions passed; root: 15 restoration-audit tests passed
before execution. Independent source review found no blocker. This formal cycle
and its independent audits completed without a runtime or audit retry.

Archive: **935 members**, 3,887,252 compressed bytes.
SHA256 `5638e6d37e0e69fb278190b7f182febf05957708067e325c7ffe764132bf26eb`.


Every decompressed member's size/hash and every original file were verified;
source tree membership was rechecked. `evidence.tar.xz` contains raw owner,
cold-journal, execution, CPU prepare, analysis, preregistration and audit sources.
The earlier representative numerical histories remain in their own immutable
[HTTP qualification archive](../E023-http-baseline-v2/README.md).

- [Point and cycle audit](independent-audit.json)
- [Restoration audit](restoration-audit.json) and [restored journal](restored-journal.json)
- [Archive manifest](manifest.json) and [verification](archive-verification.json)
- [Frozen preregistration](../../../../bench/m5_ultra_256gb/pareto_realistic/prereg-native-formal-v1/README.md)
- [New D512 chart and reproducible plotting data](../../../optimization-guide/progress/2026-10-17-v6/README.md)

The next candidate remains the default-OFF E024 long decode graph. Its minimal
cold owner and recovery adapter have CPU coverage; no E024 device qualification
or speed gain is claimed by this baseline.
