# E055: C8 batched GDN formal screen — below adoption gate

Six fresh processes, alternating ON/OFF, OFF/ON, ON/OFF. Each ran 8 warm plus 64 formal requests at C8, for 432 actual requests and 384 formal requests. Graph and workload were held fixed; only batched GDN changed. All six processes closed and original oMLX PID 52818 was restored.

| Pair | OFF tokens/s | ON tokens/s |
|---|---:|---:|
| 1 | 170.691158478 | 173.895257145 |
| 2 | 170.466303393 | 174.067278449 |
| 3 | 170.650824396 | 174.264897536 |

Independent median paired throughput +2.1124%, p50 latency -2.0309%, p95 latency -1.9941%. All pairs improve, but the declared throughput >=3% and p50 <=0.97 gates are not met. Keep as a diagnostic gain, not an adopted optimization or new frontier point. Three repeats at C8 are not a concurrency sweep.

The first independent audit failed because its generic resolver prohibited the existing tokenizer symlink. v2 narrowly accepts only the exact original link, canonical blob, size and digest; 12 CPU regression tests pass. All original timing/ID math and gates are retained. `independent-v1-failed.json` and `independent-v2.json` preserve both outcomes.

The same-schedule prerequisite covers the E053 captured history, not every formal schedule. Cross-schedule output IDs differ. No arithmetic error is inferred from that fact alone; no universal numerical qualification is claimed either. Resource review separately passes 285 physical/metadata samples, zero swap growth and saved restoration. Full raw evidence stays under `.artifacts/m5-ultra-256gb/c8-batched-formal-v1-*`; this directory is a compact summary, not the full archive. Chart source remains prepared, not rendered from v2 yet.
