# E024 formal v2: completed execution, failed first independent formal audit

All twelve registered OFF/ON processes returned and were reaped. The driver stop, owner and recovery commands each returned 0. Final recovery completed at 2026-10-18 04:21:18 UTC (October 17 21:21 PDT). Independent restoration passed: original oMLX PID 43565 was replaced by 73594 with retained model/configuration and fresh recovery observations. Warm KV and exact hidden runtime state were not restored.

The frozen formal auditor returned FAIL: `canonical unchanged pinned ref`. It verified source/preparation metadata but stopped before opening the HTTP wire/ledger. Therefore this package establishes neither formal throughput/latency qualification nor a new Pareto point. The separately retained execution summary has `performance_qualified=false`. The v12 rendering queue ended `NO_NEW_SPEED_CHART`, leaving v11 current.

The full failure, raw twelve-process experiment, restoration, frozen source, registration and preparation are retained in the verified archive. `archive-verification.json` and `archive-stream.json` bind the archive, extracted member hashes and original membership. Archive retention is independent of experiment success. This explanatory file is outside the sealed archive; sealed members are unchanged.

Next step: identify the precise reference contract mismatch. Any parser repair and offline rerun must use a new source and result directory, preserve this failure and check the same original evidence. There is no authorization from this failure to drop a source/hash, resource, count, timing or restoration requirement.

## Later independently versioned reparse

The registered HF tokenizer snapshot symlink caused this rejection. A narrow new parser and eight targeted regressions later passed; the complete same raw cycle now has independent864-request/resource/restoration PASS. See [separate corrected audit and results](../E024-route-http-formal-v2-reparse-v1/RESULTS.md). This original failed audit and sealed archive are unchanged.
