# E024 route-preserving graph: 52 diagnostic steps pass

2026-10-17. **The registered numerical matrix passes; no HTTP, full-workload or
performance qualification is granted.** The unchanged default-OFF candidate
uses bound1023 for used257..1023 and bound1536 for used1024..1536. This run used
Apple M5 Ultra/g17d, MLX0.32.3, the pinned Qwen3.8-27B NVFP4/BF16 target, physical
K/V capacity1536, and core compile/fuse/gates compile/norm reuse OFF.

| Registered coverage | Steps | Result |
| --- | ---: | --- |
| B1..8 × used257/1023/1024/1280/1536 | 40 | PASS |
| B1 chain1023→1024→1025 | 3 | PASS |
| Mixed B8 chain, longest row1023→1024→1025 | 3 | PASS |
| B1→B8→B1 reuse at used1023 and1024 | 6 | PASS |

The mixed chain starts with row lengths257/511/767/900/978/1000/1022/1023.
Each arm advances with its own greedy output. Longer histories repeat and
truncate the frozen D prompt IDs: these are diagnostic extensions, not long
generated conversations. The run loads all64 model layers but does not exercise
the complete D512 scheduler/HTTP workload.

Every step passed finite, original-dtype bit equality for full logits plus128
K/V/conv/S state arrays: **6,708 output comparisons and52 actual compiled
calls**. The48 independent prefill comparisons contribute another6,192 array
checks. Per-arm131-array inputs and all1,349 parameter values retain unchanged
before/after full-byte hash receipts. Producer source/model checks passed before
and after execution. Two callable families were observed; this is not a claim
that only two machine graphs were compiled.

## Independent evidence and its limits

The [independent numerical audit](independent-numerical.json) passes. It reads
ten complete raw snapshots and independently compares three full output pairs,
each containing129 tensors, with zero differing words or bytes:

| Point | Saved original-dtype payloads |
| --- | --- |
| matrix-B1-used257 | Canonical131-array input |
| matrix-B1-used1023 | Canonical input and both129-array outputs |
| matrix-B1-used1024 | Canonical input and both129-array outputs |
| mixed-used1024, B8 | Canonical input and both129-array outputs |

The other49 output points retain source-pinned device bit comparisons and
full-byte hash receipts. Their successful output payloads were not all saved,
so a remote reader cannot independently repeat every byte comparison from this
archive. Parameter and successful prefill checks likewise have producer
receipts, not complete offline parameter/prefill payloads. The numerical
auditor did not rehash model weights or independently audit owner resources.

The producer took222.566152 seconds, approximately223 seconds, including loading,
prefill, compilation, comparisons, hashing and raw serialization. This is
**not forward latency or throughput**. Its observed MLX peak was56,951,132,480
bytes (53.040GiB), with final active bytes8. Two verification models, diagnostic
states and graph caches contribute to that peak; it is not production memory.

The owner's46 recorded resource samples all show pressure1, AC power and
thermal code0. Recorded swap stayed421.06MB; Swapins, Swapouts, Pageouts and
Compressions had zero counter growth, while Decompressions increased283.
These are sampled observations, not a claim that all paging activity vanished.
Owner17177 directly reaped child17212 with exit0 and strict PID absence.

The original oMLX servicePID13908 was restored asPID18261. The independent
[restoration audit](restoration-audit.json) passes, with fresh resident PLE and
Lightning MTP evidence for the original `Qwen3.8-Flash-Next-oQ6e-mtp` service.
The [RESTORED journal](restored-journal.json) preserves the final identity and
persisted-visible settings. **Warm KV and exact hidden runtime state were not
restored.** This restoration scope is separate from numerical qualification.

## Archive and next gate

[evidence.tar.xz](evidence.tar.xz) contains760 small files and is1,184,892 bytes.
Its SHA256 is `0baa9992b1e00a31c17ee6dcd53e6ac41dc0d206116d4eddbad64b1bb2444cc0`.
The [manifest](manifest.json), SHA256
`74d17166633c583c6ca18434b02944de524c77dc9506e092939010a9e348f888`, records exact
source paths, sizes and hashes. The [archive verification](archive-verification.json)
confirms every archived member and every original file was rehashed unchanged.

**Ten full safetensors files remain local, outside the Git archive:**
7,902,883,133 bytes including headers, with7,902,761,092 payload bytes. Their
exact original locations are in the manifest. Remote-only raw re-auditing
requires those originals or a rerun; the small archive alone is insufficient.

The next go/no-go is forward-only timing at B1/B8×used1023/1024, first with gates
OFF and then exact gates-ON checks, followed by observer-OFF timed repetitions.
That diagnostic may decide whether further integration is worthwhile; it cannot
add an HTTP/Pareto point. Parameter replacement, ahead/retirement, current
gates-ON full-workload checks and actual HTTP scheduling remain separate gates.

[Preregistered plan](../../../../bench/m5_ultra_256gb/route_decode_graph/prereg-matrix-v1/README.md) ·
[Tutorial](../../../optimization-guide/tutorials/024-decode-graph-dispatch.md) ·
[Current measured D512 baseline](../../../optimization-guide/progress/2026-10-17-v6/README.md).
