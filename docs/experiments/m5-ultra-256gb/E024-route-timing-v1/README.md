# E024 route-preserving graph: isolated forward timing

The eight preregistered configurations passed their complete original-bit output checks.
Within this single-process diagnostic, the compiled graph has lower median forward time
in every configuration. This is **not an HTTP throughput/latency result** and adds no
Pareto point. It compares our own eager and compiled implementations, not Lithos or oMLX.

Qwen3.8-27B NVFP4, fixed revision `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`,
M5 Ultra 256 GiB. Default-OFF candidate uses bounds 1023 and 1536, physical cache at
least 1536. Core compile/fuse and norm reuse OFF; gates OFF/ON explicitly separated.

| Batch | Used length | Gates | Eager median ms | Graph median ms | Forward rate gain |
| --- | --- | --- | --- | --- | --- |
| 1 | 1023 | OFF | 24.874 | 22.475 | 10.67% |
| 1 | 1023 | ON | 23.680 | 22.197 | 6.68% |
| 8 | 1023 | OFF | 40.624 | 38.528 | 5.44% |
| 8 | 1023 | ON | 40.595 | 38.414 | 5.68% |
| 1 | 1024 | OFF | 24.315 | 22.524 | 7.95% |
| 1 | 1024 | ON | 23.638 | 22.089 | 7.01% |
| 8 | 1024 | OFF | 40.469 | 38.367 | 5.48% |
| 8 | 1024 | ON | 40.565 | 38.297 | 5.92% |

Each configuration excludes four warm samples, then runs three fixed ACCA blocks:
six formal samples per arm. A=eager, C=compiled. Rate gain is eager median / graph
median − 1. All 24 within-process block mean ratios exceed one; these are not fresh
process repetitions or confidence intervals. Total: 32 warm and 96 formal samples.

The timer covers forward, evaluation of logits plus all 128 state arrays, and device
synchronization. Input copies, qualification, comparisons, hashing, and teardown are
outside the timer. Separately recorded forward+teardown intervals are not request
latency. No HTTP, sampling, server queue, or committed output-token rate is measured.

Each configuration first checks both arms against the original gates-OFF oracle.
The source-pinned numerical auditor verifies 1,548 qualification output checks,
2,096 input checks, 3,096 prefill checks, and 33,280 post-sample checks, plus unchanged
1,349-parameter receipts. It reads full hashes and device comparison records;
**this timing run saves no successful raw tensor payloads for offline byte comparison**.
The preceding 52-step matrix has its own narrower raw-payload verification scope.

The independent numerical auditor ran once after service restoration and passed.
The producer checked model-file identities before/after; the numerical auditor did
not independently rehash those large model payloads. Resource and restoration evidence
are audited separately. All 21 resource samples have pressure 1, thermal 0, unchanged
swap, and zero increments in Swapins/Swapouts/Pageouts/Compressions. Decompressions
increased 264. MLX peak 56,450,814,156 bytes includes the two diagnostic models and state;
it is not production memory. Child elapsed 90.565 seconds includes loading and checks.

Owner 19484 and child 19518 were reaped. The original oMLX service was restored as
PID 19870, started 2026-10-17 17:44:54 UTC; the independent audit verifies original
model/config and fresh Lightning MTP/resident PLE evidence. Warm KV and exact hidden
runtime state were not restored. Cold-cycle execution completed successfully.

## Reproducibility

The plan/code were committed as `b221b83` before execution. Frozen plans and execution:
[preregistration](../../../../bench/m5_ultra_256gb/route_decode_graph/prereg-timing-v1/README.md).

- [Numerical/timing audit](independent-numerical.json): SHA256 `86564bf21aae71c733f4349fc87cefd67d7accc36eaf8783ac7ade13b51ef821`.
- [Restoration audit](restoration-audit.json): `fabbadf4e94b4db33c7fa956b0ea55b31f41386b2290322ee27f73181215fb74`.
- [Resource audit](resource-summary.json): `548cd46c32a605f71df7347748648e75e5c458b59a01486fda7079df8173d151`.
- [Evidence archive](evidence.tar.xz): 706 small files, 800,396 bytes, SHA256 `22681d6c61b7ed44e3a472abb78b0a71562bd155f31d8b26c76481fc44a5c965`.
- [Manifest](manifest.json): SHA256 `ea9f387beb27b462184949d10fb28b2d5a52c9dc349ccca137dd8442c71ca0ba`.
- [Archive verification](archive-verification.json) checks every archived/original file and tree membership. No new raw tensor files are omitted from this timing archive because none were produced.
- [Figure, source and all individual samples](../../../optimization-guide/progress/2026-10-17-v7/README.md).

Next priority is same-workload Lithos D512 C1/C8 qualification and measurement.
Parameter replacement, ahead/retire, actual scheduler and HTTP gates remain necessary
before enabling or measuring this graph in serving. Existing native D512 points are
unchanged; no competitor dominance or global fastest claim follows from this diagnostic.
