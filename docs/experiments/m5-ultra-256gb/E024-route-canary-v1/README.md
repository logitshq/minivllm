# E024 route-preserving graph: first real canary passes

2026-10-17. **Only B1/used257 is qualified. No performance point.**
The new default-OFF candidate chooses attention bound1023 below1024 and1536
above it. This cycle tested only the lower bucket, with physical K/V1536,
gates/norm/fuse/core compile OFF, original Qwen3.8-27B NVFP4/BF16 target and
verified first256 tokens of the unchanged D fixture. Both arms share unchanged
parameter values and receive independent evaluated copies of the same state.

| Comparison | Independent original-byte audit |
| --- | --- |
| Native eager257 → new compiled1023 | All129 tensors equal; zero differing words or bytes |
| Full logits +128 K/V/conv/S state | 89,803,264 original-dtype elements;255,104,000 bytes per arm |
| Each arm's131 inputs | Canonical, before and after all byte-equal |
| Actual candidate call | One compiled callable invocation, key1023, 16 K/V backings1536 |

All saved arrays are finite. Both greedy outputs are1559; this is additional
information, not an equality substitute. 1349 model parameter value/shape/finite
SHA receipts are unchanged before/after prefill and each arm. Initial independently
loaded weight/prefill equality is device-recorded evidence: successful weight and
prefill payloads were not saved for independent offline comparison. Actual model
shard/source checks were performed before and after by the frozen producer.

The raw auditor ran once, without repair/retry: [readable result](independent-numerical.json).
Its seven raw files contain913 tensor records and1,783,330,137 bytes including
headers. The audit re-read original bytes; it did not merely trust equality flags.
The producer ran35.05068975 seconds and reported32,642,042,006 peak MLX bytes.
This includes loading, hashing, serialization and graph compilation, and is
**not inference throughput or latency**. Ten resource samples: pressure1,
zero swap growth, Swapouts/Pageouts0, Swapins+4, Compressions0, Decompressions+1746.

Original service11548 was stopped at17:03:08 UTC; experiment owner13596 and
child13629 were reaped. Restoration finished17:04:28 UTC asPID13908, with original
persisted settings/model and fresh MTP/resident PLE log evidence. Independent
[restoration audit](restoration-audit.json) passes. Warm KV and exact hidden
runtime state are not restored. The immutable [RESTORED journal](restored-journal.json)
is the next cycle's predecessor. The first read-only preparation command missed
the fixed proposal-hash flag and refused before journal creation; the corrected
command changed no code or plan. All attempts are in preregistration evidence.

## Evidence storage

`evidence.tar.xz` contains677 small files (438,264 bytes), including exact plans,
source copies, raw service/resource observations, execution and audit receipts.
Archive SHA256: `fc57c287e89b332ce02e605a835aebb2a47f27d6b88cbacb804db9b7cce6d24d`.
The [manifest](manifest.json) records every source path, size and SHA.
**Seven full-state safetensors remain local and are not in the Git archive.**
Remote-only readers need those listed originals or a rerun for a complete raw
re-audit. All decompressed archive-member hashes and all original file hashes,
including the local1.783GB tensors, were verified unchanged after packaging.

## Remaining gates

B1..8×used257/1023/1024/1280/1536 (40 points), actual cross-boundary and mixed-row
chains, B1→B8→B1 reuse, parameter replacement and ahead/retirement lifecycle,
then current gates-ON full D512/HTTP checks. Observer-OFF throughput versus full
POST→DONE latency comes only after those gates. Passing this canary does not
establish higher-bucket correctness, a speedup, or dominance over oMLX/Lithos.

[Pre-registered implementation](../../../../bench/m5_ultra_256gb/route_decode_graph/prereg-canary-v1/README.md) ·
[Tutorial](../../../optimization-guide/tutorials/024-decode-graph-dispatch.md) ·
[Current D512 performance chart](../../../optimization-guide/progress/2026-10-17-v6/README.md).
