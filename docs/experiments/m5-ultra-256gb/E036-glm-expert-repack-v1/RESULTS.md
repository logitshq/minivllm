# E036 result and scope

| Observation | Result |
|---|---:|
| Actual wrapper return code | 0 |
| Offline operation duration | 502.305588 s |
| Original / output shards | 62 / 58 |
| Original / output tensors | 113,446 / 4,960 |
| Backbone expert stacks | 378 |
| Conserved payload bytes | 203,976,457,080 |
| Producer | PASS_BYTES_ONLY |
| Independent check | PASS_HEADERS_INDEX_MANIFEST_ONLY |

The independent check reconstructed all 378 stacked groups directly from actual
original headers, in numeric expert order. Gate/up packed weights retain U32
[2048,512] per expert at 4 bits; down retains U32 [4096,320] at 5 bits. Scale and
bias groups retain F16 and group size 64. Other tensors, including shared-expert
6-bit entries and all 2,631 layer-45 tensors, remain passthrough. The
unchanged original config retains its original mixed-precision mappings. Runtime
MTP sanitization is still required; the repack does not eliminate that one stack.

Global endpoint counters use 16,384-byte pages. The exact raw vm_stat/sysctl
outputs remain in the archive; memory-endpoints.json contains all parsed values
and deltas.

| Global observation | Before | After | Delta |
|---|---:|---:|---:|
| Swap used (reported M / MiB) | 2023.25 | 2015.25 | −8.00 |
| Cumulative Swapouts (pages) | 782,442 | 782,442 | 0 |
| Cumulative Swapins (pages) | 135,038 | 135,410 | +372 |
| Cumulative compressions | 74,179,378 | 74,179,378 | 0 |
| Cumulative decompressions | 30,354,490 | 30,357,569 | +3,079 |
| Compressor occupied (pages) | 264,279 | 263,776 | −503 |
| Free pages | 111,408 | 94,152 | −17,256 |
| Speculative pages | 644 | 272,406 | +271,762 |

These are whole-machine observations over the copy interval. They do not assign
memory changes to the copy, establish a peak working set, or prove zero transient
pressure. F_NOCACHE was required per payload descriptor by the recorded code and
invocation; its effect is not causally isolated here. The copy does not alter or
control the original oMLX service. No service identity/health sample was taken.

Next evidence needed is a separately registered official-loader/memory trial,
followed by numerical inference qualification and performance measurement. This
archive contains no such trial, no benchmark and no mini engine GLM result.
