# E027: reuse packed weights while preserving M1 arithmetic

2026-10-17. The first bounded M5 Ultra device screen passes all 33 local cases and
66 exact comparisons. The candidate remains default OFF and outside production.
The independent raw audit also passes. These are numerical diagnostic results;
whole-model correctness and HTTP performance remain unqualified.

For several activation rows, independent matrix-vector calls revisit the same
packed weight words and scales. A narrow alternative keeps those operands in
registers while processing two, four or five rows. Each row still follows the
original M1 accumulation order. This targets redundant operand loads without
changing quantization or introducing a different matrix-multiply reduction.
Actual savings depend on cache reuse, register pressure and occupancy; the
number of input rows is not a predicted speedup.

The [M4 publication](../../code-versions/44-shared-w-m1/README.md) supplied the
unchanged Metal source. Each lane reuses four packed uint16 words and a scale
per output row and K512 step across the activation rows. The original qdot
expression, lane accumulation, `simd_sum` and BF16 store remain intact. Source
similarity and M4 results could not establish how the M5 compiler would execute
it, so this attempt built a separately pinned M5 adapter and a fresh device gate.
It uses the original uint32 weights and uint8 scales, with no repacking or
parameter replacement. Unsupported inputs raise; required points cannot pass
by silently falling back.

The [candidate](../../../bench/m5_ultra_256gb/shared_w_m1/candidate.py) owns one
prepared object per operator. OFF returns before accessing the backend or
operands. ON lazily constructs one Metal callable and reuses it for rows 2, 4
and 5. The callable counter measures Python construction, and the launch counter
measures returned lazy graphs. Only evaluation and complete output-byte checks
establish that device work completed. No activation or output is cached in the
prepared object.

The [screen](../../../bench/m5_ultra_256gb/shared_w_m1/screen.py) first computes
each row independently with native `mx.quantized_matmul`, then concatenates the
results. It checks a separate original-expression control against that oracle
before admitting the shared-W candidate. Both comparisons require identical
shape, dtype and every BF16 byte, including signed zero, with finite outputs.
An invalid native fixture or a failed control stops the run before it can confer
candidate qualification.

| Packed projection dimensions `(N, K)` | Synthetic cases | Real-weight cases | Rows per case |
|---|---:|---:|---|
| `(48, 5120)` | 9 | 3, layer-0 `in_proj_a` | 2, 4, 5 |
| `(1024, 5120)` | 9 | 0 | 2, 4, 5 |
| `(10240, 5120)` | 9 | 3, layer-0 `in_proj_qkv` | 2, 4, 5 |

The three synthetic banks are random, signed-code/boundary, and extreme
cancellation inputs. They include repeated rows, signed zeros and finite
boundary values. The real weights come from the already verified Qwen3.8-27B
NVFP4 layer bundle; their activation inputs are also synthetic. This does not
exercise a complete GDN layer, actual model activations or the other six
geometries accepted by the adapter. NVFP4 remains group-size 16, four bits, with
no bias. MLX is pinned to 0.32.3 on Apple M5 Ultra / `applegpu_g17d`.

Do not infer device layout from a host buffer. A pinned Metal metadata probe,
with `ensure_row_contiguous=False`, reads original GPU shapes and element
strides for W/scales before and after each bank and X before and after each
case. Only those readings reach the candidate's layout guards. Host memoryviews
serve a separate purpose: recording original bytes. All X/native/control/shared
arrays are saved, while complete W/scales bytes are shared across the three
cases of each bank and saved again after the bank. Every case checks original
operand identity, metadata and byte hashes after execution.

The completed receipt contains 33 passing cases, 66 exact comparisons, 121 row
vectors and 486,464 BF16 output elements per implementation. All 33 read-only
checks cover X, W and scales: 99 operand records. Eleven prepared objects each
report one callable construction and three returned graphs. All 110 GPU
descriptor records and 68 self-footprint phases are present. These counts were
first recomputed from screen and owner receipts. A separate independent auditor
then verified all 319 binary files, including every full output byte, original
operand evidence, actual descriptors, physical phases and lifecycle records.

| Observation | Maximum | Registered limit |
|---|---:|---:|
| Sampled MLX active bytes | 29,900,800 B (28.516 MiB) | Active + cache ≤ 1 GiB |
| MLX historical peak | 30,097,544 B (28.703 MiB) | 1 GiB |
| MLX cache | 0 B | Cache limit 0 |
| Derived physical current | 409,830,192 B (390.845 MiB) | 2 GiB |
| Physical peak observation | 413,172,552 B (394.032 MiB) | 2 GiB |

Each raw physical reading is bounded separately; current is the larger of top
and auxiliary current, and peak is the larger of top and auxiliary peak. These
are observations, not OS hard caps. Final MLX active and cache bytes are both
zero; process physical memory is not zero. The six external resource snapshots
retain pressure 2, swap use reported as `445.06M`, Swapins 75,930 and Swapouts
316,235 throughout, with at least 62.522 GiB of real free pages. This stable
pressure-2 admission was separately registered for the bounded numerical
diagnostic. It does not relax the pressure-1 model/performance gate.

The direct child, PID 45274, exited 0, was waited/reaped and passed strict PID
absence. The outer lease restored the same service identity, PID 74384, from
stopped to sleeping. Instrumentation, compilation, serialization and monitoring
are included in the diagnostic lifecycle. Its elapsed duration is not a kernel
benchmark or token-throughput measurement.

The remaining path to the requested **throughput versus complete-request latency
Pareto frontier** is explicit. Qualify every geometry actually selected by the
eventual route, then propagate the candidate through full-model logits, raw
token IDs and recurrent/KV state under the same schedule, including retirement
and refill. Keep unsupported routes explicit and switches default OFF until
those gates pass. Only then run uninstrumented paired HTTP experiments with
fixed workloads, normal EOS, retained negative samples and independent fresh
processes. Plot total committed-token throughput against POST-to-DONE request
latency; speculative acceptance, concurrency and scheduling can outweigh a
local projection change. E027 supplies no new performance point and establishes
no advantage over oMLX or Lithos yet.

The run is identified by plan SHA
`840b1eb0a4e6127bf6fb5e322fdca82d34fabcd7f92712b429b25e596f0f2351`
and screen SHA
`d01f28e36b07176e73830b424f3ab65113f1263cc1f810678f2c88b50c11dacf`.
The [raw screen receipt](../../../.artifacts/m5-ultra-256gb/shared-w-m1-v1-owner/child/screen/complete.json)
has SHA `f0e4906ceeb3c443b7b7e01419840b952f95837d09b7b3c7bc119ff51614602c`;
[resource observations](../../../.artifacts/m5-ultra-256gb/shared-w-m1-v1-owner/resources.json)
and [direct-child release](../../../.artifacts/m5-ultra-256gb/shared-w-m1-v1-owner/child/release.json)
retain the lifecycle evidence. The completed [independent raw audit](../../experiments/m5-ultra-256gb/E027-shared-w-m1-v1/independent-audit.json)
is PASS; its implementation and six corruption-oriented CPU tests are saved
alongside the candidate. Root separately compared all saved matrix bytes.
[Full archived evidence](../../experiments/m5-ultra-256gb/E027-shared-w-m1-v1/README.md)
retains successful outputs, resources and source identities.
