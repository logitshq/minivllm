# Two bounded decode experiments after E024 + GDN

2026-10-18. Source review only: no model/backend import, tests, device profiling,
payload hashing or service operation. Read the local oMLX checkout at HEAD
`328edc756010693e61f8bd4e47f00996e0758176` and Lithos at
`223658090ae549e59220bdec459957cbdad7926a`. These identify the inspected
checkouts, not a claim that every current source was used by an earlier formal
run. The frozen measurement manifests remain authoritative for those results.

**Prioritize a live flat-parameter binding experiment if the existing C1
diagnostic shows material callable/submission time. Keep native B1 GDN input
projection packing as a separate second hypothesis.** Neither has a measured
win here. Framework size is not evidence of a bottleneck.

## What the competitors actually implement

| Actual source | Consequence for this target |
| --- | --- |
| [oMLX packed eligibility](/Users/ryanli/code/omlx/omlx/patches/qwen35_packed_linear.py:579), packing at593–614, tuple projection at544–575 | This path requires **affine INT4/group64**, BF16 scales and quantization biases. It rejects our NVFP4/group16/U8-scale storage. Its shared projection call is also wired into `verifier_cls._linears` at756–773. It is not evidence that oMLX uses the same optimization on our target-only NVFP4 decode. |
| [oMLX Qwen3.5 decode prework](/Users/ryanli/code/omlx/omlx/patches/qwen35_gdn_prework.py:2053) | This branch retains qkv/z/gate projections, fuses convolution/normalization preparation, then recurrence/norm/output projection. That structural opportunity is already addressed by our qualified B1 prework. The cached broad decode plan at1479–1551 is specifically **Qwen4Exp**; do not attribute it to Qwen3.5/our target. |
| [oMLX static decisions](/Users/ryanli/code/omlx/omlx/patches/module_cache.py:47) | It caches derived decisions, but checks current module/child object identities on reuse and keeps referenced objects alive. Useful ownership design; not a license to cache old parameter arrays or assume a model-wide mutation epoch. |
| [Lithos projection IR](/Users/ryanli/code/lithos-metal/monolith/nn/linear.py:1), [GDN lowering](/Users/ryanli/code/lithos-metal/monolith/nn/gdn.py:96) | Row slabs and same-input siblings are real executable compiler concepts. However, its documented27B GDN uses FP8 qkv/z plus BF16 a/b, and its output epilogue/norm arithmetic is its own contract. Our target has native NVFP4 projections and different pinned reduction/rounding boundaries. |
| [Lithos NVFP4 tile packing](/Users/ryanli/code/lithos-metal/monolith/compiler/nvfp4_tiles.py:1), [compiler binding](/Users/ryanli/code/lithos-metal/monolith/compiler/static_fusion.py:444) | Codes and E4M3 scales remain losslessly represented in separate derived planes; the compiler actually binds these buffers to specialized kernels. [Executable recipe selection](/Users/ryanli/code/lithos-metal/monolith/backends/metal/m5_max_40c/serving.py:32) admits matching hybrid27B **with a drafter**, and selects long-context recipes. Target MLP/head tile layouts appear at `selected-contexts.json:194–225` and its long-context entries. This is more than dead experimental code, but not a same-arithmetic proof or evidence that target-only serving selected it. |
| [Lithos automatic GDN fusion](/Users/ryanli/code/lithos-metal/monolith/backends/metal/m5_max_40c/scheduling.py:12), [region predicate](/Users/ryanli/code/lithos-metal/monolith/compiler/gdn_fusion.py:30) | Automatic fusion requires Apple10/M5Max40, static eight rows, non-speculative, normalization commuting and accelerator ON; projections must be FP8/BF16 and output FP8. That gate does not establish M5 Ultra NVFP4 equivalence. |

Lithos's host-runtime distinction is concrete: [Engine:78–101](/Users/ryanli/code/lithos-metal/monolith/runtime/engine.py:78)
builds dispatch bindings and an ICB once; [native ICB construction:236–250](/Users/ryanli/code/lithos-metal/runtime/src/metal_core.mm:236)
encodes each operation once. [Runner:346–375](/Users/ryanli/code/lithos-metal/runtime/src/metal_core.mm:346)
replays several steps per command buffer and drains outstanding buffers. This
is persistent **program/buffer/command reuse**, not proof that one GPU kernel
runs forever. Its optional megakernel is a separate compiler transformation.
Importing that runtime would require a new backend/state/stop contract, so it
is not the proposed small minivllm experiment.

E024 retains `mx.compile(..., inputs=model)` at
[candidate.py:118](../m5_ultra_256gb/route_decode_graph/candidate.py:118).
The matching-tag MLX analysis in [the capture-overhead review](2026-10-18-long-decode-capture-overhead.md)
shows that a cache hit avoids the Python model forward but still flattens
current captured parameters, substitutes roots and constructs a lazy graph.
It must not be described as rerunning all64 Python layers. The source-derived
1349 parameter leaves plus131 explicit inputs are not a timing measurement.
The existing [one-pass candidate](../m5_ultra_256gb/long_decode_onepass/candidate.py:66)
only consolidates outer guard/state collection; it deliberately retains this
model capture cost. Native [ahead:82–103](../../tutorial_code/miniengine/engine/model_runner.py:82)
already overlaps successor construction/submission with current work, so saved
host time can be hidden rather than improving POST→DONE latency.

## 1. Flat *live* parameter slots, not a stale captured list

New default-OFF per-model experimental callable: resolve the fixed model's
array-owning dict slots once; read their **current values** into an explicit
flat input tuple at each call. During a trace only, bind those explicit tracer
arguments to the original slots, call the unchanged model/explicit-state
forward, and restore every slot in `finally`. Retain original bounds, cache
guards, gates/norm key, native math and ahead routing. Structural/module
replacement requires recreation or explicit rejection, as the existing
candidate requires for structural changes; ordinary array-value replacement
must still work immediately. A one-time copied parameter list without live
reads and tracer rebinding is incorrect.

Potential saving: recursive traversal of the model dictionary and non-array
metadata on every compiled call. It **does not** remove1480 input roots, MLX
lazy-graph reconstruction, GPU launches or weight traffic. Python slot reads
may cost more than MLX's C++ traversal. Tracing errors, nested compiled gate
calls, temporary wrapper replacement and retained references are real risks.

Fastest falsification: first use the existing profile OFF/ON overhead check to
separate outer submission and actual callable time. If that interval is small
or entirely hidden by ahead, stop this direction. Otherwise test one unchanged
real B1/used1023 state, then B1/1024, against original E024; require full logits
and128 state arrays exact, input/parameter preservation and no new eval/sync.
Run the existing42 changed/restored parameter calls, including B1→B8→B1, before
any serving claim. A short observer-OFF ABBA complete-forward test is the
go/no-go, not a CPU-only list-flatten microbenchmark. No public zero-cost epoch
or persistent-MLX-ICB API is assumed.

## 2. Native B1 GDN4→1 projection, independent of custom qmv

Only concatenate current qkv/z/b/a **output rows** and scales, issue native
`mx.quantized_matmul`, then split at10240/16384/16432. Keep K5120, all original
quantization values, BF16 outputs, prework/gates/recurrence/norm and native
out-projection unchanged. Use the same qualified prework setting in both arms.
The existing [PackedProjection:35–43](../../tutorial_code/miniengine/engine/packed_decode.py:35)
calls custom `qmv`; it is not this neutral experiment. Current long E024 does
not pass `_decode_projections`, while adopted short packing is bounded≤256.

Potential saving is144 fewer projection operations across48 GDN layers at B1;
weight bytes do not decrease and this is not144 guaranteed command-buffer
savings. Derived copies cost about2.122GiB across48 layers, plus preparation
temporaries. Current-parameter replacement must rebuild the affected pack;
do not overwrite the original parameters or silently reuse stale arrays.
Native dispatch/reduction can depend on output N despite identical formulas.

Fastest falsification: one real early/late B1 activation at GDN0/24/47; compare
all four projection outputs bit-for-bit **before** feeding the state kernel.
Then own-state three-step layer chains, input/parameter invariants, and the
full-model gates only if these pass. B>1 initially falls back to original
native batched projections. Preserve [M4 packing's +0.147238%,4/6 performance STOP,
E004's B16 mismatch/B32 regression, and E027's per-row-M1 mismatch](long-d-gdn-projection-packing.md).
Those results lower the prior probability of a large win; they do not license
different quantization, norm or reduction kernels as an equivalent replacement.

Both proposals remain hypotheses. No new throughput/latency point, gain,
combined-switch qualification or hardware bottleneck is established by this
review.
