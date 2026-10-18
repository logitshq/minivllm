# B1 native GDN input projection pack — source prepared, unqualified

2026-10-18. Default OFF. Ten fake CPU test methods are written but **not run**.
No MLX import, GPU execution, numerical qualification, model integration or
performance claim has occurred for these files.

## Minimal API

```python
# mx and ActualLinear come from the future source-pinned qualification caller.
pack = candidate.prepare(mx, (gdn.in_proj_qkv, gdn.in_proj_z,
                              gdn.in_proj_b, gdn.in_proj_a),
                         enabled=True, linear_type=ActualLinear)
outputs = pack.project(x)  # BF16[1,5120] -> qkv/z/b/a; otherwise None
```

Omitting `enabled=True` returns `None` before inspecting operands or the backend.
Enabled preparation requires exact original Linear instances, NVFP4/group16,
K5120, output rows10240/6144/48/48, U32 weights and U8 scales, and no biases.
It concatenates output rows, then **explicitly materializes the two derived
arrays with one `mx.eval` in preparation**. Preparation must happen fresh and
idle, outside model execution, tracing and measured regions. It never rebinds
the original parameters. Per group, the two derived arrays contain47,462,400
bytes; this is logical payload, not a measured memory peak.

`project` calls the original `mx.quantized_matmul(x, weight, scales, None,
transpose=True, group_size=16, bits=4, mode='nvfp4')`, then splits at
10240/16384/16432. It requires BF16 outputs without a cast. B>1 and other input
shapes/dtypes return explicit `None`, so the caller can execute original
Linears. No fallback runs implicitly; native errors propagate. There is no
custom qmv, output projection change, normalization, recurrence or state code.

A prepared pack is a **one-shot derived snapshot**, not a hot cache. Each later
qualification case must rebuild from current parameters. There is no source-id
freshness check, in-place `__setitem__` detection, captured-model support,
automatic replacement refresh or model hook. Source parameters must remain
stable during preparation and use. Fake replacement tests only check that a
new preparation reads new operands; they do not qualify mutation or tracing.

## Why this is distinct, and what is already known

Current D512 fixtures have886–978 prompt tokens. Original
[`CompiledDecode`](../../../tutorial_code/miniengine/engine/compiled_decode.py)
rejects used>256 before its packed path. The qualified E024 long callable calls
the model without `_decode_projections`; both normal GDN and the qualified
prework implementation retain four independent native qkv/z/b/a QMM calls.
Packing-only would remove three such operations per GDN layer, not weight
traffic or a proven number of command buffers. A change in output N can still
change native reduction/dispatch and therefore BF16 bits.

[M4 v20](../../../docs/experiments/18-projection-packing-evaluation.md) **already
tested native QMM GDN4→1**, with120 byte checks passing but only+0.147238%
improvement and4/6 winning pairs: performance STOP. It was not a custom-kernel
failure. This new M5/real-activation hypothesis does not erase or reinterpret
that result. The old M4 adapter also rebound original weights to slab views;
this interface preserves originals. E004's larger-batch failure bundled
packing, custom qmv and other fusion changes; it does not isolate this B1
operation. The repository's E019 is drafter history-tail, not projection
packing. See [the source review](../../reviews/long-d-gdn-projection-packing.md)
for the retained negative evidence and exact route distinctions.

The next device gate, not implemented here, should prepare afresh per case from
real early/late D512 B1 activations at GDN0/24/47. Compare all four original
native Linear outputs against the packed outputs in original BF16 bits, with
finite/input/parameter checks and first-failure raw retention. Do this before
feeding any state kernel. No random-only check, per-row M1 B>1 oracle or
successful metadata mock grants numerical, full-model, HTTP or performance
qualification. Full-model/capture ownership is intentionally deferred.

Future CPU command (not executed):

```sh
python -m unittest bench.m5_ultra_256gb.native_gdn_input_pack.test_candidate_cpu
```
