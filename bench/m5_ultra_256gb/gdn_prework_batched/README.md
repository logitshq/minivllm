# B1–8 GDN prework adapter — source preparation only

This separate default-OFF adapter extends the frozen
[`gdn_prework_unbound/candidate.py`](../gdn_prework_unbound/candidate.py)
B1 shape gate to B1–8. **The new tests have not run. The extension has no
device, full-model, HTTP or performance qualification.** Existing B1 canary
and 37-request results do not establish correctness for the added B2–8 path.

`install(layer, native_gdn_module, enabled=False)` returns `None` before
inspecting either argument or importing kernels. An enabled installation is
an instance-only `_decode` scope with `restore()` and context-manager support.
Install on fresh idle target layers before requests/observers/graphs; drain
or tear down the model before restoration. Restore outer observers first.
There is no runtime CLI, model loader, owner, controller or registered plan.

The change keeps the five original native `Linear` calls and reads current
weight/scales, convolution and gate parameters on every call. It uses batch
dimensions in x/conv/S guards and z/output reshapes, then invokes the existing
`gdn_decode_fused.fused` once. No projection packing, per-row M1 loop, parameter
cache, tensor evaluation or new Metal arithmetic is introduced. The original
`_decode_gates` retains its B1-only compiled gate: B2–8 still use its ordinary
`_gates` fallback. Gates configuration OFF remains on `_gates` for every B.
The native order of conv/S replacement before norm/out projection is retained;
an exception is propagated and never retried against an advanced state.

The admitted source domain is M5 Ultra / MLX 0.32.3, ordinary target decode,
exact `BatchState`, BF16 x `[B,5120]` and conv `[B,3,10240]`, FP32 S
`[B,48,128,128]`, Hk16/Hv48/D128, the original NVFP4 group16/4-bit projections,
and norm reuse OFF. Other B, metadata, dtypes, contexts, quantization or backend
fall back to the untouched native method before candidate state mutation.
The layer receives no used-length argument; history length is not inferred
from its fixed-shaped state. This is a proposed long-decode experiment, not
a claim that all histories or arbitrary strided inputs have been qualified.
Do not combine it with E024, E028, speculative decoding or retained compiled
graphs under this preparation.

`candidate.PINS` verifies the same seven native sources as the original B1
adapter and also the real reference `kernels/gdn_metal.py`
(`51757224b18a424eaee5ebb7346075837bf768c289b8721f6279b6d844de13a9`).
Kernel functions are loaded from the supplied native module's actual package;
the production entry verifies source paths and loaded callable origins.
`make_decode(module, fused, enabled=False)` exists only for dependency-injected
CPU contracts. Its counters describe Python submissions/state commits, not
device completion, compiled graph hits or successful full forwards.

The 12 source-prepared fake tests cover inert OFF, all batch shapes with five
native calls and a single fused submission, unchanged gate delegation,
unsupported fallback, current parameters across B1→B8→B1, own-state chaining,
exceptions before/after commit, scoped cleanup, and changed reference-kernel
source rejection before import. They do not simulate numerical correctness or
prove native gate compilation. A later root-owned, guarded CPU invocation is:

```sh
python -B -m unittest bench.m5_ultra_256gb.gdn_prework_batched.test_candidate_cpu
```

The [staged design](../../reviews/2026-10-17-gdn-prework-batched-next.md)
starts with B2/B8 real long-history activations at early/middle/late GDN layers
and independent continuing states. Compare full native-batched output, conv
and S in original unsigned bits with finite checks. Then cover B3–7, mixed
histories, row changes and current-parameter replacement before full-model
37-request and real HTTP capture/replay gates. Preserve the E004 and packing
negative results cited there. No measured throughput or latency win is claimed.
