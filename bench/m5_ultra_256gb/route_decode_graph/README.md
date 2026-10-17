# E024 route-preserving graph canary

Update2026-10-17: the B1/used257 canary passed on the actual device and the
independent original-byte audit: all129 outputs exact, both131 inputs unchanged.
[Execution evidence and limits](../../../docs/experiments/m5-ultra-256gb/E024-route-canary-v1/README.md).
The candidate remains default OFF; higher-bucket/matrix/lifecycle/HTTP correctness
and performance are still unqualified. The old rejected graph and four-arm
isolation remain unchanged. Below is the source and pre-registered CPU contract.

`RoutePreservingDecode(model, enabled=False)` calls the original model. When
explicitly enabled, only the original pinned M5 Ultra/MLX0.32.3 NVFP4/BF16 model
and normal decode B1..8/used257..1536 are eligible. It uses bound1023 below1024,
otherwise1536, with the original explicit-state forward and captured model
parameters. K/V capacity must still be at least1536 for both buckets; this
canary does not expand eligibility to capacity1024. Insufficient backing, unexpected context/state or unsupported
backend uses original eager behavior. Compilation errors propagate. There are
two bound callable families; shapes can require additional specializations.

The mechanism and limits are in the [source review](../../reviews/2026-10-17-e024-route-preserving-bounds.md).
The claim to test is that valid keys retain the original 32-way or128-block
reduction order while added boolean-false padding skips arithmetic. This is
not a general guarantee for arbitrary masks/devices/head dimensions.

The only registered child implemented here is **B1/used257**. It loads two
normal independent original engines, checks complete parameter and256-token
prefill equivalence, then uses one unchanged model with two independent saved
canonical-state copies: A=eager257 and C=new compiled1023. Both arms preserve
all131 input arrays before/after, all129 output arrays (logits+128 caches),
finite/shape/dtype/unsigned bytes and actual graph-call witnesses. Parameters
are fully hashed before prefill, after prefill and after each arm; array handle
IDs are recorded but are not a value-equivalence gate. All original model and
source pins are rechecked. The canary uses the unchanged 2GiB allocator cache
limit and full-model resource policy supplied by its external cold owner.

Run only through a source-pinned external owner enforcing deadline, resource
observations and direct-child reap. The child CLI is:

```text
python -I -B qualify.py --plan PLAN --plan-sha256 SHA --out NEW_DIRECTORY
```

The exact five-field plan is `E024-route-canary-plan-v1`: `schema`,
`qualifier_sha256`, `candidate:{path,sha256}`, `e024_plan:{path,sha256}` and
`deadline_s` in (0,1800]. The inherited E024 plan is the unchanged SHA
`5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123`.
The actual source pins include the new candidate, frozen E024 qualifier and
candidate/model-source manifest, E023 qualifier and the frozen four-arm helper
`9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e`.
Runtime/OS/process evidence belongs to the external owner; this child does not
pretend its own observations are a memory hard cap or restoration proof.

`complete.json` schema is `E024-route-canary-result-v1`. `PASS` requires
`canary_qualified`, `diagnostic_complete`, `numerical_equal` and
`source_model_pins_rechecked` all true. Complete finite numerical differences
retain both original-dtype outputs and all129 comparisons, then return
`FAIL/exit1` with `canary_qualified:false`. Execution/invariant failures stop
and preserve the completed prefix. No successful result here grants the
remaining40-point matrix, boundary/mixed-row/batch reuse, parameter replacement,
ahead lifecycle, full D512 gates-ON, HTTP schedule replay, or a throughput versus
complete-request latency Pareto point. Those are separate subsequent gates.

CPU tests use explicit mocks and no MLX/model/service:

```text
.venv/bin/python -B -m unittest discover -s bench/m5_ultra_256gb/route_decode_graph -p test_cpu.py -v
```
