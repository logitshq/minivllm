V92 is the default-OFF **vendor RMSNorm leaf rounding** profile; it changes plain RMSNorm only. It does not represent whole oMLX semantics or pass old38/44 gold.

Candidate: `candidate-v92-vendor-rms-leaf-build1-root` (27py), parent exact candidate44. Only `model/layers.py` differs. Original GatedRMSNorm and GDN are unchanged; removing the profile branch restores the old RMSNorm AST.

Root GPU preparation, immediately after existing model load and before ANY token:
```python
profile = vendor_reference.prepare_model(model, mx=mx, np=np, nn=nn, device_info=mx.device_info())
profile.set_enabled(True)
with profile.reference_context():
    # Build/evaluate/synchronize independent A prefill or decode here.
    ...
# B runs normal candidate path with prepared gamma; no legacy GDN binding.
```
`entries` is 161 frozen GammaEntry records: name, norm, source_weight, raw_gamma, cached_gamma, reference, checkpoint_key, raw_sha256, cached_sha256, nbytes. `owners()` exposes322 actual handles with unique labels/roles for external Shared/native read-only pins. Original161 offsets remain among1349 model parameters. The two gamma arms total2,674,688 bytes; caller keeps both alive and independently verifies distinct addresses/bytes. `metadata()`/`snapshot()` contain only serializable primitives. `validate_current_bindings()` checks unchanged model/weight ownership; `set_enabled()` accepts only bool. Context supports repeated nonnested entry and restores the plain class method in finally.

Factory reads only161 detached small raw ranges (1,337,344 bytes) plus three headers; no shard/model load. A raw gamma comes directly from checkpoint, B from live loader offset. Actual prepare checks every gamma byte, dtype/shape/finite and original parameter identities. No per-token gamma arithmetic. Weight identity/signature changes invalidate B and fall back.

CPU check: `python3 -B check_cpu.py --inert`. This imports no MLX/NumPy and proves source/guard/context mechanics only. Runtime numerical/reference/native qualification belongs to V93; no GPU work, inference, timing or adoption was performed here.
