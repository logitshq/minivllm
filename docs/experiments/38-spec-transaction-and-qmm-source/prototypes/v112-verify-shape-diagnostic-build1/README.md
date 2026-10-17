V112 is a numerical diagnostic only. The frozen V108 candidate remains byte-for-byte unchanged. No new shader, native code, draft load, generation profile, performance path, adoption, or relaxed ID gate is introduced.

The input comes from the actual first V108 B.base transaction: prompt19, live generated anchor, and the saved real DFlash proposals `[6297,220,95761,95779]`. The saved anchor and posterior are assertions after live computation; saved expected continuation is never fed. The driver loads the same target once, performs the same prompt prefill, and requires all128 physical cache array signatures to match the original transaction's initial history. It then reproduces the original five-row posterior before any alternative operator is built.

Deferred class/function hooks retain the original operator input and output objects and return the original output. Original posterior/features/literal histories are evaluated first. The diagnostic then compares each original Linear and any used Embedding.as_linear against concatenated M1 calls using the same weights and exact same actual input rows. All target Linear modules must be observed once. It saves the actual first-layer QKV projection input as little-endian BF16 raw bits. No parameter arrays or trace attributes are added to NN modules.

Other comparisons use the same actual input: RMSNorm, GatedRMSNorm, SiLU, softplus, sigmoid; GDN beta/g against both explicit M1 and original compiled-decode M1 gates; RoPE against per-token offset; SDPA against each causal prefix with one query and either mask=None or decode's all-true Boolean mask. Q/K/V are already projected, normalized and rotated when SDPA is compared. The comparison does not substitute a new projection result upstream, so it isolates local batch-shape sensitivity rather than proving a repaired full recurrence.

All comparisons use raw-bit equality, including signed zero; finite checks, exact first differing coordinates/bits, differing counts and SHA are saved. Max absolute difference is diagnostic only. Kind/variant summaries include equal and unequal calls; the first differing captured operation is reported. Nested composite gates are recorded after their constituent pointwise operations, so order means captured call completion order, not an observed GPU execution order. The transaction commits keep0 after diagnosis and requires all128 original arrays/offsets restored; target1349 parameter object identities and raw bytes are checked before/after. Diagnostic intermediate input/output bytes must remain unchanged.

No root cause is claimed before actual execution. A reproduced batch/M1 difference establishes local sensitivity on this trajectory. Causality for the old96 IDs requires a later separate default-OFF ablation/full qualification. Actual PSO selection, dispatch count, reduction implementation and performance are not observed here. Norm BF16 rank>=2 uses the same norm_metal row kernel for both shapes; do not label this as a norm implementation switch.

CPU checks (no MLX/NumPy import):
```
python3 -B /private/tmp/minivllm-engine-optimization-20261014/v112-verify-shape-diagnostic-build1/check_cpu.py
python3 -B /private/tmp/minivllm-engine-optimization-20261014/v112-verify-shape-diagnostic-build1/diagnose.py --inert
```

Root sole GPU, after prior owner cleanup/release:
```
env -u DYLD_INSERT_LIBRARIES MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v112-verify-shape-diagnostic-build1/diagnose.py --execute-gpu --out ABSOLUTE_NEW_CHILD_OUTPUT
```

MLX/metal0.32.3, M4 Max applegpu_g16s, numerical limit34GiB/cache2GiB. These are guard budgets, not measured results. One target load, no draft model; this cannot measure speculative throughput or comparative memory. V108 failure and the old96-ID hard gate remain unchanged.
