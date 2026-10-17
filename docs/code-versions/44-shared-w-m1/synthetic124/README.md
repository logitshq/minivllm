# V124 shared-W M1 local numerical gate

CPU-prepared only; no GPU qualification or performance claim. Frozen source is read from `../v124-shared-weight-m1-source-build1`, manifest `a633fa7557edccb725101150772c4ea0fd7a492e889dad8e60be91a9fb967e9f`. No engine/model/checkpoint load or source mutation.

Four actual representative (N,K) shapes: (48,5120), (17408,5120), (5120,6144), (5120,17408), T=2/4/5. Three purposeful banks plus one T5 row permutation per shape give40 cases,80 full-matrix strict comparisons,152 rows per arm. They cover random original packed weights, all16 E2M1 nibbles, finite signed E4M3 scales, signed zeros, lane/block sparse boundaries, repeated/permuted rows, BF16 subnormals and +/-2^40 cancellation. `protocol.json` lists five unsupported-by-this-test actual shapes, including the large LM head. N48 is a complete SG2/R4 tile; partial/other N is rejected by the frozen helper, not silently run.

For each X row, compute a fresh native `mx.quantized_matmul(M1)` reference. First compare the verbatim original-expression custom control, then shared-W; all BF16 bytes including signed zero must match. Each matrix record contains per-row hashes. Native nonfinite fixture is classified INVALID_FIXTURE, not a candidate mismatch. First failure saves `first-failure.json`, both full BF16 outputs and input X bytes, seed/shape/bank+weight/scale hashes; remaining cases stop. Complete W/scales/X owner/bytes metadata is checked after each arm and in finally. Observed strides come from MLX's zero-copy memoryview of the equal-width integer BF16 view; there is no asarray-copy fallback.

Largest synthetic W+scale payload is50,135,040B. Worker sets1GiB MLX memory limit and64MiB allocator cache limit, rejects observed active/peak above1GiB, and records raw resource snapshots before derived/guard output. This is an allocator active/peak bound, not a total-RSS or physical-memory claim. Snapshot points: before weights, after each bank, finally. No timing/adoption loop exists. Root owns outer resource/timeout/PID/reap protection and all GPU execution.

Default inert, no MLX/NumPy imports:

    /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v124-shared-weight-m1-localqual-build1/localqual.py
    /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v124-shared-weight-m1-localqual-build1/check_cpu.py

Root-only actual CLI (OUT fresh absolute; root's owner must wrap this worker):

    /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v124-shared-weight-m1-localqual-build1/launch.py --execute-gpu --out ABSOLUTE_NEW_OUTPUT

Launcher removes **all** inherited selected keys using the frozen common selector predicate, then supplies exactly50/50/20/0 to the exec'ed worker. No DYLD/Metal overrides, no raw environment values logged. Do not set controls on an owner that requires its parent-selected environment empty; let this child launcher apply them. `launch.py` itself defaults inert.

Output: preimport identity, runtime, incremental cases/checks/readonly, original resource raw plus derived resources, and finally `receipt.json`. Only `PASS_LOCAL_V124_SHARED_WEIGHT_M1_STRICT_SYNTHETIC` with40/80 counts and all readonly/resource gates passes. Any error returns1 and preserves first failure; even all-pass remains local synthetic evidence, not the required497 actual-operator/fullmodel/samefrontier or HTTP qualification. No actual PSO/occupancy/DRAM evidence is inferred.
