# V95: held32 four-segment SIMD transport (CPU-ready, GPU UNRUN)

A is the frozen V44 public two-kernel helper. B preserves its BF16 activated prework, raw FP32 Q/K plus scales, FP32 g/beta/S, safe compile policy, canonical `[1,48,128,128]` state and both held32 local arithmetic loops byte for byte. The state kernel has the new name `gdn_simd4_segment_cached_state_v95_build1`; V44's unchanged prework name stays intact.

The `(32,4,1)` threadgroup remains. Required x-fast lanes map `sp=lane/8`, `dvLocal=TG.y*8+lane%8`; each SIMD now holds four contiguous K segments for eight value columns. Each kv/acc merge starts at `+0.0f` and adds the four original partials in sp0→sp1→sp2→sp3 order. Both shuffle loops are uniform, including the acc merge; only the o store is guarded by sp0. The state threadgroup array and three barriers disappear. Every state read/write retains `base+d*Dv+dv`. The acc merge also runs in sp1..3, whose extra merged scalar is discarded.

This is distinct from V27's held4/transposed-state/SIMD-sum profile and V29's held4/per-term shuffles. It preserves V44's local arithmetic tree, but compiler contraction/register/spill behavior is unobserved. One SIMD S instruction now accesses four separated 32-byte spans rather than one contiguous 128-byte span; actual coalescing, cache behavior and speed are unknown. No claim of numerical equivalence or benefit precedes the actual GPU gate. V93's old-gold failure remains unchanged and is not relaxed by this candidate.

The stdlib CPU checks reverse the state substitutions back to the exact donor string, check unchanged wrapper AST apart from the PSO name, enumerate unique state ownership and ascending source lanes, validate fake launch metadata and reject shape/dtype mismatches. These do not compile Metal or observe native reflection. Expected S slots 6/9 are ABI expectations only; future native integration requires real reflection and a new whitelist. No native/model connection or performance driver is provided.

The numeric gate first runs a qualification-only uint32 lane probe and requires observed `lane==TG.x` across all 128 threads. It then uses the original 15 immutable banks (9 real, 6 synthetic), with fixed original S/g/beta and no output feedback. One externally materialized BF16 `nn.silu(u)` is shared by A/B. Their public two-kernel APIs produce 30 finite FP32 o/S outputs compared as raw bytes, preserving signed zero. All original inputs plus activated input must remain byte-identical in all 15 cases. Partial failures retain truthful completed counts and the first mismatched A/B output bytes. This is local numerical qualification only, with no timing/full-model claim.

CPU checks (no MLX/NumPy import):

    python3 -B /private/tmp/minivllm-engine-optimization-20261014/v95-gdn-simd4-segment-build1/check_cpu.py
    python3 -B /private/tmp/minivllm-engine-optimization-20261014/v95-gdn-simd4-segment-build1/numeric.py

Root-controlled GPU qualification only (output directory must be new and absolute):

    env MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v95-gdn-simd4-segment-build1/numeric.py --execute-gpu --out /private/tmp/minivllm-engine-optimization-20261014/v95-numeric-ROOT-NEW

The source pins retain the complete V44 pin subset, its actual local PASS receipt, V27/V29 historical source identities and all new executable/source/proof/license files. The 60 saved bank binary hashes and shapes are rechecked against their pinned bank.json files before either CLI mode. MLX-derived material retains LICENSE-MLX-MIT. No old evidence is edited.
