# GDN source plan — CPU only, not an implementation freeze

Installed chain: omlx/engine/vlm.py:1819/2050/2167 VLMBatchedEngine loads mlx_vlm; qwen3_5/qwen3_5.py:11/56 binds LanguageModel; language.py:982/1103 calls local gated_delta_update, whose cache bridge is gated_delta.py:543–565 and cache.py:926–952. The direct T1 route is gated_delta.py:298–355/581–595; packed requires T>1. Runtime patches, actual reflection, PSO compiler flags and dispatch materialization were not observed here. Exact installed paths/SHA are in source-map.json.

| B1,Hk16,Hv48,Dk=Dv128 | current44 | installed vendor generic T1 |
|---|---|---|
| Physical FP32 state | [B,Hv,Dk,Dv] | [B,Hv,Dv,Dk] |
| State grid / TG | (128,192,1)/(32,4,1) | (32,128,48)/(32,4,1) |
| Source groups / threads | 192 /24,576 | 1,536 /196,608 |
| State registers / thread | held32 | held4 |
| Dot reduction | 4×32 serial, ascending4segment merge | 4serial/lane +simd_sum32 |
| Source TG barriers | 3/group | none;2SIMD sums |
| Logical S read+write | 3MiB+3MiB/layer | same; T1 offers no state-byte reduction |

48 layers logically read+write288MiB of S per decode step in both cases. This is a source-byte count, not measured DRAM traffic. Vendor has more threads/groups, shorter per-lane serial chains and no TG barrier; source alone cannot establish which is faster. Vendor normalization/gates are upstream operations, so a one-kernel state is not proof of fewer total GDN dispatches than44's prework+state.

Vendor math is different: language.py:1038–1042 uses RMSNorm(eps1e-6) then scalar scaling. Ignoring rounding, this corresponds to sum+128e-6, whereas44 uses sum+1e-6; BF16 RMS/scalar materializations also differ. gated_delta.py:71 decays each FP32 S before key dot, versus44's log-g exp applied after the old-state dot; vendor emits BF16 y (:86), while44 emits FP32 o and its model casts before GatedRMSNorm (gdn.py:217). Gate dtype/casts differ at source too and must be bound explicitly; no whole-oMLX equivalence claim. Vendor leaves compile_options omitted;44 explicitly uses safe. Effective compiler behavior is unknown.

V27 already used compact heads, held4, SIMD sums and persistent transposed physical state, but preserved old normalization/decay expressions. V29 used held4 plus per-term shuffles to emulate old serial reductions. Neither is the following proposal. Existing v*-gdn* helper scan finds no held[32]+simd_shuffle combination; scan scope is recorded, not a proof about unrecorded experiments. V60 exp, V61 tile64 and V85 scaled-ready STOPs stay unchanged.

**Next smallest candidate, old44 profile:** keep original prework byte-exact, canonical [Dk,Dv] state, held32, every local kv/update/acc expression and ascending4segment totals. Keep TG(32,4,1) and grid(128,192,1): lane=x, sp=lane/8, dvLocal=y*8+lane%8. Each SIMD then contains4sp×8dv; use four uniform simd_shuffle(part,(lane%8)+i*8), i=0..3, for each reduction. Remove the shared red[128] and3TG barriers; all lanes execute shuffles before only sp0 stores o. A 3D equivalent is TG(8,4,4), grid(32,192,4), lane=x+8y, dvLocal=8z+x, but the unchanged2D launch is simpler. Actual lane mapping still requires a tiny GPU probe later.

Memory tradeoff: old SIMD accesses one contiguous128B S slice per instruction; the remap accesses four32B slices separated by32 key rows (16KiB). Across the full128-thread TG logical addresses/bytes are unchanged, and four SIMDs collectively cover the same lines. Coalescer/cache behavior may penalize the split access; do not assert fourfold DRAM traffic or guaranteed speed. Register pressure remains held32; compiler allocation/contraction can change, so strict local bytes remain mandatory.

**Vendor-profile alternative, deferred:** independent A must directly call the installed normalization +generic T1 kernel (ops only as secondary reference); B may port the same vendor math, not relabel V27 tolerance as old-gold strict. Physically transpose S once at the prefill/decode boundary and keep that storage thereafter; equal128×128 shapes require an explicit layout/profile tag. Any conversion back is only at an external comparison boundary, not per step. BF16 Q/K and y, decay-before-dot, eps and gate input/compile policy must be jointly declared. New native PSO/schema/whitelist would require actual observation. This is not implemented or adopted. Actual93 vendor plain-RMS profile failed old warm32 IDs after gamma preparation; preserve that FAIL and do not relax or combine it implicitly.

Only source plans were written: no shader, GPU import, benchmark, new performance result or version reservation. The authorized follow-on95 CPU prototype is separate and retains old44 math; any performance claim awaits actual numerical and timing gates.
