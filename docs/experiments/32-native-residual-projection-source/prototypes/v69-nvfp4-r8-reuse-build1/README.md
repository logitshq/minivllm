V69 independent local numeric and frozen micro.

R8 reuses each activation chunk across8 output rows, versus current4. Original two-SIMD64-thread TG. Native fp4/fp8/qdot/K loop/simd_sum and weight format unchanged. Increased row registers may hurt occupancy; no expected speed assertion. Lithos gemv_T RG loop is the scheduling idea, not a copied slab layout or a claim of its chosen M4 recipe.

First root run must use --qualify-only:36strict BF16 checks,9vectors, L0MLP gate only, no model/performance. Only after numerical PASS may run a fresh performance marker with predeclared prime2048/repeat8192,6AB/BA+4AA/BB, gain>=5%,5/6wins and original drift gates. Keep STOPs; no protocol retune. Both arms mx.compile; no engine/HTTP/Pareto claim.
