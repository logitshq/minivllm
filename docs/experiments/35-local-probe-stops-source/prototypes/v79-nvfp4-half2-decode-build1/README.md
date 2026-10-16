V79 paired FP4 half2 decode; numeric-only.

The small integer helper is adapted from Lithos NVFP4_DECODE1, Apache2. MLX R4/SG2 qdot association, scale placement, K loop, SIMD reduction and final BF16 cast stay unchanged. Two exact decoded half2 values replace four scalar decoder calls. This is not a claim that Lithos selects this decoder on M4, or that it is faster.

CPU exhaustive65536 packed16-bit words/262144 half-bit results match all16 E2M1 encodings including signedzero. GPU compiled behavior remains unqualified until strict36 comparisons. Fresh root must use --qualify-only; per-call wall micro forbidden. No model/engine/nativeICB/performance/adoption claim. Both source license texts retained.
