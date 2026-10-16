# V31 FP4 factor fold — CPU ready, no GPU result

Single L0 gate, N17408/K5120, native U32 payload/U8 group16 scales totaling50,135,040B. Keep original R4,64threads,2176groups,K512 loop,lane ownership,qdot expression/sum/update order and native payload layout. Header has exactly two edits: remove FP4 half `converted*=16384`, and return FP8 scale multiplied by16384 **in FP32**, then cast to the qdot scale type (float). No reordering, copied/requantized weight, activation operation, epilogue or engine integration. Prepare retains the same two input handles.

Donor mechanism: pinned Lithos5485ad6 `monolith/formats/nvfp4.py:91–118` defaultdecode3 returns tiny half→float values and folds16384 into block scales. Its documented2–25% shape results are M5Pro observations, not M4 evidence. These scalar/SIMD operations require no Apple10 accelerator instruction; actual M4 gain and half-subnormal conversion/FTZ behavior remain UNKNOWN.

CPU contract exhausts16 nibbles×254 finite E4M3 scale codes=4064 combinations, including536 signed-zero products. Single-weight FP32 value/sign matches exactly. Raw FP4 minimum nonzero is2^-15 (half-subnormal); folded scale range is±32 through±7,340,032, plus signed zero. Multiplying in half would overflow, so the implementation explicitly converts scale to float first. Codes0x7f/0xff are nonfinite and excluded. This scalar proof does not prove every qdot: finite BF16 activation0x7f7f with FP4 value2 and scale0x01 makes the old intermediate overflow while the folded path can stay finite. No universal all-finite-activation losslessness claim; actual fixed-input GPU strict checks remain mandatory. CPU report records this counterexample and GPU FTZ uncertainty.

Runner loads only the selected two tensor ranges. Gate order: native R4 vs native QMM on8 fixed synthetic +1 pinned historical real x; only after all9 pass, folded vs QMM and folded vs native R4 on the same9. All27 compare shape/dtype/finite/raw bytes, preserve input readonly before/after qualification and timing, and STOP before timing on the first failure. Cold original/native/folded calls and one no-copy prepare host/sync/allocator/RSS are recorded separately. No MLP activation/SwiGLU qualification is repeated.

Timing A=native QMM; B=factor-fold R4. Prime2048/repeat512,8 activation rotation,10-pair schedule AA/BB+AB/BA×3+AA/BB; every output eval, hashes after both blocks. Same controls each[.95,1.05]/median[.98,1.02], local gate≥5% and≥5/6 wins; no model inference/adoption. Longer prime is preset before execution following V28build2 experience, with no relaxed controls. AC/thermal0or1/pressure1/original resource guard, raw-first, explicit2GiB allocator cache, exact owned child birth/PPID/wait/reap/absence,170s child timeout; root retains external ownership/deadline/release.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/minivllm-engine-optimization-20261014/v31-nvfp4-factor-fold-build1/cpu_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/minivllm-engine-optimization-20261014/v31-nvfp4-factor-fold-build1/micro.py cpu --out /private/tmp/minivllm-engine-optimization-20261014/v31-nvfp4-factor-fold-build1/cpu-source-check1
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v31-nvfp4-factor-fold-build1/micro.py run --out /private/tmp/minivllm-engine-optimization-20261014/v31-factor-fold-ac-attempt1
```

Fresh output directory/sole GPU owner required. CPU author has not imported MLX, initialized GPU or run numeric/performance work. Frozen source must not change after handoff. New math-policy/shape/model work needs a new build and qualification.
