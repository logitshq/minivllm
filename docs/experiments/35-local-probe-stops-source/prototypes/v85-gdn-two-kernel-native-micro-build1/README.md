# V85: frozen V81 GDN range, local native micro

This driver reuses V81's exact native library, helpers, 15 saved banks, immutable original S and no-feedback two-kernel paths. It changes no native or mathematical source. The actual V81 numeric receipt is pinned at `bd8e1b359462f210a5f539f0922a571b4f101eba730748b3a838b771f0f42f26`: 30 public +60 captured +480 native terminal raw-byte checks, actual reflection and S/S_new slots 6/9 verified. The new pin dictionary must preserve all original 56 path/hash entries as an unchanged subset and additionally pins this driver; it is not required to equal the old dictionary.

Before any performance work, a fresh process/marker binds that exact receipt path/SHA before MLX import and repeats V81's original numeric preparation. The original `gpu` function AST is identical after removing the one conditional handoff to `performance`; its original numerical statements, closures, live owners, arena and read-only checks are unchanged. No new native library is built or loaded by CPU checks.

Timing follows the previously frozen plan: each block primes 128 groups, then times one FFI call for 1024 groups ×30 dispatches =30,720 dispatches. The 15 banks rotate through two dispatches each; every state barrier and trailing group Buffers barrier remains in the frozen native library. Schedule: AA, BB, AB, BA, AB, BA, AB, BA, AA, BB. Gate: at least 5% arm-median reduction and 5/6 positive pair wins; each control ratio within 5% and each-arm control median within 2%. Control failure takes precedence. A local pass is exploratory and never adopts the candidate.

The wall clock includes the native host encode/commit/wait call. Prime work, reports, copies/hashes, resources and read-only verification are outside it. Each timed block checks its final 15 banks ×o/S =30 FP32 arrays against the public A44 reference, giving 600 terminal checks over 20 blocks. This does not verify every dispatch output. Original captured inputs/outputs and native shape/constants remain read-only. First mismatch preserves both raw byte arrays and stops. Numeric GPU envelopes remain diagnostics; no full-model or HTTP claim follows.

CPU-only checks: `python3 -B check_cpu.py`, then `python3 -B micro.py --inert`. Actual GPU and micro execution have not been performed by this author. Root owns sole GPU use, process supervision, release and any actual result.

```sh
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
DYLD_INSERT_LIBRARIES=/private/tmp/minivllm-engine-optimization-20261014/v81-gdn-two-kernel-native-range-build1/libv81replay.dylib \
/Users/ryanli/code/minivllm/.venv/bin/python -B \
/private/tmp/minivllm-engine-optimization-20261014/v85-gdn-two-kernel-native-micro-build1/micro.py \
--performance \
--qualified-receipt /private/tmp/minivllm-engine-optimization-20261014/v23-root-control-build1/fullmodel-v81num-attempt1/child-output/receipt.json \
--qualified-sha bd8e1b359462f210a5f539f0922a571b4f101eba730748b3a838b771f0f42f26 \
--out ABSOLUTE_NEW_PERFORMANCE_DIRECTORY
```
