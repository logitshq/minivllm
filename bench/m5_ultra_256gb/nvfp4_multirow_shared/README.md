# Parallel-token NVFP4 shared-memory prototype

Status: **CPU mechanism checks PASS (7/7); device compilation, numerical equivalence and speed UNQUALIFIED. Default OFF.**

The first register-reuse candidate passed six real-weight numerical cases in E038
but lost every median speed comparison. This separate prototype tests whether
parallel token SIMD groups with shared global weight loads improve that result.
It does not modify the old candidate, probe, engine or existing qualification plan.

`candidate.py` SHA256: `d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e`.
`cpu-v1/` preserves the actual guarded test runner, command, source snapshots,
before/after hashes, stdout/stderr and result. Command executed:
`/usr/bin/python3 -I -B /tmp/run-nvfp4-shared-cpu-v1.py`.
MLX/NumPy/Torch imports and subprocess/network/service operations were blocked;
all seven tests passed, no errors/failures/skips, and source hashes were unchanged.
CPU tests verify metadata dispatch and indexing, not Metal execution or arithmetic.

## Implementation and source review

- Only BF16 X[T,5120], U32 W[N,640], U8 scales[N,320], T in 2/4/8,
  positive N divisible by 4, fixed MLX 0.32.3 / M5 Ultra / applegpu_g17d.
- Each group covers two output columns. There are 2T SIMD groups of 32 lanes,
  one for each (token,column): 128/256/512 threads. Grid `(32,N*T,1)` and
  group `(32,2*T,1)` produce exactly N/2 groups, with no padded boundary group.
- 1,280 U32 words plus 640 U8 scales = **5,760 bytes** of static threadgroup
  memory. The linear thread ID copies adjacent global words/scales; strided loops
  give each tile element one writer. Every thread reaches the same single barrier,
  then the tile is read-only. No thread returns or branches around that barrier.
- The packed U32 tile is aligned for a threadgroup U16 view. The exact U16 indices
  cover 0..2559 and scale indices cover 0..639. Activation indices cover each
  token's 5120 elements; output indices end at T*N-1. The N*K uint32 guard is
  conservative for all products used; these ranges are enumerated by CPU tests.
- Each lane keeps scalar accum/result. Original B1 k step512 / j four-word
  arithmetic, f4/f8 conversion, scale application, final simd_sum and BF16 cast
  remain ordered the same in source. `math_mode=safe` and FP contraction OFF.
  Decoding repeats per token from shared memory; there is no serial token loop.
- Unsupported explicit opt-in returns None; the caller must reject it. No hidden
  fallback, catch/retry, input mutation, eager eval, model patch or weight packing.
  MLX row-contiguous staging is enabled and must remain inside measurements.

Source review finds no tile bounds, write-race or barrier-divergence defect under
this guarded domain. Metal compilation and SIMD mapping remain device gates.
Shared-memory traffic, repeated decoding, barrier cost, occupancy at 512 threads,
and memory-bank behavior can erase the intended benefit. The logical global-load
reduction is not an estimate of measured DRAM traffic or speed.

## Smallest actual device screen (root-owned)

Reuse the frozen 155-line E038 `nvfp4_multirow_reuse/real_projection_probe_v1.py`
as a new adjacent `real_projection_probe_v1.py`, replacing only its adjacent
candidate SHA from `ee9c5eef0e1282125310ee6661be1ec5ce64dc93a82add48a6d6c17f8958816f`
to `d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e`. The prior probe SHA is `10753926e5fff26053c079ba56b22fba4d78140a59364dc596ac41b0bc8d8c21`.
Its relative ROOT and adjacent candidate loading already work in this namespace.
No new owner, model loader, numerical oracle or timing framework is needed.

After root verifies that one-literal adaptation, the existing lifecycle owner can
run this argument array with its normal isolation, external resource monitoring,
direct Popen handle, deadline/reap and service restoration:

```text
/Users/ryanli/code/minivllm/.venv/bin/python -I -B
/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/nvfp4_multirow_shared/real_projection_probe_v1.py
--run --out /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/nvfp4-shared-real-projection-v1
```

The proposed child output must be fresh. The probe retains the 120s hard deadline,
4GiB allocator limit, pinned real head/up projection weights, synthetic BF16 X,
all original output-byte/finite checks before timing, native/custom B1 references,
five warmups and five groups of twenty paired AB/BA samples per baseline with
fresh graph/concat/staging/eval/synchronization included. Keep every failed compile,
numerical or timing attempt. A local projection PASS does not qualify full hidden
states, prefix caches, a full model, HTTP performance or a Pareto point. Only after
an actual favorable screen should the full-model head integration be re-sealed for
this new candidate and run by the lifecycle owner.
