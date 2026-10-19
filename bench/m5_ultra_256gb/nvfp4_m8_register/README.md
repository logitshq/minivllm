# One native nv8 register tile

This default-OFF candidate directly instantiates the pinned MLX 0.32.3 `fp_qmv_wide_impl<DT,16,4,8,16>`. Native M8 dispatch currently uses two nv4 tiles. The new tile keeps eight accumulators per lane and shares each decoded weight group across eight vectors, with the same 64-thread group. It allocates no threadgroup storage and introduces no barrier. The previous 512-thread shared-memory candidate is preserved separately; its actual head/up screen was numerically correct but slower.

The 92-line candidate reads three installed native headers only after explicit ON, checks their hardcoded SHA256 values, and assembles the unchanged fp4/fp8 definitions, dependency helpers and full wide function. The wrapper's one helper call passes `x_shape[1]`, `scales_shape[0]` and `x_shape[0]` to the unchanged constant-reference parameters. The [official signature generator](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.3/mlx/backend/common/metal_kernel.cpp) declares shape metadata as `const constant int*`. CPU evidence includes copies of the actual three pinned headers; Apple MIT license is retained in `THIRD_PARTY_NOTICES/MLX.txt`.

Supported operands remain BF16 `[8,5120]`, U32 `[N,640]`, U8 `[N,320]`, N divisible by four, with guarded index range and M5 Ultra / applegpu_g17d / MLX0.32.3. Group `(32,2,1)` and grid `(32,N/2,1)` produce one nv8 tile and four output columns per group. The original helper retains each row's `g += 16`, float4 dot additions, scale accumulation and shuffle 8/4/2/1. Unsupported inputs return None; no fallback or model patch is installed. Contiguous staging remains explicit and included in timing.

Source identity does not prove generated native/custom binary equivalence. Register pressure, spills and occupancy may offset the intended reduction from two tile-level weight loads/decodes to one. Numerical and performance flags remain false pending an actual device receipt.

## Reused actual screen

The adjacent `real_projection_probe_v1.py` differs from E043 only in the candidate SHA. It uses the same fixed checkpoint, original head/up weights, full-shard/header/selected-slice integrity checks, deterministic synthetic BF16 activations, complete native-M8-versus-candidate BF16 comparison and finite checks. It retains five warmups and 5×20 paired AB/BA trials per case: 200 samples per case, 400 total. Child limits remain 120 seconds, 4 GiB allocator and 256 MiB allocator cache. Default invocation only describes the experiment.

Actual child command, only under root's bounded pause/restore controller and physical-memory guards, with a fresh output directory:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/nvfp4_m8_register/real_projection_probe_v1.py --run --out /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/m8-register-real-projection-v1
```

Native reference is a single `mx.quantized_matmul(x,w,s,**Q)` over the full M8 input. Independent M1 calls are not a valid oracle. Warm local projection timing does not reproduce a full-model working set; a hot up-projection loss alone cannot establish that sequential full-layer weights would also lose. There is no measured DRAM traffic or engine frontier claim.

## Executed CPU evidence

`cpu-v1`: **11/11 PASS**, zero failures/errors/skips, 0.103255 seconds. The guarded runner prohibits MLX/NumPy/Torch imports, subprocesses, network and signals. It checks OFF/runtime/shape rejection, single 64-thread launch and no retry, unchanged native helper bodies, source tamper rejection, output ownership/index bounds, per-row group order, independent shuffle-lane ancestry, inert/exact-native-M8 probe routes, complete BF16 comparison failure persistence and two exact 200-sample AB/BA fake timing cases. Sources matched before/after. Commands, source/header copies and stdout/stderr are retained. No Metal compilation or GPU work was performed by the implementation subagent.

- Candidate SHA256: `361fe483cde02a13c505287652249016f74b16d1d179c3741da843a3060cf71a`.
- Probe SHA256: `734bb1b7e14a054d2be90846bb3527d574f4c368f2b26dc26138a78b2c98820a`.
