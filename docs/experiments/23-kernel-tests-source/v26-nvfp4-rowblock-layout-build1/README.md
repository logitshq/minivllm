# V26 row-block NVFP4 layout — CPU prototype, no GPU result

The proposed change is a one-time lossless payload/scale transpose. Keep native R4, two SIMD groups/64 threads, K512 chunks, lane assignment, FP4/FP8 helpers, `qdot`, FP32 accumulation, `simd_sum` and the final BF16 cast. Only the weight/scale addresses change. This differs from rejected V22 R4→R2: R stays 4. No activation permutation, fusion, requantization, or engine integration is installed.

Native storage is `[N,K/8] U32 + [N,K/16] U8`. The new physical arrays are `[N/8,K/512,8,64] U32 + [N/8,K/512,8,32] U8`, flattened. A threadgroup's eight output rows for one K chunk become adjacent. Native loads already coalesce across lanes; this is a bounded locality/streaming hypothesis, not a fix for proven uncoalesced traffic. Extra integer-address overhead may erase a gain.

Initial target is only layer0 MLP gate N17408/K5120: 44,564,480 payload + 5,570,560 scale = 50,135,040 bytes; zero padding. `prepare` retains native and packed handles for qualification, adding 50,135,040 packed bytes plus temporary/backend overhead. After materialization, the source handles must still hash unchanged. Timing excludes packing, but separately report cold packing host/eval time and allocator/RSS deltas. Full-model packing while retaining native prefill weights would duplicate targeted quantized weights; it is not currently authorized by this prototype and needs a separate memory/lifecycle plan.

Evidence: existing C1 profile steady model graph 2.455167ms vs `async_successor` API 33.644542ms; neither is a GPU kernel breakdown. V25's small measured gain makes another host-graph-only change a weak next candidate. V22's isolated L0 native QMM median was 231.091267578us, R2 230.5998125us, below its 5% gate; its Battery run is not an AC model budget. There is no direct current per-QMV GPU profile, and ≥5% whole-decode improvement is UNKNOWN. The hypothesis targets the dominant weight stream and deserves a single strict local screen before broader work; do not infer model gain from a local 5% result.

Lithos actual source: `formats/blm.py:8–30,195–206` reorders rows/lanes and scales; `compiler/emit.py:434–449,644–652,729–734` selects common NVFP4 rows for accelerator ON/T1/K≥4096/K%1024/R8or16. `kernels/common/gemv_nvfp4_rows.metal:55–97` reads physical payload/scales and preserves x across rows. This is eligible ordinary scalar/SIMD code on Apple9; `kernels.py:51–59,299,343–344` selects Apple9 input layout. M4 Max config says unmeasured/native fallback; the benchmark's exact loaded recipe/epilogue remains UNKNOWN. Our layout is hand-written and intentionally keeps native MLX arithmetic, rather than copying Lithos's different accumulation/casts. M5 neural/tensor instructions and Apple10-only scheduling are excluded.

CPU-only command:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/minivllm-engine-optimization-20261014/v26-nvfp4-rowblock-layout-build1/cpu_contract.py
```

Owner API, after normal runtime/source/resource fences and importing MLX only in the owner:

```python
sys.path.insert(0, D)
from rowblock import prepare
p = prepare(mx, weight, scales, runtime_version="0.32.3",
            device_name="Apple M4 Max", architecture="applegpu_g16s")
mx.eval(p.weight, p.scales)  # explicit preparation outside timing
y = p.project(x)            # lazy output [1,17408] BF16
```

Use V22's frozen single-range L0 weight/scales loader and eight synthetic plus one historical real x cases. First verify native QMM vs unchanged custom R4, then packed vs native QMM and custom R4, shape/dtype/finite/strict raw bytes, source inputs readonly, raw inversely unpacked bytes exact; STOP at first failure. No tolerance or math-policy override. Metal compilation and numerical exactness are still UNKNOWN. Reuse stationary controls and AC/resource/owned-PID protocol; prime both sides, preserve AB/BA order and all raw timings. Initial screen should require a materially larger local win than 5% before claiming plausibility for a 5% model effect, then qualify other MLP shapes, full logits/all128 states, lifecycle and service boundaries in new directories. Native prefill/B2/B4/spec/capture paths stay separate until explicitly implemented and qualified.

Root executable runner (`micro.py run`) now implements that narrow 27-comparison gate, readonly checks before/after qualification and timing, actual inverse packing bytes, one preparation, first-call timing, the unchanged V22 prime128/repeats512 and ten-pair control/AB schedule. Its predeclared local gate remains 5%/5-of-6, which only selects further work; it never qualifies a model or claims ≥5% decode. Actual checkpoint headers show 9,625,927,680 MLP bytes (63.63% of quantized linear/embedding bytes). Even if time followed those bytes exactly, all-MLP 8.53% or gate/up-only 13.36% local speedup would be needed for 5% whole-time reduction; actual time fractions remain UNKNOWN.

```sh
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v26-nvfp4-rowblock-layout-build1/micro.py run --out /private/tmp/minivllm-engine-optimization-20261014/v26-rowblock-micro-ac-attempt1
```

Root must use a new output directory, sole GPU ownership and external owner resource/deadline/reap tracking. The runner supervises one exact direct child with 170s timeout and frozen C1 birth-identity TERM/KILL/wait/strict-absence cleanup. Source/runtime/header mode (`micro.py cpu --out NEW`) completed CPU-only; source-manifest binds all imported old helpers and native kernels. No full checkpoint shard or model is loaded. Resource raw is saved before derived MLX allocator observations and original guards; allocator cache is explicitly 2GiB. No author GPU execution occurred.
