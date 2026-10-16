# V32 fixed-shape compiled QMV factor-fold screen

A and B both use `mx.compile(fn, shapeless=False)` over the same immutable L0 gate weight/scales; x is explicit. B calls only the frozen V31 helper. No old source/kernel changes, copied/reordered weights, model or engine integration. This is a new compiled-boundary experiment; V31 uncompiled CONTROL_DRIFT_STOP remains unchanged.

Only50,135,040B weight/scales ranges load. Bind V31 actual27 strict qualification and independently compare9 inputs for compiledA-native, compiledB-native and compiledB-A (27 new strict checks), plus all captured inputs readonly before/after qualification/timing. First byte/finite/shape/dtype/readonly failure stops before timing. Factory, cold function-call host and subsequent eval host costs are separately recorded outside timing. Trace-body counters confirm same-shape warm measurement does not re-enter the heavy Python QMM/helper construction body; counters do not execute on compiled hits.

Prime2048/repeat2048; unchanged4 AA/BB controls +6 balanced AB/BA pairs, controls each[.95,1.05]/median[.98,1.02], local5%/5-of-6 gate. Every output eval, hashes after both blocks. Same AC/thermal0or1/pressure1/raw-first/original guard/2GiB allocator,170s child deadline and exact birth/PID cleanup. Root owns external resource monitor/deadline/release. Numeric/performance UNKNOWN until actual execution; no model or adoption inference.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/minivllm-engine-optimization-20261014/v32-compiled-qmv-factorfold-build1/micro.py cpu --out /private/tmp/minivllm-engine-optimization-20261014/v32-compiled-qmv-factorfold-build1/cpu-source-check1
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v32-compiled-qmv-factorfold-build1/micro.py run --out /private/tmp/minivllm-engine-optimization-20261014/v32-compiled-factorfold-ac-attempt1
```

Fresh output directory and sole GPU owner required. Frozen V31 scalar overflow/half-subnormal caveats still apply; no universal all-finite-activation equivalence claim. Author has not imported MLX or executed GPU. Ready source is immutable.
