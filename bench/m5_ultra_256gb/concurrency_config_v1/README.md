# E045: choose a configuration per concurrency

This diagnostic tests Qwen3.8-27B NVFP4 with eight 512-output-token fixtures.
C1 uses the E033 combined entry with GDN+route graph; C8 uses the same entry
with graph only. C is concurrent HTTP requests, not an asserted GPU batch size.
C8's historical same-entry baseline is 170.460644 tok/s; the formal E024
170.619844 tok/s result used a different graph entry.

Only `MLX_MAX_MB_PER_BUFFER` changes: 16, 50, 200, 800. The other explicit
settings remain OPS=50, BFS=20, FAST_SYNCH=0. The variable name refers to MLX's
internal commit threshold, not measured MiB of DRAM traffic. MLX v0.32.3 adds
array `data_size()` values to the internal buffer counter and commits when
either its operation threshold or shifted buffer counter is exceeded. Increasing
this threshold may do nothing if the operation limit dominates.

Source: [MLX Metal device.cpp](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/device.cpp).

Each trial is a fresh server with 8 warm and 8 screening requests. C1 order is
50, 200, 16, 800, 50; C8 order is 50, 800, 16, 200, 50. Both groups retain raw
SSE and full output IDs. The second group is explicitly `admission_probe`, not
an eligible formal result. C8 measures one wave; it does not cover refill.

The independent audit compares complete returned token sequences against both
50 bookends within each concurrency. A candidate is recommended for confirmation
only when both bookend throughput and p50 drift are at most 2%, and both its
throughput and p50 improve by at least 3% against both anchors. Eight-request
p95 is descriptive only. No confidence interval or production policy is claimed.
Changed MLX environments do not inherit the old full-state qualification.
A candidate needs fresh full-logit/cache-state checks and longer paired runs
before entering a formal engine Pareto curve.

The owner uses the existing authorized pause lease for oMLX PID 52818 and checks
all three health endpoints before/after. One model server runs at a time. The
screen has a 1,800-second deadline, driver 1,950 seconds, process ceiling 64 GiB,
system ceiling 240 GiB, normal pressure and zero swap/Swapouts growth. Physical
and metadata probes each have a 5-second budget; resource waits request 3 seconds,
but the enforced maximum observation gap is 12 seconds. Sampling is not a hard
allocation cap. Cancellation settles in-flight probes before server cleanup.
The owner preserves failed receipts and never interrupts the lease's normal
restoration merely because its runner has naturally exited.

Prepare and commit the frozen plan, then run with its printed SHA:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/concurrency_config_v1/execute.py --prepare
.venv/bin/python -I -B bench/m5_ultra_256gb/concurrency_config_v1/execute.py --run --plan-sha256 PLAN_SHA
```

Default invocation only describes the experiment. Output paths are fresh and
exclusive. Never reuse a failed output directory or edit a frozen plan to turn
a failure into a pass.
