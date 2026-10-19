# C1 successor submission: source mechanisms and the next probe

Source-only review after E035, while the independent offline GLM copy is owned
by root. No model/backend import, GPU/service action, runtime edit or diagnostic
execution occurred here. Matching-tag upstream source describes mechanisms; it
does not prove the installed binary took a particular branch or rebuild its
provenance. Installed headers were read first; implementation `.cpp` files were
not available in the local wheel, so official v0.32.3 sources were consulted.

## Observation to explain

The retained [ordinal readout](../m5_ultra_256gb/c1_profile_cycle_v1/analysis-v1/readout-v1/async-calls.json)
(SHA256 `f9505c9d714cb2e505798292d12a8a4c871dbe5118926ece7b26e15580f4a2ce`)
records 4,072 full-mode steady ahead hits. Their successor `async_eval` calls total
75.821625094 s, mean 18.620241919 ms. The first, already-submitted root call totals
only 18.523077 ms across the same hits. These are host-call wall intervals,
**not exclusive GPU time or CPU utilization**. The small repeated-root cost is
consistent with the core's already-scheduled-output early return; it is not the
18.62 ms operation that `ahead_submit_once` can omit.

The actual [full specification](../m5_ultra_256gb/c1_profile_cycle_v1/prereg-v1/full-spec.json)
SHA256 `e095ec97cab19051e4aafbda7330829468c7e267b292685db9f1e700b6938ac8`
sets `MLX_MAX_OPS_PER_BUFFER=50`, `MLX_MAX_MB_PER_BUFFER=50`,
`MLX_BFS_MAX_WIDTH=20`, `MLX_METAL_FAST_SYNCH=0`.

## What runs inside an asynchronous call

- The Python binding flattens roots, releases the GIL, then calls C++
  `async_eval` directly. It does not merely queue a Python worker. The caller
  can therefore spend CPU time or block while the GIL is released.
  [Python binding:1112–1119](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/transforms.cpp#L1112-L1119).
- C++ `async_eval` returns immediately when every root is already scheduled.
  Otherwise `eval_impl` records graph degrees, constructs a width-limited tape,
  and visits its primitives. During that visit it finalizes GPU streams and
  waits if active tasks exceed 10, or if active memory exceeds its memory limit
  while tasks exist. It also finalizes streams at the end. An asynchronous API
  does not guarantee a wait-free invocation.
  [Core evaluation:74–365](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/transforms.cpp#L74-L365).
- GPU evaluation invokes each primitive's `eval_gpu` synchronously to allocate
  outputs/encode commands. A threshold-triggered command-buffer commit increments
  active tasks; its completion handler decrements them. The finalization path is
  separate and does not increment that counter, so task count is not a complete
  count of all in-flight GPU work. The scheduler condition variable waits for a
  lower task count when the previous count exceeds one.
  [Metal evaluation:26–70](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/eval.cpp#L26-L70),
  installed `scheduler.h:49–58`.
- Commit thresholds use dispatch count and a size counter. **The latter is not
  a reliable byte/DRAM counter:** `set_input_array` adds `array.data_size()` once
  per newly seen backing-buffer pointer in the current encoder; the array header
  defines that value in elements. Output registration also takes that path.
  `needs_commit` compares `buffer_sizes_ >> 20` with the nominal MB setting.
  [Encoder:323–394,477–527](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/device.cpp#L323-L527),
  [array units:324–335](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/array.h#L324-L335).
  Do not translate setting 50 into an asserted 50 MiB of traffic per buffer.
- Same-stream scheduled dependencies are not host-waited by the core's event
  branch. Cross-GPU-stream waits encode Metal events; CPU-stream waits enqueue
  a CPU task. With fast synchronization disabled, fences take that event path.
  Dependency ordering can delay device completion and therefore pacing, but
  does not establish an explicit host wait per attention/GDN dependency.
  [Events:47–65](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/event.cpp#L47-L65),
  [fences:44–58](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/fence.cpp#L44-L58).

## Three hypotheses, in priority order

1. **Device completion paces submission.** A token has enough work/commits for
   the active-task bound to force the caller to await earlier buffers. This can
   explain a long, stable successor call even if graph construction is quick.
   If true, removing a Python check cannot remove the work. Reduce actual work
   per committed token, such as qualified multirow weight reuse, or reduce
   harmful command fragmentation. Merely moving the wait to the final eval is
   not a speedup. This is a hypothesis, not a demonstrated pacing trace.
2. **Caller-side primitive traversal/Metal encoding is substantial.** The
   compiled callable produces lazy work; its short span does not include all
   later evaluation and encoding. Per-primitive allocation, resource bookkeeping,
   command encoding and graph detachment may dominate even when the outer Python
   graph wrapper is negligible. A specialized projection or larger exact graph
   primitive could remove many such operations; cached Python guards cannot.
3. **Allocator/driver pressure or dependency scheduling contributes.** Active
   memory can trigger a different wait loop. The Metal allocator can reclaim
   cached buffers or allocate new buffers under locks; commit can also enter
   driver/residency work. Check actual memory/task/stack evidence before choosing
   this explanation. Raising memory limits is not the proposed remedy.
   [Allocator:101–164](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/allocator.cpp#L101-L164).

## Small controlled diagnostic, proposed only

First use a new source-bound observer around the *existing* successor submission
and final eval: record both monotonic wall time and the caller thread's CPU time
(`time.thread_time_ns`), with an observer-OFF control. Do not add per-token GPU
synchronization or read tensor values. Warm separately; separate attention-route
boundaries, EOS and first-call compilation. Record original output counts, IDs,
full end-state bytes and the normal final drain; compare complete request time,
not just the location of waiting. This separates caller CPU consumption from
non-running time, not GPU occupancy or which wait primitive caused it.

Then, only if useful, make a tiny fresh-process 2×2 scheduling probe: ops limits
{50,200} × nominal size limits {50,200}; retain BFS20/fast-sync0, source/model/
request history, arithmetic, capacity, physical/swap gates and recovery. Each
setting is fixed **before importing/initializing MLX**: installed `utils.h` caches
these environment values in function statics. Qualify original full logits and
all cache states before timing the changed scheduling regime; the old receipt
does not automatically qualify it. Use the same short original request sequence
and alternating repeats, including the complete final completion/drain.

If wall time greatly exceeds thread CPU and larger buffers only move delay to
final eval, prioritize actual kernel/weight-traffic cost. If caller CPU remains
large, obtain a short separate native stack/Metal command trace to distinguish
graph traversal, primitive encoding, allocator locks and driver work. If both
request time and command count improve with a threshold change, it is a candidate
for formal qualification. Keep global memory limits, fast-sync, BFS width and
model kernels unchanged in this probe; do not bundle several explanations.

## Installed small-source pins

All paths below are under `.venv/lib/python3.13/site-packages/mlx/include/mlx/`.
They were hashed as small source files; no runtime/model binary was loaded.

| Header | SHA256 |
| --- | --- |
| `scheduler.h` | `9b44de7e593a8612b63522c5fc17220b03f3550f2ad3d6efe9425448c207ee1a` |
| `utils.h` | `f09f97894bf9479410082e59850289a03e852e0215a9ddefd5c4855e2d99458f` |
| `array.h` | `543d9096f3ea52870087059db5f793925dfb5e00097638776bc52b01e6189c3b` |
| `backend/metal/device.h` | `ab1e07495b916a86eb762cddc68c6f1ae5e4c61020baacdea3c0dadd59150603` |
