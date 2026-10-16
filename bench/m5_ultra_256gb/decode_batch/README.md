# M5 Ultra batching and Pareto frontier

The opt-in `compile_decode=True, fuse_decode=True` path now covers batches 1–8,
at used lengths up to 256, on the pinned M5 Ultra / MLX 0.32.3 / BF16 NVFP4 model.
The single-row QMV and GDN pipelines are preserved. Batched GDN preparation and
state kernels add independent batch offsets without changing arithmetic.
Existing packed projections use MLX's original small-batch matrix multiplication.
Larger batches, longer contexts, and unsupported metadata retain eager execution.
The original parameters stay intact; derived weights remain about 8.714 GiB.

Exploration of larger fused batches is retained under
`.artifacts/m5-ultra-256gb/pareto-batch-*`. Packing mixer projections changes
MLX's reduction route at batch 12 and above. Preserving those projections fixed
the mismatch, but the large-batch candidate regressed at B32. It was not adopted.
Only the verified small-batch extension is included here.

`qualify.py` compares full logits and all cache arrays through unsigned integer
views, checks parameter hashes, replacement, mixed histories, length limits,
metadata fallback, and staggered insertion/removal. `qualify_kernel.py` checks
every batch 1–8 with synthetic scales and strided inputs. `free_generation.py`
compares raw token IDs with eager execution at the same batch and schedule.

`frontier.py` runs fresh sequential servers, with three warm and three measured
groups per process. Use `--repeats 3 --concurrency 1 2 4 8 16 32 64 --lithos`.
`--source` must point to the frozen source tree; `--out` must be a new directory.
The experiment pins source/runtime assets and records launches, SSE events,
resources and strict server cleanup. The new loopback shim has a listen backlog
of 128; the previous default of 5 reset connections in the first C8 screen.
That failed attempt is retained and is excluded from the final comparison.

All engines use the same 27B checkpoint, greedy sampling, normal EOS, 19 input
tokens, a 32-token output budget, and no prompt cache or drafter. Both axes are
client-observed: x is the per-user post-first-text delivery-rate estimate; y is
total actual output tokens divided by the complete concurrent group window.
These are not raw GPU decode rates. C is HTTP concurrency, not an assumed GPU
batch. Lithos's official endpoint supports C1 only and is shown as a measured
point, with its local M5 Ultra profile and accelerator enabled.

Keep all measured points, including configurations dominated by another point
from the same engine. An engine's observed Pareto frontier is the nondominated
subset. A comparison point is covered only when an actual measured minivllm
point has both x and y at least as large, with one strict improvement. Lines
between points are visual guides, not measurements. This workload does not
establish a universal result for other prompt lengths, workloads or speculation.
