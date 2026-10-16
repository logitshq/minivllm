# E008: allocator reuse, not prefix caching

`run.py` starts sequential native servers using frozen retirement source. It
compares only a 2 GiB versus 16 GiB MLX allocator cache limit, at C1/C64, with
three alternating fresh-process pairs and three warm plus three measured
groups per process. The decision rule is written before measurements.

`native_child.py` is the previous loopback shim with active/cache memory
counters added identically to both arms. The cache limit is not preallocated
memory or a hard cap on live tensors. Both arms use the same token arithmetic,
normal EOS, greedy decoding, no prefix cache, and no draft model.

Example from the M5 repository root (requires the archived local frozen source
and workload; create a new output directory name for a new attempt):

```sh
env MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 \
  MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
  .venv/bin/python -B bench/m5_ultra_256gb/allocator_http/run.py \
  --out .artifacts/m5-ultra-256gb/allocator-http-v1 \
  --source .artifacts/m5-ultra-256gb/retire-batch-source-v1 \
  --workload .artifacts/m5-ultra-256gb/fullqual-build2/workload.json
```

`summarize.py --root RAW_DIRECTORY --out RESULT_JSON` independently recomputes
all request/group timing metrics, counts, medians and the acceptance decision.
It checks source identity, resource samples and reaped-process receipts. It
does not claim every raw ID sequence is equal across different HTTP arrival
schedules, or that a small sample establishes statistical significance.
