# E041 — actual CPU-only file-like IO byte smoke

Result: **PASS_BYTES_ONLY, 3/3 paths, zero failed attempts**. The default command
returned description only (exit 0); the single explicit actual run returned exit 0.
Actual MLX 0.32.3 / Python 3.13.16 on Darwin used `Device(cpu, 0)`; the script sets
CPU before any array/load/eval and passes CPU explicitly to each load. No model,
GPU workload, service action, memory/cache limit, or global cache control was used.

The manually encoded input is **276 bytes** with 84 payload bytes: U32[9],
F16[11], BF16[13]. Float fixtures include positive and negative zero, finite
normal/subnormal values and maximum finite magnitudes. The raw tensor offsets
(192, 214, 240) and small final tensor exercise reads not aligned to pages.

| Actual input path | Raw dtype/shape/payload | FD closed before later eval/save | IO after close |
| --- | --- | --- | --- |
| Filename | PASS | Reader belongs to MLX | Not instrumented |
| Cached binary file-like | PASS | Yes | 0 events |
| F_NOCACHE binary file-like | PASS | Yes | 0 events |

The uncached path used the **same descriptor passed into MLX**, with Darwin
`fcntl(fd, 48, 1)` returning 0 before its first read. Both file-like loads returned
before their descriptors were closed. CPU eval and a safetensors save then
succeeded, and all original tensor bytes were identical, including signed zeros.
No arithmetic or dtype cast was used for the comparison. An independent stdlib
parse rechecked the archived input against all three actual outputs.

This qualifies this tiny CPU API/byte path and its descriptor lifetime only.
It does not prove large-shard behavior, bounded total memory, less swap, full-model
loader success, performance, or Pareto improvement. File-like MLX safetensors
loads evaluate before returning; ordinary filename loading is lazy, so a future
cached-file-like versus uncached-file-like comparison is needed to isolate the
cache-policy hypothesis. The previous source investigation is copied under
`sources/` with official MLX v0.32.3 links.

## Reproduction and evidence

Source: `bench/m5_ultra_256gb/glm_filelike_io_smoke_v1/smoke.py`. Default is OFF.
`execution/execution.json` retains both exact commands, true exits and elapsed
wall times. `actual/` contains the actual runtime, invocation, per-mode IO events,
original fixture and tiny output files. `sources/` retains the executed source,
runner and archive script. All attempt outputs are retained; no failed attempt
occurred. `manifest.json` and `verification.json` verify each copied member
against its original and re-read archive member, without model payloads.
