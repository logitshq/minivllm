# GLM loading: file-like IO and F_NOCACHE

E039 ended **FAIL / RESTORED**. Restoration passed independently; the actual
GLM diagnostic completed zero requests and produced no loader-return or
successful pool-await witness. At physical sample 0018, whole-system swap grew
1,247,805 bytes and Swapouts grew 76, while recorded child/system footprints
remained below 220/240 GiB. The same sample reports 5,985,359 file-backed pages
(about 91.33 GiB) and 1,631,479 compressor-occupied pages (about 24.89 GiB), using
16,384-byte pages. These categories are global, not GLM-exclusive allocations;
this observation does not prove what caused the swap increase. Exact evidence
is retained in [E039](../../docs/experiments/m5-ultra-256gb/E039-stacked-glm-load-v1/README.md),
including `owner/physical-0018/sample.json` inside its metadata archive.

The next cache hypothesis is technically testable without replacing model
arithmetic. It is not yet a demonstrated explanation or a runtime patch.

## Verified interface and ownership

The installed `mlx/core/__init__.pyi:3284` accepts a file object, string or Path.
Its SHA256 is `bfa3c09f0f1e3edb925bf92e021f3121f0b953959f43ffd91cfa13895a41e8ce`.
The pinned MLX version is 0.32.3; the following are official v0.32.3 sources,
not proof that an untested native wheel behaves identically in every detail.

- The Python binding recognizes file-like objects by `readinto`, `seek`, `tell`
  and `closed`. For safetensors, it creates a PyFileReader and evaluates each
  array before returning from `mx.load`. The reader holds Python references and
  reads directly into an MLX-provided memoryview. Consequently the file must
  remain open through that call; source inspection does not require retaining
  it through a later `mx.eval(model.parameters())`. Explicit
  `format='safetensors'` avoids relying on a custom object's name. Importantly,
  this path is eager per shard, so changing from a filename is more than a cache
  flag change. [Python binding, lines 90–189 and 255–286](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.3/python/src/load.cpp).
- The safetensors decoder builds Load primitives from shapes, dtypes and byte
  offsets. On the non-CUDA path it uses the CPU stream. Filename input selects
  ParallelFileReader. [Safetensors implementation, lines 105–207](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.3/mlx/io/safetensors.cpp).
- ParallelFileReader opens with O_RDONLY/O_BINARY; its shown path does not set
  F_NOCACHE or mmap the file. Offset reads use pread; large reads are divided
  into 32 MiB batches with a shared pool. [Reader declaration](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.3/mlx/io/load.h),
  [read implementation, lines 318–399](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.3/mlx/io/load.cpp).
- Load::eval_cpu allocates the output buffer before reading bytes into it.
  The Metal allocator uses shared Metal buffers; this is distinct from keeping
  the model as a mapped file. Ordinary OS file caching and the destination
  allocation can therefore coexist, but the source code does not identify the
  ownership or cause of the 91 GiB global file-backed observation.
  [Load primitive](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.3/mlx/backend/common/load.cpp),
  [Metal allocator](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.3/mlx/backend/metal/allocator.cpp).

The local Darwin SDK `usr/include/sys/fcntl.h:257` defines F_NOCACHE as 48,
per file descriptor. The earlier repacker used this on its own IO only; it did
not change the later official loader's newly opened descriptors. Setting it on
a different descriptor and then passing a filename to mx.load would not implement
the intended policy. No global cache control or cache purge is proposed.

## Official loader seam and minimal next experiment

Installed `mlx_vlm/utils.py:965–967` gathers each shard with `_load_safetensors`;
`:1199–1201` uses `mx.load(path)` and `:1190–1191` evaluates model parameters when
`lazy=False`. Source SHA256:
`2045e18d1a16222d07b157251e4a5108b2b12bf740b885f9682a071c8dc59134`.
A future scoped adapter could alter just this shard-read seam while retaining
the actual official model/sanitize/quantization/fusion/request implementation.
No such adapter was created in this investigation.

First use a tiny standalone fixture with U32 packed weights, F16 scales/biases
and BF16 values. Compare exact original bytes, keys, shape and dtype across:

1. Ordinary filename input, followed by explicit evaluation.
2. A cached binary file object passed to `mx.load(..., format='safetensors')`.
3. The identical file-object path with `fcntl(fd, 48, 1)` checked before its first
   read; failure must reject, never silently fall back.

Keep each descriptor open through its load call and close on success/error.
Exercise nonaligned tensor offsets and a final short chunk. If a bounded custom
readinto wrapper is used, it must fill the whole requested memoryview through
bounded reads: the binding treats a short read as an error. Verify outputs again
after closing the descriptor and confirm the actual wheel's materialization
behavior. This is a byte/API smoke, not a model or performance qualification.

Only after that smoke, register a new cold loader experiment with the same E036
checkpoint and resource/recovery gates. Cached-file-object versus uncached-file-
object is the primary contrast: both have the same eager shard semantics.
Keep the filename baseline separately. Retain loader stages, MLX active/cache/
peak readings, raw global file-backed/compressor/swap counters and settled child
cleanup. F_NOCACHE is not an allocation limit, does not establish zero swap, and
is not evidence that pre-existing cached pages were purged. Full-model loader
success and four-request completion remain separate future gates; no speed or
Pareto claim follows from a loading-memory improvement.
