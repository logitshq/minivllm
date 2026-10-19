# GLM raw expert checkpoint repack — default OFF

This is an offline layout candidate for the exact downloaded
`mlx-community/GLM-5.3-Flash-4bit@fa04cbc3622b16aa7475404060b05e506d81bd4e`.
The original checkpoint is read only. **No real payload repack or device load has
run.** This does not qualify mini GLM inference, quality, memory fit or speed.

The E034 failure motivates removing an avoidable load-time layout operation.
Original `mlx_vlm.load` retains shard arrays while GLM sanitize calls DSV32's
`mx.stack` for every expert projection. Storing those exact stacks on disk means
that backbone branch has no `experts.0` key to trigger. It removes the 42 backbone
expert-layer stack DAGs and their source handles; the actual oMLX MTP wrapper still
stacks the separate nextn layer. It does not establish that file-cache pressure/compression
will disappear or that the model fits without swap.

## Exact transformation and compatibility evidence

Read-only inspection of all 62 safetensors headers matched the previous inventory
hashes, including each **8-byte length prefix plus JSON header**. No payload was
read for that inspection. One initial scratch check incorrectly hashed only the
JSON and rejected; after checking `runtime_probe.py:78`, the prefix-inclusive
comparison passed. These scratch observations are not a full payload hash.

The original checkpoint has 113,446 tensors. Only 108,864 routed-expert tensors
from backbone layers **3–44** change layout: 42 layers × 3 projections × 3 suffixes
× 288 experts become **378** tensors. The output has **4,960** tensors. All
203,976,457,080 payload bytes are conserved (headers account for the difference
from 203,992,076,296 original shard-file bytes).

For every layer, expert order is numeric **0 through 287**, exactly the reviewed
DSV32 list comprehension. For a target tensor, concatenating the original
contiguous tensor bytes in that order implements a new leading expert dimension;
there is no dequantization, conversion, transpose or floating-point operation.

| Projection | Original weight per expert | Original scales and biases per expert | New leading dimension |
| --- | --- | --- | --- |
| gate / up | `U32 [2048,512]`, affine 4-bit | `F16 [2048,64]`, group 64 | 288 |
| down | `U32 [4096,320]`, affine 5-bit | `F16 [4096,32]`, group 64 | 288 |

The exact raw key change is:

```text
model.language_model.layers.L.mlp.experts.E.PROJ.SUFFIX
→ model.language_model.layers.L.mlp.switch_mlp.PROJ.SUFFIX
```

The installed vendor GLM container maps both forms to
`language_model.model.layers.L...`, then passes `model.layers.L...` to language
sanitize. DSV32 sees an already-stacked `switch_mlp` key and leaves it alone.
The loader's `_quantization_path_aliases` strips `language_model.` from module
paths. The unchanged config already specifies
`model.layers.L.mlp.switch_mlp.down_proj = {bits:5, group_size:64}`; gate/up retain
the default 4-bit setting. Shared experts retain their existing 6-bit configuration
and **unchanged raw tensors**. Both quantization dictionaries, config, tokenizer,
vision tensors, router tensors, attention tensors and other sidecars stay byte
identical. MTP layer **45** stays in its original per-expert form. The actual oMLX
`utils/model_loading.py:982–1023` installs the VLM runtime sanitize patch even with
MTP decoding disabled. `mlx_vlm_mtp/glm5_next_vlm_runtime.py:808–890` lifts nextn
layer 45 out before backbone sanitize, passes it through the stock sanitizer as
layer 0, and rebinds the results to `mtp.0.*`. **One MTP expert layer still stacks
at load time** (nine projection/suffix arrays). Its fusion tensors are remapped
and its shared embedding/head duplicates are dropped by that unchanged wrapper.
`model_loading.py:123–161` also supplies the nextn quantization aliases. Keeping
these original bytes preserves that path; this candidate does not remove the MTP
module or change MTP settings. Other original sanitize operations remain the
official loader's responsibility. The earlier draft's claim that the actual
loader discards layer 45 was incorrect and is superseded by this path review.

`candidate.REVIEWED` now covers seven loader/container/language/config/DSV32/MTP
sources. The model-loading SHA comes from the existing v3 source-pins receipt;
the MTP-runtime source was hashed and sealed only after the timed C1 driver
finished and restored its service. Its SHA256 is `d2abbc013d2d5c6e3bb63124e3a0c51adaef64342e730884bbc36ebe4c53bb28`.
This is a source-level compatibility argument, not a completed
EnginePool load. Output shards remain at the checkpoint top level: several oMLX
discovery and size-estimation paths glob there, even though the VLM index loader
itself can read nested shard paths.

## Implementation and two explicit steps

`candidate.py` imports only the standard library. Default invocation describes
and does nothing. The actual header-only preparation completed with exit 0
(tool `d8b38f`) after the timed profile closed. The retained plan is
`bench/m5_ultra_256gb/glm_expert_repack/prereg-v1/plan.json`, SHA256
`323ff8da5d2faf6bb774b1c03f3a52f49198cbb9898f21c91ac25aa546a4c50a`.
Its summary binds 113,446 input tensors, 4,960 output tensors, 378 stacked
tensors, 203,976,457,080 payload bytes, 203,992,076,296 original file bytes,
45 backbone layers and 288 experts. This is **HEADER_PLAN_ONLY**; no actual
payload copy or device load has run. The preparation command shape is:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/glm_expert_repack/candidate.py \
  --prepare --out /absolute/new-directory/plan.json
```

Preparation checks exact index/header coverage, all 288 members of every group,
shape/dtype/byte ranges, mixed quantization, metadata references and reviewed
source hashes. It reads headers and small sidecars, including the ~20 MiB tokenizer
and ~11 MiB index. It does **not** rehash the ~190 GiB payload. The plan binds the
candidate SHA, input metadata/stat identities, historical original shard SHA256s,
and a deterministic complete output-layout SHA. A source edit invalidates it.

Only a separately reviewed, explicit run writes the full new checkpoint:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/glm_expert_repack/candidate.py \
  --run --plan /absolute/new-directory/plan.json --plan-sha256 ACTUAL_SHA256 \
  --out /absolute/fresh-checkpoint-directory
```

The actual CLI run requires Darwin. Every original payload-SHA read, source-part
read, output-shard write and output verification read uses a single
`payload_open` helper. Before its first IO, it calls per-descriptor
`fcntl(fd, F_NOCACHE=48, 1)`; refusal raises, closes that descriptor and never
silently retries with caching. It does not use `F_GLOBAL_NOCACHE`. The plan,
invocation and final receipt retain the policy. Linux is explicitly marked
`not_applicable_linux_cpu_fixture` for portable tiny tests; it cannot run the
actual specialized CLI copy. This reduces page-cache pollution during roughly
570 GiB of reads, but cannot guarantee zero compression, zero swap or a fixed
physical footprint.

The run checks a fresh non-overlapping output and disk space for the complete
payload plus headers/sidecars/manifest allowance and **64 GiB reserve**. Read-only
`statvfs` during feasibility inspection reported 954,368,000,000 bytes available
(~888.82 GiB); the run must measure again.

1. Recheck all original shard full SHA256s sequentially **before output creation**.
2. Stream each output tensor in original expert order; source reads seek to exact
   header-derived ranges. Payload buffers are at most **8 MiB**, with one input
   and one output file open. Metadata structures and sidecars have a separate
   footprint; 8 MiB is not a total RSS or OS file-cache bound.
3. Compute each source component and concatenated output-group hash while copying.
   Reread every completed output shard and every group independently. Preserve
   original dtype and new shape in the safetensors header.
4. Check source identities and reviewed source hashes again, copy unchanged
   sidecars, then publish the new index only after every shard is complete.
   `complete.json` is the final success receipt. Failures retain partial files
   and `failure.json`; callers must require completion before loading the output.

This intentionally spends roughly three payload reads (original SHA, source
copy, output validation) plus one write to make a reusable byte-verified layout.
It is an offline storage operation, not a timed inference result. `repack-manifest.jsonl`
records every output tensor, every source byte range/component hash, and every
concatenated group hash. The receipt binds the plan, candidate, manifest, index
and complete output shard hashes. The original downloaded checkpoint is never
modified or deleted. Original input header metadata is preserved; the new index's
`total_size` records actual payload bytes.

## Tiny CPU check and next gates

The first guarded tiny-fixture run, before the MTP-wrapper/F_NOCACHE revision,
passed **11/11**, no skips or forbidden
operations, 0.032719208 s. Retained evidence:
`.artifacts/m5-ultra-256gb/glm-expert-repack-cpu-v1/`, tool `eb250d`.
The guard prohibited MLX/model imports, mmap, subprocess/network and process
signals. Tests cover exact original SHA-checked DSV32 sanitize AST on fake packed
arrays, 12-expert numeric order, multishard byte readback, unmodified sidecars and
MTP tensors, mixed quantization, malformed headers, missing experts, changed
source bytes, disk reserve, fresh output and failure before index publication.
This is not an actual MLX evaluation. The later actual `--prepare` is recorded
above; no actual `--run` has run.

The revised source adds four tests (15 total). After the C1 driver completed and
restored its service, a fresh guarded run passed **15/15**, no failures, errors,
skips or forbidden operations, with all ten tracked source files unchanged.
Elapsed harness time was 0.067393917 s on Darwin. Evidence is retained in
`.artifacts/m5-ultra-256gb/glm-expert-repack-cpu-v2/` (tool `6b1cab`); v1 remains
untouched. The candidate SHA is
`fd3c30de034a2eba2e3002eb61df08737b07652bd86a9edaeaba75fd35d99d5b`.
These tests extract SHA-pinned original container, language, DSV32,
and nested MTP sanitize AST definitions into a fake packed-array namespace; no
backend modules are imported. The tiny fixture compares original versus repacked
key/shape/dtype/bytes after that path, verifies nine remaining MTP stacks and
fusion names, and checks the actual quant-alias helper for default 4-bit,
backbone/MTP down 5-bit and shared-expert 6-bit recipes. This is a specific raw
expert-layout proof, not a complete MLX numerical or vision-sanitize test.
Additional cases exercise the real local payload helper with offset 3, 73-byte
reads and a short final block; on Darwin that reaches real F_NOCACHE. Mock cases
check call order and fail-closed descriptor cleanup. The real Darwin uncached
fixture and all revised path checks passed; this remains byte/layout validation
on tiny inputs, not an official model load or physical-memory qualification.

One source-edit command had a quoting SyntaxError before any write (tool
`a61a55`); the corrected source-only command succeeded. A separate documentation
edit initially failed at tool JavaScript parsing, also before execution. No test
or payload action was run as part of either failed edit.

Both original CPU attempts and their source snapshots are independently archived
under `docs/experiments/m5-ultra-256gb/glm-expert-repack-cpu-v2/`, with per-member
SHA256 verification. There are no failed CPU test attempts among these two runs;
the source-edit failures described above are separate from test execution.

Next: root reviews/commits the prepared source and actual header plan before the
explicit full offline copy. The
new checkpoint needs a separately registered official loader/control run with
unchanged physical/swap gates and original recovery guarantees. Only that run
can test the memory hypothesis; no Pareto point follows from this utility alone.
