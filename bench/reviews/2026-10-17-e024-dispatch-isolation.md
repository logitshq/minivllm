# E024: isolate attention length from compilation

2026-10-17, source-only design. No new device execution, tolerance change or
performance claim. The actual first E024 point failed the strict BF16 logits
comparison at B1/used257. That comparison changed **two variables**: compilation
and the attention tensor length. It does not by itself identify compilation as
the cause. The original candidate, plan, failure and hard gate remain unchanged.

## Pinned local behavior

Installed runtime metadata is MLX **0.32.3**. The original E024 owner pins the
actual runtime binaries separately; reading matching tagged source is evidence
about the implementation, not a claim that this note rebuilt those binaries.

- Frozen model attention:
  `.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine/model/attention.py:79–90`,
  SHA256 `112997ec74e55ecdb88641e86f7bb29ad8c6c6a73481f1afe463a4332156d52c`.
  Decode takes `L=ctx.max_len`, slices K/V to L, makes a boolean `arange(L)<lens`
  mask, and calls `mx.fast.scaled_dot_product_attention` with a one-row query.
- `bench/m5_ultra_256gb/long_decode_graph/candidate.py:127–136`, SHA256
  `eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84`:
  its closure instead uses `max_len=1536` and is compiled with `inputs=model`.
- `long_decode_graph/qualify.py:234–251,279–292` constructs physical KV capacity
  1536 for both arms, but the reference's logical attention slice is 257 at the
  first point. Independent loaded weights and prefill outputs/states are checked
  before the decode comparison. The later all-cache comparison was not reached
  after the first failing logits comparison; do not infer its result.

## Concrete dispatch hypothesis

The official tagged [Metal SDPA source, lines 893–957](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/scaled_dot_product_attention.cpp#L893-L957)
routes query length <=8 through the vector path. Within it, an architecture
suffix `d` or `s` and K length >=1024 selects `sdpa_vector_2pass`; otherwise the
single-pass vector function is selected, unless the separate GQA >=4096 rule
applies. Thus on the registered M5 Ultra `g17d`, one-query K257 and K1536 select
different reductions despite identical visible tokens and a padding mask.
Lines 529–568 also make the two-pass block count architecture dependent.

This is a **testable mechanism**, not the established cause of the observed
logits difference. The next run must retain the actual device architecture,
runtime/source pins, Q/K/V shapes and mask length; no inferred generic “M5”
identity is sufficient. Do not change SDPA environment knobs or force a kernel
while measuring this first separation.

## Minimum next numerical diagnostic

Register only the original failing B1/used257 state first. Use the same fixed
Qwen3.8-27B NVFP4 target, BF16 activations, original D fixture's first 256 IDs,
positions=[256], lens=[257], physical KV capacity1536 and greedy input token.
Keep gates/norm/packed fusion/core decode compilation OFF as in E024. Verify
the independent original prefill outputs/states and model parameter bytes.

| Arm | Attention bound | Forward execution |
| --- | ---: | --- |
| A | 257 | Original eager model |
| B | 1536 | Original eager model |
| C | 257 | Explicit-state forward closure compiled with `inputs=model` |
| D | 1536 | Original E024 compiled candidate |

Each arm receives its own cache objects/array descriptors made independently
from the same already-evaluated, verified canonical prefill state. No arm uses
another arm's output state. Retain and compare every source byte before/after;
parameter arrays stay unchanged. Reuse the original forward/state construction
for C, changing only its declared bound in a new diagnostic file. Do not edit
the frozen candidate or mutate its private `_bound` to mislabel old evidence.
Record actual compiled-call witnesses for C/D; fallback is a failure.

Save all four original-dtype logits and all 128 K/V/conv/S arrays with shapes,
dtypes, full unsigned-bit hashes and finite checks, plus context metadata and
greedy IDs. The primary contrasts are A–B (bound), A–C (compile at257), B–D
(compile at1536); retain A–D to reproduce the rejected candidate. Report the
first and total differing words for each contrast and preserve failing tensors.
For this newly registered diagnostic, finish all four finite arms rather than
masking a later contrast after an expected numerical mismatch; a resource,
execution, source or input-mutation failure still stops immediately and leaves
the matrix incomplete. This does not revise E024's prior fail-fast gate.

If A–B fails and B–D passes, bound/dispatch is independently sufficient to
explain this point without compilation. If A–C or B–D fails, compilation adds
a discrepancy at a fixed bound; that still does not isolate fusion from other
graph transformations. A follow-up may inspect the first attention output or
use 1023/1024 as a separate registered boundary test. No such follow-up, full
B1..8 matrix, parameter replacement, ahead lifecycle or real D512/HTTP
qualification is claimed by this four-arm diagnostic. No benchmark timings.

## What “no fusion” actually exposes

Installed `.venv/lib/python3.13/site-packages/mlx/include/mlx/compile.h:10,43`
declares public C++ `CompileMode::{disabled,no_simplify,no_fuse,enabled}` and
`set_compile_mode`; header SHA256
`d6405672fe2470b9c304a9816644b94b1bd79981f430799dbd2ac2dc2cdbf739`.
The tagged [compiler, lines 1080–1108](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/compile.cpp#L1080-L1108)
shows `no_fuse` keeps tracing/tape caching and simplification but skips kernel
fusion. It does **not** promise a completely untouched arithmetic graph.
Cached entries do not include mode in their matching key (lines326–360), so a
later mode change must not be assumed to rebuild an already compiled function.

There is no corresponding public Python `no_fuse` argument/export in the
installed `.venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:4700–4741`
(SHA256 `bfa3c09f0f1e3edb925bf92e021f3121f0b953959f43ffd91cfa13895a41e8ce`)
or the tagged [Python bindings, lines1359–1422](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/transforms.cpp#L1359-L1422).
Python exposes compile plus global enable/disable. `MLX_DISABLE_COMPILE`
disables compilation altogether; it is not graph replay without fusion.
Do not guess an API/environment variable or use ctypes to reach C++ symbols.
The four arms above need no such extension; a future public-C++ bridge would
be a separate runtime/source change and separate experiment.
