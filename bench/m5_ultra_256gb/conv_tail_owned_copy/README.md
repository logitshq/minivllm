# E026: GDN convolution-tail ownership and isolated candidate

2026-10-17. **CPU/source review only. No GPU execution, allocation measurement,
numerical qualification, integration, performance result, or E023 cause claim.**
The helper is default OFF and does not modify or patch production/frozen code.

## Confirmed source path

Frozen E020 core `model/gdn.py:26–34` constructs a contiguous
`xp = concatenate([previous_3_rows, qkv])`, returning `xp[T:]` as the convolution
state. For the fixed Qwen model this is BF16 `[3,10240]`, backed by the
`[T+3,10240]` concatenation.

The exact v0.32.3 sources and SHA-256 values are copied under `sources/` and
listed in [source-pins.json](source-pins.json); Apple notices and
[license](LICENSE.MLX) are retained. The source chain is:

1. [GPU Slice](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/gpu/primitives.cpp#L208-L219)
   calls [slice_gpu](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/gpu/slicing.cpp#L9-L16).
2. [Common slicing](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/common/slicing.cpp#L20-L35)
   computes the view strides and offset and calls `copy_shared_buffer`.
3. [array::copy_shared_buffer](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/array.cpp#L193-L205)
   shares the original `Data` pointer. The shared object's destructor owns the
   buffer deleter; changing the slice's logical shape does not truncate it.
   [buffer_size](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/array.h#L369-L375)
   queries the underlying allocator buffer, not logical array bytes.

This confirms a concrete backing-retention path, including after the original
Python `xp` variable dies. It does not establish the size or duration of any
particular allocation in the aborted E023 process.

## Why this small helper can use contiguous

In [Contiguous::eval_gpu](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/gpu/primitives.cpp#L50-L61),
sharing is allowed only when the existing buffer is at most logical output
bytes +16,384 and has the required layout. Otherwise it invokes a General copy.
[set_copy_output_data](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/common/copy.h#L25-L47)
allocates `out.nbytes()` for General copy; buffer donation belongs to its separate
Vector branch. The [Metal copy implementation](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/copy.cpp#L13-L23)
then dispatches the copy. Thus a contiguous *view* can still be compacted in this
version. An already small backing can correctly remain shared.

The candidate is only:

```python
owned = mx.contiguous(tail.view(mx.uint16)).view(mx.bfloat16)
```

Equal-size dtype views preserve storage and avoid numeric conversion. The copy
operates on uint16 words, preserving BF16 bits. No NAX/custom Metal, dtype
relaxation, convolution change, or GDN recurrence change is required.

`own_conv_tail(mx, tail, enabled=False, materialize=False)` accepts only the fixed
BF16 `[3,10240]` tail on pinned MLX 0.32.3 / M5 Ultra / `applegpu_g17d`.
OFF and unsupported-domain calls return the same array object. Flags must be
actual booleans; materialization without enabling is rejected.

**The two switches are distinct interventions.** `enabled=True` constructs a
lazy graph; `materialize=True` additionally calls `mx.eval(owned)` synchronously
at this helper call. The latter is a diagnostic placement, not an accepted
production policy. Current target-only runner evaluates logits/sample outputs;
the stored tail copy is not their dependency. Without separately evaluating it,
the copy can wait until a later decode or observer comparison. Other snapshots,
graphs and aliases may still retain the old backing after the copy completes.

## Nominal bytes and upper bounds

Per layer, logical tail bytes are `3 × 10240 × 2 = 61,440` (60 KiB). The original
concatenation has `(T+3) × 20,480` nominal bytes, so its excess is `T × 20,480`.
Even T1 exceeds the 16 KiB contiguous sharing allowance. Across 48 GDN layers,
T1024 corresponds to 960 MiB of excess nominal backing per request.

The recorded E023 refill steps had these two prefill lengths:

| Step | Lengths | One arm, nominal excess across 48 layers | Both arms |
| --- | --- | ---: | ---: |
| 513 | 978 + 886 | 1747.500 MiB | 3495.000 MiB |
| 514 | 926 + 921 | 1731.5625 MiB | 3463.125 MiB |
| 515 | 954 + 935 | 1770.9375 MiB | 3541.875 MiB |

These are upper bounds on the logical excess reachable through **these tail
references**, under the assumption every listed parent exists simultaneously.
They are not measured peak/RSS reductions: aliases can share a parent, other
users can keep it alive, allocator rounding/cache matters, and lazy evaluation
changes overlap. They cannot explain a system-pressure event on their own.
The independent [cache capacity review](../../reviews/e023-cache-memory-budget.md)
adds committed/staging/retirement inventory, decode-tail backing, and rebuild
temporaries without counting shared observer handles as duplicate buffers.

## E023 observer/release review

The frozen qualifier's `release_pair` restores model/decode/run/scheduler
proxies, clears `calls`, `previous_future`, `postrun`, both ownership lists,
then collects and synchronizes/clears MLX. The load function's local weight
maps and engine variables leave scope. No explicit Python chain retaining C1
models into C8 was found. This is not a runtime weak-reference or allocator proof.

Every D prompt exceeds 512 tokens; the frozen full-model `CompiledDecode`
guard requires `max_len <= 256`. Its graph/packed-weight capture path therefore
does not qualify for this workload. B1 GDN gates compilation uses explicit
array parameters, rather than capturing the model in a Python closure.

Observer snapshots use `item[:]`, so they create handles, not owned copies.
`begin()` clears calls/postrun but retains `previous_future` until `finish()`;
candidate `begin()` occurs only after the reference step. Therefore the old
candidate postrun/call snapshots overlap the reference refill, and carried
ahead evidence may survive until that step's comparisons finish. These are
bounded previous/current generations, not an accumulating history in source.
Comparisons also evaluate whole cache arrays and finite/equality masks, which
adds qualification work and potential transient allocations.

At retirement/rebuild, stored pre-retirement snapshots intentionally retain
old batches while the live core creates survivor/new-row batches. In particular,
prefill call snapshots can retain the large convolution-tail backing through
the next step, even if a newly concatenated decode batch is compact. Removing
such evidence before its required comparisons would weaken the qualification;
this helper does not do that.

The last complete raw rows, 513–515, each report B6 decode plus two prefills,
two additional retirements, and PASS. This establishes workload overlap only;
the resource abort may have occurred in any unrecorded phase of the next step.

## Minimal next device validation — not run

1. Pre-register fresh-process OFF, ON-lazy, and ON-materialize arms. Retain the
   fixed resource guard, direct owner, timeout, and restored-service lease.
   Use synthetic parent/tail fixtures at T1/19/886/935/978/1024; include strided
   and already compact inputs to test copying/legitimate sharing separately.
2. Evaluate each parent first. Check original BF16 shape/dtype and uint16 bits,
   source unchanged, and eligibility/materialization route. Binary copy tests
   may cover all uint16 payloads; actual model gates still require finite tensors.
3. After evaluating `owned`, record `get_array_buffer_size(parent, tail, owned)`
   and each individual tree. The installed public API counts unique *full
   allocator buffers* of evaluated arrays. It excludes graph objects, cached
   buffers, and non-array memory; record active/cache/peak alongside it.
4. Drop all original parents/tails/comparison temporaries, collect/synchronize,
   then measure the retained output alone. Do not retain an oracle alias while
   claiming its parent was released. Keep cache policy identical and distinguish
   active allocation from allocator cache; do not use RSS as a buffer identity.
5. Only after this small test passes, propose an explicit scoped prefill hook
   and a new E023 numerical run. Record buffer unions at before/after-model,
   postrun, retirement, observer reset, and pair release. Weakrefs to both old
   engines/models plus their post-GC liveness can check cross-pair ownership.
   Preserve all full logits/cache/ahead comparisons and original failure gates.

The first integration must keep materialization placement explicit. Copy-only
and per-layer synchronous materialization cannot be treated as the same
performance intervention. No new frontier point follows from this prototype.

## CPU checks

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/conv_tail_owned_copy/test_cpu.py -v
```

Seven tests pass: no-op OFF, exact word-view route, independent materialization,
fallback/flags/backend guards, copied-source SHA/ownership-branch checks, and
import under forbidden MLX/NumPy/Torch/core/socket access. They establish the
Python interface and static source contract only, not device results.
