# E026: GDN convolution-tail ownership and isolated candidate

2026-10-17. **The isolated tiny device diagnostic passed: 48 exact-bit cases,
with retained backing reduced to 64 KiB in both ON arms. The real-weight layer-v2
diagnostic also passed 24 exact-output/state cases; full-model correctness
and throughput/latency remain unqualified.** See the
[archived results, independent audit, and memory figure](../../../docs/experiments/m5-ultra-256gb/E026-conv-tail-tiny-v1/README.md).
The helper remains default OFF and does not modify production/frozen code.
The source analysis below predates the device run; it is not an E023 cause claim.

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

## Original device-validation plan and remaining controls

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


## First tiny device run: separately frozen numerical-only contract

`prereg-tiny-v1/` freezes three fresh-process arms, OFF / ON-lazy / ON-materialize, each with a 30-second external deadline, under a 180-second owner. The first screen covers T1/19/935/978, three exact-word fixtures spanning all 65536 uint16 values, and a finite fixture: 16 cases per arm. It deliberately covers only contiguous synthetic parents. The strided/already-compact controls and model integration above remain future work.

This diagnostic allocates no weights: its largest named parent is below 20 MiB, with observed active/peak required below 64 MiB at every recorded boundary. That is an observation guard, not an OS memory cap; MLX `set_memory_limit` is only a guideline and is not claimed as enforcement. A separate policy admits an already present system pressure level 2 only for this bounded bit/backing test: actual free physical pages must be at least 50 GiB, pressure may not worsen (including after recovery), Swapouts must not increase, swap-used growth must stay within 16 MiB, and the original AC/thermal/process checks apply. Sampling has interval blind spots. **E023, full-model and all performance admission gates remain pressure 1; this run produces no latency or throughput evidence.**

The existing, authorized SIGSTOP/SIGCONT lease pauses and restores original PID74384. It does not unload/restart the service or depend on the pending cold-restart approval. Direct child ownership, strict reap/absence, partial-result rejection and failure-stop behavior are CPU checked. Root reran 7 helper, 9 screen and 11 owner checks. The 561 source/runtime pins include MLX and NumPy but do not purport to package all system frameworks/stdlib.

Frozen plan SHA: `037999b2ede8daf5e5d244c41871a36dd9c77e864f2959cfdd3f3540b1cac93e`. This paragraph records preparation, before any device result.

## Completed tiny device run

The frozen three-arm contract above completed with 48/48 exact-bit cases and
528 recorded phases passing. An independent standard-library auditor and root
rerun agree. For T978, the retained output backing changes from 20,103,168 to
65,536 bytes; each case ends with zero observed active array memory. These
figures describe isolated synthetic ownership, not full-model peak memory.
Finite-validation temporaries also release bytes in OFF, so use the audited
retained-backing difference rather than the full active-byte delta.

All children and the owner exited and were reaped; the original service was
resumed with its original PID/start identity. No cold restart occurred.
The preregistration and CPU receipts remain historical preparation evidence;
[the result archive](../../../docs/experiments/m5-ultra-256gb/E026-conv-tail-tiny-v1/README.md)
records the separate execution and audit. Correctness is an admission gate:
only a subsequently qualified full-model HTTP run may add a point to the
throughput-versus-complete-request-latency Pareto plot.

## Scoped integration and qualification scope

`integration.py` temporarily binds only selected GDN instances on ordinary
eager single-sequence prefill. It preserves the original `_prefill` code object
and changes only its private convolution wrapper, leaving convolution output
identity and all arithmetic intact. Copy and immediate materialization remain
separate switches, both default OFF. Decode, grouped prefill, record/capture/spec
routes are excluded; every temporary instance attribute is restored on exit.
This is a diagnostic hook, not a production timing path.

Root reran 12 CPU tests; independent review found no blocking issue and checked
the installed MLX Module attribute behavior without importing MLX. See
[integration-cpu-review.json](integration-cpu-review.json). An outer `mx.compile`
is explicitly unsupported. Future observers must store scalar receipts and
serialized errors, not tensor references or exception tracebacks.

The actual convolution output graph also references the old parent. Immediate
tail evaluation alone does not prove the parent is released at the hook. Record
hook, normal-product evaluation, common validation, and observer-release phases
separately; validation-induced evaluation is not a measured production benefit.
The [single-layer feasibility review](../../reviews/e026-single-layer-feasibility.md)
prepares a small real-weight diagnostic without loading the whole model. It
requires a separate resource contract; the tiny run grants no larger admission.

## Real-weight single-layer v2 result

After separately preserving the v1 footprint-observation failure, the reviewed
v2 monitor retained the same budgets and all original model arithmetic. Three
fresh arms completed 24 cases and 204 phases. Both ON configurations exactly
match OFF for all complete outputs/states, inputs and parameters. T978 retained
state is 23,248,896 B OFF versus 3,211,264 B ON after observation aliases are
dropped; active and unique buffers each save 20,037,632 B. Every process exits,
source pins hold, and the original service is restored.

[Full original arrays, independent audit and tutorial result](../../../docs/experiments/m5-ultra-256gb/E026-conv-tail-layer-v2/README.md).
This does not qualify full-model execution, ordinary prompt-prefix continuing
state, the production evaluation placement, or any throughput/latency point.

### Staging at the existing runner eval (CPU prototype)

`staged_integration.py` adds an independent default-OFF scope around the exact
ordinary eager runner instance. It replaces only that instance's private view
of its original single `mx.eval`, stages eligible lazy tail copies immediately
before it, and includes them in the same eval. It does not wrap GDN/conv or add
per-layer evals. Grouped prefill, speculative/capture and decode-only paths are
excluded; unrelated staging requests are not scanned. Context exit restores
the exact prior runner attribute, including exceptions.

Thirteen CPU tests execute the original runner AST with fake arrays and check
routing, the eval argument boundary, source guards, exception restoration and
reference release. An initial weakref test was retained as a fixture failure:
`unittest.mock` held the runner in call history; clearing that history makes the
unchanged weakref assertion pass. This is not a GPU result. Original eval args
can retain parents until return; neither peak reduction nor whole-model output
identity or throughput/latency gain has been demonstrated for this prototype.
