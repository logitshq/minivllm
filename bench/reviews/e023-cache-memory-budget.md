# E023 cache capacity and bounded memory diagnosis

2026-10-17. Static review only: no MLX import, GPU execution, weight-tensor read, or service change. The prefill conv tail is a proven shared-buffer retention path. **The cause of E023's resource abort is not established.** At the reported pressure level 2, do not load another full model this round.

The owner reported C1 completing 4,096 steps and C8 stopping after 515 passing steps; resource sample 164 had pressure 2 and swap growth about 145 MB. Child RSS about 29 GiB is not its physical footprint. The paused external service still had about 142.82 GiB physical footprint. These observations establish resource pressure, not a leak or which allocation caused it. The narrow ledger facts below were supplied by the independent reviewers; this review did not rescan the 1.40 GB ledger.

## Exact state sizes from the pinned model

Configuration snapshot `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`: 64 layers, 48 GDN and 16 full-attention; GDN 48 value heads, key/value dimensions 128, conv width `2*16*128+48*128=10240`, kernel 4; attention 4 KV heads, head dimension 256. GDN S is FP32; conv and K/V are BF16. MiB/GiB below are powers of two.

| Array group | One row, all relevant layers | Bytes |
|---|---:|---:|
| GDN S: `48 layers * [48,128,128] * 4` | 144 MiB | 150,994,944 |
| Logical conv tail: `48 * [3,10240] * 2` | 2.8125 MiB | 2,949,120 |
| K+V: `16 * 2 * [4,L,256] * 2` | `L/16` MiB | `65536*L` |
| Total logical cache, L=1024 | 210.8125 MiB | 221,052,928 |
| Total logical cache, L=1536 | 242.8125 MiB | 254,607,360 |
| Total logical cache, L=2048 | 274.8125 MiB | 288,161,792 |

| Batch | L=1536 logical cache | L=2048 logical ceiling |
|---|---:|---:|
| B4 | 971.25 MiB | 1,099.25 MiB |
| B6 | 1,456.875 MiB | 1,648.875 MiB |
| B8 | 1,942.5 MiB | 2,198.5 MiB |

The eight D prompts have lengths 978, 886, 926, 921, 954, 935, 903, 958; cap=512. Every prompt fits one prefill, its staging KV capacity rounds to 1024, and its requested decode capacity rounds to 1536. The general plan maximum is 2048, but it is unnecessarily pessimistic for these D fixtures. `DecodeBatch.rebuild` preserves an existing larger L while rows remain; no D request requires it to grow beyond 1536.

These are exact array-payload sizes, **not process-memory upper bounds**. Allocator rounding, old and new graphs, weights, projection activations, prefill workspaces, comparison temporaries, graph metadata, Metal command buffers/pipelines, framework cache, and unrelated processes are excluded. The 2 GiB `set_cache_limit` limits free cached buffers, not live arrays or physical footprint, and is process-global, not 2 GiB per arm.

## Proven backing retention and simultaneous ownership

Frozen `model/gdn.py:26-34,142-161` builds `xp=[zeros(3,C);raw_qkv(T,C)]` and stores `xp[T:]` as the three-row conv state. MLX 0.32.3's GPU Slice invokes shared-buffer slicing, which copies the underlying Data shared pointer, not only the selected bytes. Source evidence is in `.artifacts/m5-ultra-256gb/conv-tail-review-v1/`: `mlx__backend__gpu__primitives.cpp` Slice, `mlx__backend__common__slicing.cpp:20-35`, `mlx__array.cpp:193-204`. Thus the tail can keep the full `(T+3)*10240*2` payload per GDN layer alive after evaluation. It does not imply every preceding activation also remains alive.

| Two fresh prefills | Logical conv tails | Referenced xp payload, all 48 layers | S + KV + referenced xp |
|---|---:|---:|---:|
| T926 + T921 (step 514) | 5.625 MiB | 1,737.1875 MiB | 2,153.1875 MiB |
| T954 + T935 (step 515) | 5.625 MiB | 1,776.5625 MiB | 2,192.5625 MiB |

For step 515, this single retention path exceeds logical tail bytes by 1,770.9375 MiB per arm, or 3,541.875 MiB across two arms. Removing this path does **not** promise that exact reduction in process footprint: other aliases may retain xp and released buffers can move into the allocator cache.

The ordinary decode path also stores `xp[:,1:]` (`model/gdn.py:216-225`), retaining four conv rows for a logical three. That is an additional 0.9375 MiB per batch row across all GDN layers, much smaller than prefill retention. A B6 decode-state snapshot therefore has 5.625 MiB additional conv backing beyond the logical table. After retirement gathers, the B4 conv arrays have compact three-row outputs.

`qualify_native.py:71-85` snapshots create independent array handles with slices; they initially share buffers. Never count each snapshot as another physical copy. Nevertheless, snapshots keep *old versions* alive when core code subsequently replaces state:

- `Observer.call:270-274` retains the current decode state and both prefill caches. `Observer.run:237` takes postrun snapshots before retirement.
- `LLMEngine.step:63-75` then retires rows. `ModelRunner.drop_many:199-202` compacts surviving rows into new arrays, while calls/postrun retain the pre-retirement state.
- `run_case:392-399` clears reference observations before running reference; candidate's previous-step calls/postrun remain alive throughout that reference run. It clears candidate observations only immediately before running candidate.
- Calls and postrun pointing at the same version share storage. `previous_future` can retain another version during ahead use, but the observed steps 513–515 include prefill, so `ModelRunner.run:82` prevents a new ahead on those steps. There is no justification for adding an ahead generation for these particular points.

The reviewers observed step 515 as `before batch B4 -> sync/decode B6 + two fresh prefills -> retire two -> batch B4 + staging two`, all L=1536. The hypothetical next B8 reconstruction is not established by this row; after two retirements the next decode can instead be B6.

After both arms' step-515 retirement, named state versions can hold, **per arm**, pre-retirement B6 (1,456.875 MiB logical + 5.625 MiB decode conv backing), post-retirement B4 (971.25 MiB), and the two fresh staging caches (2,192.5625 MiB): **4,626.3125 MiB**, or **9,252.625 MiB = 9.03577 GiB across both arms**. This is an inventory of named payloads with shared call/postrun aliases counted once; it excludes all items listed above and is not a measured physical peak.

## Rebuild's additional capacity and specific copy sites

Frozen `cache.py:116-145` constructs an advanced-index gather for kept K/V, S, and conv even when `keep == list(range(old_batch_size))`; it then concatenates the kept rows with incoming rows. New staging K/V is padded to L separately before concatenation. `_sync_batch:180-182` pops staging entries but retains them in the local `new` list until rebuild returns; lazy expressions can retain their source buffers beyond that return.

For L=1536 with no L growth, an explicit source-level materialization envelope is:

`old compact batch + two staging caches + kept-row gathers + two padded KV rows + reconstructed batch`.

| Transition | Old batch | Staging backing | Gather | New KV pads | Output batch | Sum |
|---|---:|---:|---:|---:|---:|---:|
| B4 -> B6 using step-514 prefills | 971.25 | 2,153.1875 | 971.25 | 192 | 1,456.875 | 5,744.5625 MiB |
| Conditional B6 -> B8 using step-515 prefills | 1,456.875 | 2,192.5625 | 1,456.875 | 192 | 1,942.5 | 7,240.8125 MiB |

The sum conservatively counts those named materialized payloads together. It is **not** proof that they are simultaneously allocated: evaluation may recycle inputs, views share backing, and concatenate/pad can optimize. It is also not an upper bound on hidden workspace, allocator rounding, or observer-retained older versions. If the old batch is a decoded view rather than compacted output, add its four-row conv excess. If L grows, count padded kept K/V separately too.

Concrete default-off candidates, each tested independently:

1. Compact only the final three prefill conv rows to owned storage. Keep arithmetic and every output/state bit identical; this removes a proven large retention edge. The separate conv-tail review establishes the version-specific `contiguous` allocation condition; merely writing `.copy` or adding a view is not sufficient evidence. Explicitly evaluate the compact tail and measure its backing before claiming release.
2. In `cache.py:128,140-141`, skip advanced-index gather only for identity `keep` with unchanged L, passing the old arrays to the existing concatenate. This avoids a possible extra kept-state materialization (971.25 MiB for B4 at L1536). It preserves row order and keeps the necessary new combined output. Actual allocator benefit needs measurement, not just a Python operation count.
3. Retirement compaction followed immediately by refill concatenation copies survivors twice (`drop_many:202`, next `_sync_batch:182`). Combining them would require changing cache lifecycle semantics and observer qualification, so it is a later experiment, not the first diagnosis or a silent baseline change.

Prefill workspaces also matter. For T954/T935, chunk padding gives 960 tokens: one FP32 `[48,960,128]` group is 22.5 MiB; one `[48,15,64,64]` group is 11.25 MiB; solve output `[48,15,64,256]` is 45 MiB; each recurrent S version is 3 MiB per layer. `kernels/gdn.py:43-78` has several such arrays and a chain of 15 state updates. Their evaluation liveness/reuse cannot be recovered by summing logical shapes. Both prefills are built before the runner's final eval; the observer snapshots do not insert an immediate model-output eval in `call`.

## Minimum bounded diagnostic, after pressure recovers

Do not rerun the complete 4,096-step case for this question. First use an isolated layer-sized state-only child (no model weights), reproducing just compact B4 -> B6, two synthetic T954/T935 conv tails, B6 -> B4 retirement, and observer aliases. Run at most three cycles, one allocator policy (2 GiB), with owner hard timeout/resource stop. An explicit later full-engine diagnostic must use a fresh plan and remain a correctness run, not performance evidence.

The installed `mlx/core/__init__.pyi:831` declares `mx.get_array_buffer_size(*trees)`: inputs must be evaluated; it counts each **unique backing buffer once**, using full allocator-buffer size. It excludes graph objects, cached buffers, and other non-buffer allocations. This is a static API finding; the diagnostic must check `callable(getattr(mx, 'get_array_buffer_size', None))` and fail closed if unavailable.

Pseudocode, not executed:

```python
# leaves() reads existing array attributes/dicts/lists; it creates no slices.
# Observer.calls/postrun['arrays'] already provide array trees.
def sample(label, evaluated_groups):
    groups = {k: list(leaves(v)) for k, v in evaluated_groups.items()}
    flat = [a for arrays in groups.values() for a in arrays]
    # Only at declared diagnostic boundaries; changes liveness, never a timing run.
    mx.eval(*flat)
    mx.synchronize()
    union = mx.get_array_buffer_size(flat)
    record(label=label,
           active=mx.get_active_memory(), peak=mx.get_peak_memory(),
           cached=mx.get_cache_memory(), union_backing=union,
           group_backing={k: mx.get_array_buffer_size(v) for k, v in groups.items()},
           without_observer=mx.get_array_buffer_size(
               groups['committed'], groups['staging'], groups['ahead']))
    # union - without_observer = extra backing held only by observer roots.
    # Do NOT sum group_backing: groups can share the same underlying buffers.
    # Do NOT deduplicate by Python id: different handles can share a buffer.
    mx.reset_peak_memory()  # next phase's peak; retain all previous samples
```

Record these bounded phases: initial evaluated roots; after refill materialization; after each of the two prefill calls; after snapshot creation; after retirement; after both arms' comparisons; after clearing observer roots; after clearing all experiment roots plus synchronize/GC; finally after `clear_cache`. Immediately before graph-building phases also record active/cache/peak without evaluating newly built graphs. A build-time peak cannot be inferred from a later endpoint alone.

Pair each synchronized phase with child PID/start identity and monotonic timestamp. The owner should obtain that child's actual `phys_footprint` (the already-used `/usr/bin/footprint -p PID --noCategories --sysFootprint` route), plus system pressure/swap and external-service identity. RSS remains auxiliary. Footprint sampling and synchronization are diagnostic overhead; do not put those runs into throughput/latency plots. Initial child service state and final restoration remain the owner's responsibility.

If backing/active returns after observer and cache release but physical footprint does not, inspect allocator/system accounting before calling it a leak. If observer-exclusive backing or active memory grows cycle over cycle after identical release boundaries, save those bounded receipts and locate the remaining root. Compare compact-tail OFF/ON only after all original-dtype tail and state bytes pass; no allclose gate, no precision change, no ignored EOS.

## Source identity

Paths relative to the repository unless otherwise stated. `core` means `.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine/`. Only these small files were hashed; no weights were read.

| Source | SHA-256 |
|---|---|
| E023 prereg-qualification-v1/plan.json | `21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb` |
| pareto_realistic/qualify_native.py | `26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf` |
| core/cache.py | `056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791` |
| core/engine/model_runner.py | `14fd964c57814846887585a9cbbdf1e11d2500ab80c23b10c945d2aa9316b7b1` |
| core/model/gdn.py | `c88096bf3b51a1ad7a24dd4f4e2c71bbb6b220b288b83a9242ca190d6c0a5030` |
| core/kernels/gdn.py | `6be56a7d7ef41d3d422973a5385f598182b56f2cfd7495f6559d964460b6b0ab` |
| pinned model config.json | `7380cfc6eb77891381ee24e944e6bd0caed7a957707f2512ddd6effc82c7e706` |
| .venv/.../mlx/core/__init__.pyi | `bfa3c09f0f1e3edb925bf92e021f3121f0b953959f43ffd91cfa13895a41e8ce` |

MLX slice ownership was cross-checked against the small official v0.32.3 source copies and SHA receipts in `.artifacts/m5-ultra-256gb/conv-tail-review-v1/sources.json`. Runtime execution of that route and all future memory differences remain unmeasured in this review.
