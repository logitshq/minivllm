# GLM‑5.3‑Flash load-memory diagnosis — 2026-10-19

Read-only review of the failed `glm-flash-v3-owner-v1` cycle and its installed runtime. No model/backend import, GPU work, service action or retry was performed. Only this review file was written. A read-only `sysctl` check after recovery returned `iogpu.wired_limit_mb: 0` and `hw.memsize: 274877906944`; this is a current observation, not a recorded limit from the failed child. The first sandboxed read was denied; the permitted read succeeded without modifying either value.

**The zero-global-swap gate was the immediate reason for cancellation; a successful 190 GiB resident fit has not been established.** In particular, the small swap delta hides substantial compression already recorded before the abort. Physical fit, swap-free loading, and swap-free warmed inference are three different questions. This run answered only that its cold-load contract failed.

## What the retained samples prove

Source: [cold sample](../../.artifacts/m5-ultra-256gb/glm-flash-v3-owner-v1/cold-1/sample.json), [abort sample](../../.artifacts/m5-ultra-256gb/glm-flash-v3-owner-v1/physical-0031/sample.json), [monitor](../../.artifacts/m5-ultra-256gb/glm-flash-v3-owner-v1/monitor/monitor.json), and [load events](../../.artifacts/m5-ultra-256gb/glm-flash-v3-owner-v1/phases/glm-official-load/probe/events.jsonl). GiB means 2^30 bytes; `vm_stat` uses the recorded 16,384-byte pages.

| Metric | Cold | Last failed sample 0031 |
|---|---:|---:|
| `footprint` system accounting | 23.331 GiB | 203.920 GiB |
| Child footprint | child not started | 182.446 GiB |
| VM pages free | 204.053 GiB | 0.091 GiB |
| File-backed pages | 21.146 GiB | 50.437 GiB |
| Pages stored in compressor, logical | 0.659 GiB | 110.096 GiB |
| Pages occupied by compressor, physical | 0.208 GiB | 108.604 GiB |
| Global swap used | 1,587.50 MiB | 1,819.12 MiB |
| Global Swapouts | 743,802 | 758,682 |
| Pressure level | 1 | 1 |

The monitor explicitly reports `ValueError: runtime swap growth`; swap used increased 231.62 MiB and Swapouts increased 14,880 pages. The child/system 220/240 GiB ceilings were not crossed at this sample. However, accounting fields are not additive and `256 - sys_footprint` is not demonstrated available RAM: free pages were near zero and the compressor occupied 108.6 GiB. Pressure level 1 does not establish that pages stayed resident. Global counters cannot identify which process's pages were swapped or compressed; these samples do not prove all compressed pages belonged to the model.

The original `mlx_vlm.load(..., lazy=False)` returned on its executor thread after cancellation had begun. Its final scalar observation was active **190.128 GiB**, cache **5.573 GiB**, peak active **192.555 GiB**, guideline **243.200 GiB**. Active plus cache at that same boundary was 195.702 GiB; the active peak must not be added to a cache reading from another instant. No independent child/system physical sample covers the later boundary. `EnginePool.get_engine` did not return, gate/up fusion and engine admission were not witnessed, and **0 requests** ran. This is neither OOM evidence nor successful steady-state residency/performance evidence.

## Runtime mechanisms and bounded hypotheses

1. **The probe did not run the ordinary server's memory setup.** `load_probe.py:383–441` creates `EnginePool` directly and explicitly records that server memory-enforcer callbacks are absent. It calls neither `set_wired_limit` nor `set_cache_limit`. The installed MLX API declares default wired limit 0, a cache limit that defaults to its memory guideline, and a guideline that can allow allocation into swap; it is not a hard physical cap. The observed 243.2 GiB guideline exceeds the owner's 220 GiB child ceiling by 23.2 GiB. No child receipt records `device_info()`/recommended working set or an actual wired-limit setter result, so the effective runtime wired limit cannot be reconstructed from this trace. Do not infer it from 243.2 GiB or from generic “75% of RAM” commentary.
2. **Ordinary oMLX is not evidence that this probe had pinned buffers.** `cli.py:452–463` raises its buffer cache limit to total RAM; `process_memory_enforcer.py:245–285` deliberately skips changing MLX's wired limit when the kernel setting is 0. Its source warns against automatically releasing buffers still in flight. This means blindly setting cache limit to 0 is not the first control to try. The `mlx_vlm.generate.common.wired_limit` context is around generation; the failed cold load never reached generation.
3. **Load layout needs scrutiny before adding inference features.** `mlx_vlm/utils.py:965–967` accumulates all shard arrays in one dictionary, sanitizes/remaps/quantizes module shapes, loads parameters, then evaluates the complete parameter tree at `:1192`. The GLM sanitizer delegates to `deepseek_v32.Model.sanitize`; `deepseek_v32.py:525–534` lazily stacks per-expert weights, scales and biases. The actual checkpoint has per-expert keys and no `switch_mlp` keys. Its retained header inventory reports 183,911,841,792 bytes (**171.281 GiB**) of backbone routed-expert payload, a maximum single projection stack of **1.547 GiB**, and maximum new per-layer gate/up payload of **2.531 GiB**. Input and stacked-output buffers can overlap during materialization. Their exact live overlap, file-cache contribution and compression provenance were not observed. The recorded MLX active peak was only 2.427 GiB above final active bytes: it does not support a full additional 190 GiB copy counted as MLX active allocations. A large untracked/file-cache/driver or reference-retention contribution remains a hypothesis, not a measured cause.
4. **Late cleanup cannot rescue this aborted load.** `engine_pool.py:4024–4035` synchronizes and clears loading buffers only after `engine.start()` returns. `moe_gate_up_fusion.py:107–136` also evaluates/clears between fusion operations, but fusion was not reached. These paths suggest a bounded loader, not merely a higher external ceiling. The reported cache after load was only 5.573 GiB; cache alone does not explain the entire 108.6 GiB compressor occupancy.

## Smallest next diagnostic/control

Keep the original failure, all physical ceilings, zero-swap cold-load gate, cancellation/reap/recovery guarantees, and the formal timing contract unchanged. The most relevant first-principles control is a separately registered, default-OFF **projection-bounded loader**: retain the original per-expert order and original `mx.stack`, evaluate one projected weight/scales/biases group, then explicitly drop consumed source-array references, synchronize and clear retired buffers before advancing. The header-derived single-projection payload bound is 1.547 GiB; that is a tensor payload estimate, not a physical peak promise. Do not load a second full model as a reference. Validate resulting stacked tensor bytes against the concatenated original expert bytes with a bounded read/compare, without dequantization or tolerance changes.

A patch to the innermost `mx.stack` alone is insufficient if upstream `weights`, `remapped`, or `lang` dictionaries still keep every consumed source handle alive. Review ownership across `mlx_vlm.load_model → GLM Model.sanitize → LanguageModel.sanitize → DSV32Model.sanitize`, and either transfer dictionary ownership through a specialized loader or postpone eager evaluation until those upstream aliases are out of scope. Do not add eager `mx.eval` while retaining the entire input dictionaries: that could make overlap worse. The control must retain the official key transformations, per-module quantization, dtype conversions, vision handling and final strict parameter checks.

Before each diagnostic run, record `device_info()`, active/cache/peak/guideline and intended allocation policy. Record scalar boundaries for each shard load, sanitize, each bounded group, final materialization and synchronized cleanup. Wrappers must call each original exactly once, retain no extra array references, and write bounded scalar metadata only. The existing parent already captures raw compressor/file-backed/free/swap values; summarize their time series too. This will distinguish an accumulation of references from GPU evaluation overlap or kernel/file-cache effects. These are diagnostic boundaries, never timed-performance measurements.

A smaller independent allocator experiment is an explicit child-only **200 GiB memory guideline**, compared with observed 243.2 GiB; it may trigger allocator throttling/reclamation earlier while remaining above the observed 192.555 GiB active peak. It is an optional separate arm, not an asserted remedy or hard cap; record the prior setter return and restore it only after the load thread settles. Do not combine it with the first loader control, raise system `iogpu` limits, pin ~200 GiB blindly, or set the cache to zero. The priority is to bound live buffers and leave measurable resident headroom.

A sequential repack into already-stacked checkpoint tensors is another way to eliminate the repeated layout conversion, but requires disk budget and exact tensor-byte validation. It is a distinct implementation experiment and must preserve the original downloaded checkpoint and failed evidence.

Only after a successful load and bounded warm-up should a *new, explicit* experiment ask whether a settled model can run with zero additional swap/Swapouts. Starting a post-warm counter baseline would answer that different question; it must never erase this cold-load violation or silently replace the existing qualification. No formal latency/throughput point may be admitted from either load diagnostic.

## Exact local sources inspected

Paths below identify the installed sources, not upstream latest. SHA256 records the bytes read during this review; existing cycle-pinned files remain unchanged. Additional inspected files such as MLX stubs, CLI, enforcer and `deepseek_v32.py` were not all in the original 24-file runtime inventory.

- [probe](/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/glm_flash/load_probe.py:383) — `cd9df050384af83d92f0321d95f5bba70dca349defbedf0bb7d1d1e8b57385fc`
- [physical gate](/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/glm_flash/physical_guard.py:39) — `908e0dc74812fd82b65c63a22e03ed8c5692e812bdbf3d6957e219aa52b14166`
- [MLX API contract](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:867) — `bfa3c09f0f1e3edb925bf92e021f3121f0b953959f43ffd91cfa13895a41e8ce`
- [VLM loader](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/mlx_vlm/utils.py:965) — `2045e18d1a16222d07b157251e4a5108b2b12bf740b885f9682a071c8dc59134`
- [expert stacking](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v32.py:525) — `c2140067fb3e35d5f195b051b2189d1170ba11d144a73aac810cc859f5d2fe4d`
- [GLM sanitize](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py:2020) — `1e449afa2f435d6b8c8a1ca4ffb697fd0476262572b536330a913f282022ca5a`
- [VLM engine](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/engine/vlm.py:2148) — `b24406d697cd549ed083a9cb83e21cec4a7efb3f11f9fbf3747ae6bccdbf5121`
- [pool cleanup](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/engine_pool.py:4024) — `48e326e94cfc5e84fa1ab8da96ff2c50de68c912a71119e046420d3df3f192aa`
- [CLI cache](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/cli.py:452) — `721b4012d5bd9b8e3b67e8fc321020e4271a2396661ebd795c12ff007d2c5ce3`
- [wired policy](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/process_memory_enforcer.py:245) — `b7541598a1e06e4117849f1d9fdff0209d4e261a81aae11009bf507f0786e97e`
- [generation wired context](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/mlx_vlm/generate/common.py:213) — `c69e7e38a09990404d299b0a8d4be55c8220e60652e67a2456e8177633813ba0`
- [fusion](/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/patches/moe_gate_up_fusion.py:107) — `440bf3effd8b82258f3d600e4edfeacf48b169d37d25b97f7af998af8c781b9e`
