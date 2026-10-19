# Borrowed GLM C1 text reference — CPU checked, device unexecuted

2026-10-18. This is a small reference facade over the **already loaded** tested
oMLX GLM-5.3-Flash VLMModelAdapter. It is default OFF and has no loader, model
imports, sampler, global patch, new executor, service command or lifecycle owner.
The 11 stdlib fake tests passed on 2026-10-19 under the guarded CPU runner;
exact source/log evidence is in docs/experiments/m5-ultra-256gb/next-candidate-cpu-2026-10-19. No model/HTTP/numerical/performance qualification is granted.

`bind_loaded(engine, loaded_engine_identity, enabled=True, mx=actual_mx,
assert_exclusive=caller_lease_check)` returns a reference. OFF returns `None`
without inspecting its arguments. The supplied identity is the existing load
probe's `loaded-engine.json`. Constructor source paths and object/class metadata
are checked; this does not rehash or independently verify source/model payloads.
The injected PoolingCache is located by constructor code filename, not its
rewritten `mlx_lm.models.cache` module string.

Run the **entire session in one submission to the existing EngineCore single
MLX executor**, on which its worker thread must already be registered. The
facade enters the engine/scheduler's same MLX stream. The caller must hold an
exclusive admission and immutable-parameter lease until `retire()` and `close()`
finish, and retain that lease until its own outstanding evaluation is complete.
Queue/UID/output maps must be empty and `batch_generator` must still be `None`.
This deliberately rejects an existing post-generation BatchGenerator: it does
not guess that its singleton merge preserved all typed cache state. No engine
stop/start is performed. The caller retains responsibility for actual route
checks (including embedded/block MTP), complete source closure and resources.

```python
# Inside the original executor submission, after official load and admission:
ref = bind_loaded(engine, loaded_identity, enabled=True, mx=mx,
                  assert_exclusive=lease.assert_held)
session = None
try:
    session = ref.new_request(prompt_ids)          # 1..128 plain IDs, B1 only
    ref.prefill_prefix(session, prompt_ids[:-1])   # original cache-state eval
    logits = ref.step(session, [prompt_ids[-1]])   # full [1,1,154880], lazy
    # Caller evaluates/saves the diagnostic logits + ref.snapshot(session)
    # here, before another step; subsequent IDs are explicit teacher forcing.
finally:
    if session is not None:
        ref.retire(session)
    ref.close()
```

At most 32 single-token steps and a 2048 consumed-ID ceiling are admitted. The
first pending ID must be the last prompt ID. Each subsequent ID belongs to the
caller; no recorded/generated token is silently substituted. No greedy/EOS
sampler or EnginePool request parity is claimed. Future generation diagnostics
must independently apply the actual normal-EOS/suppression policy; this API
alone is teacher forcing.

The original `adapter.make_cache`, `_omlx_prefill`, and adapter text call own all
state advancement, the model's internal async evaluation and mixed-precision
arithmetic. The facade never calls `cache.advance`, batches/merges/restores
caches, casts logits, edits parameters, removes vision, or changes compiled model
caches. Prefix cache-state evaluation matches the official scheduler's existing
boundary; `step` adds no eval/synchronize. It is a primitive call-boundary
reference, not a reimplementation of scheduler admission or output delivery.

Snapshots expose current **borrowed tensor handles**, original shapes/dtypes,
ArraysCache bookkeeping/speculation fields, full KV backing/offset/step and
PoolingCache remainder/full backing/previous-window/undo fields. Unknown fields,
active speculation and batch cache variants are rejected. Containers are new;
tensor payloads are neither copied nor materialized. Save/evaluate them before
another model call if historical evidence is needed. These are not durable raw
snapshots, finite/byte proofs, clones, or rollback checkpoints. A failed forward
poisons the session; retirement still drops its references, without rolling
back partially advanced state or closing the engine.

Source basis: tested wheel `omlx/models/vlm.py:258,585–724`,
`omlx/scheduler.py:4001–4012,4166–4181`, `mlx_vlm/models/cache.py:500–550,
838–876,1193–1284,1448`, `omlx/patches/deepseek_v4/cache_extras.py:35–68,141–188`,
vendor `glm5_next/language.py:1880–1942,1998–2018,2108–2123`.
See [the source review](../../reviews/2026-10-18-glm-text-facade-next.md).
There is no claim of reduced vision memory, B2/dynamic support, mini engine GLM
support or a Pareto point. Before device use: source review, guarded CPU tests,
explicit runtime/transitive-source binding, then independent original-dtype
logit/cache comparison against the actual official C1 call/cache boundary.
