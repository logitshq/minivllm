# External D512: shortest path to two measured points per engine

CPU/source plan, 2026-10-17. No external D512 device qualification or new speed
result is claimed here. Native recorded-HTTP qualification and one observer-OFF C1/C8
baseline are complete; external device qualification/timing remain pending.
First deliver C1/C8; C2/C4 require their own capacity/arrival qualification.

## Fixed common task and chart

- Eight prepared coding prompts, 886–978 tokens, greedy normal EOS, cap 512.
  `pareto_realistic/fixtures/prepared-D-v1/fixtures.json` SHA256
  `6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db`.
- Target snapshot:
  `/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`.
  Keep the existing NVFP4 weights, quality settings and EOS handling. Same weights
  do not imply cross-framework bit equality; earlier Lithos continuations differ.
- After each engine/C passes its own request/route/count checks: fresh official
  server, 8 warm requests then N=max(64,8C) formal requests, three fresh processes
  per point, alternating engines. At C1/C8 this is 64 formal requests per run.
- X: per-request POST→protocol DONE p50; Y: sum of actual committed output tokens
  / (last DONE − first POST). Show all three process points and their median;
  do not pool old E020 results or instrumented qualification timings.
- Environment: `MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50
  MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0`; resource/cold-service owner remains
  external. No observer in formal runs.

## oMLX target-only first

Reusable official launch/config construction:
`external_spec_http/launch.py` SHA256
`7b2ddc6d85b4f612b19c51711f7cd126f13b261fdc152e0a426ceae51d1d66e3`.
Its C1 constant needs a new caller/config builder, not a frozen-file edit.
Copy the **complete** `qualified_model_settings['omlx-none']` from
`external_spec_http/prereg-v1/plan.json` SHA256
`e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc`.
Both MTP flags, DFlash, SpecPrefill, TurboQuant and ANE/offload optimizations
remain OFF. New private state uses balanced mode, cache disabled,
chunked_prefill=false, prefill_priority=speed, decode_fairness=true, and
scheduler.max_concurrent_requests=C. Symlink `OUT/models/Qwen3.8-27B-nvfp4`
to the target snapshot. Original formal command (owner supplies PORT/KEY/OUT/C):

```sh
/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/bin/omlx serve \
  --model-dir OUT/models --base-path OUT/state --host 127.0.0.1 --port PORT \
  --api-key KEY --headless --no-hf-cache --no-cache \
  --max-concurrent-requests C --sse-keepalive-mode off
```

POST `/v1/completions`, model `Qwen3.8-27B-nvfp4`, original `prompt_text`,
temperature=0, top_p=1, repetition_penalty=1, max_tokens=512, n=1,
stream=true, stream_options.include_usage=true; no stop/min-token/ignore-EOS.
Use `pareto_realistic/native_cleanup.py` (SHA256
`6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6`)
for actual concurrent SSE requests and strict closer handling.

Narrow missing qualification: a separately versioned official output tap plus
raw-wire join. It must bind the real 8-hex `cmpl-` response ID to the engine's
request ID (the VLM path normally does not receive `_request_id`), retain actual
response tokens including terminal EOS and committed IDs, full prompt IDs,
actual target-only route, cache=0, cap/finish/count/text, and all 8/16 requests.
Original scheduler excludes EOS from completion_tokens: committed=usage, with
raw=committed+[EOS] only on stop. Never subtract EOS twice. Qualification is
protocol/route/count evidence, not a substitute for native full-state replay.

Official installed sources under `.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/`:

| Source | SHA256 | Relevant boundary |
| --- | --- | --- |
| server.py | b05c1388278db419082220ede5aa2e3e13569857a256f3f3ec3d60b800d708c4 | create_completion:4110; stream_completion:5437 |
| cli.py | 721b4012d5bd9b8e3b67e8fc321020e4271a2396661ebd795c12ff007d2c5ce3 | main:1114 |
| scheduler.py | 88bf4d988934a60e268d4205f30a86aae3012eefaef4b1f9672a0b86b412b083 | _process_batch_responses:12720; EOS exclusion:12770 |
| engine/vlm.py | b24406d697cd549ed083a9cb83e21cec4a7efb3f11f9fbf3747ae6bccdbf5121 | stream_generate:4856; add_request:4945 |
| engine_core.py | c9b37d06a462b0fbc253c2087b6bd1a74884a22ced16c55ac059344c76e39a1e | AsyncEngineCore.add_request:1446 |
| patches/mlx_lm_mtp/batch_generator.py | f758e39cff16b3aff036f621775cbd540d5a2dee2e0de228c63c3c333f51dea9 | drafter/embedded-MTP route guards |

MTP is deliberately a later point: its single-request route can fall back when
other requests are active (scheduler:10394–10428). A C8 CLI flag is not proof of
batched speculation.

## Lithos target-only next

Existing pack (no repack):
`/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/lithos-spec-packs-v1/none/target-b6042c39f78b9df33a19ac1c`;
capacity 2048, 16,069,099,520 bytes; manifest SHA256
`30b0723affb28128d896c85e5120553723ec7a615df8438a41736fd4a6b7ab96`.
Run from `/Users/ryanli/code/lithos-metal`, HEAD
`223658090ae549e59220bdec459957cbdad7926a`:

```sh
MONOLITH_ACCELERATOR=on LITHOS_METAL_API_KEY=KEY \
/Users/ryanli/code/lithos-metal/.venv/bin/python -B -m monolith.serve \
  --model TARGET_SNAPSHOT --pack EXISTING_NONE_PACK --local-files-only \
  --host 127.0.0.1 --port PORT --max-context 2048 \
  --prefill-chunk-size 128-exact --keep-warm 0 --no-warmup --no-draft
```

Use `/api/chat` with the fixture's original single-user messages and
options={num_predict:512,temperature:0,top_p:1,top_k:0,seed:0}. Its FIFO lock
admits C>1 HTTP callers but executes one GPU request at a time; report that
truthfully. Other serving routes return busy 429 under contention.

Use `pareto_realistic/lithos_client.py` SHA256
`13a629f096434bc6c422803bf7bfd8992e24396cb122880b493e893cc5ecd2e9`:
native NDJSON `done:true` terminal clock, followed by bounded EOF/close drain.
`eval_count` includes terminal EOS on stop; committed=eval_count−1 on stop,
otherwise eval_count. First qualify actual Generation.tokens, one final EOS,
no post-EOS IDs, visible text and exact usage. Preserve raw actual counts.
`bench/reviews/2026-10-17-lithos-d-prefix-cpu.json` shows these frozen fixtures
share only a 3-token prefix (<128 checkpoint): this is task-specific uncached
evidence, not a general cache-off switch. Actual per-request cached_prompt_tokens
must be 0; formal setup_ms must be 0 after warmup. Missing wire cache counts
are unknown, not fabricated zero.

| Lithos source | SHA256 |
| --- | --- |
| monolith/serve.py | 9759ee14c4860ed43218118d6a3eafbb680e9e722e3ddad72bae52473579f419 |
| monolith/generate.py | 434ac1398c6010b37fa44f138505d5ca1757f4e7b747bc6b582d386e2e3ba3b4 |
| monolith/serving/protocol.py | 5f015e2c55037bd0ae73f8b33e7eb64adf8fcb67f19f0f1dc2d25157b1f83ada |
| monolith/serving/events.py | aaa11acb7cb058f55916365e94a0b45b160f286356ffd1d610f08fbaf67b316d |
| monolith/runtime/prefix_cache.py | 2ffa0f9bfed792765d27af811f8df3d08f9c6b5709c05d6b1e996ba0721297ad |

Before execution the outer owner must recheck the existing model/runtime/pack
pins and enforce exclusive GPU/resource/reap/recovery boundaries. This plan
adds no service-control mechanism and grants no performance eligibility.
