# Lithos D512 qualification taps

CPU implementation only; no Lithos D512 device result or performance point yet.
The experiment is 8 frozen D requests at C1 or 16 at C8, cap 512, greedy,
normal EOS. This is route/request/raw-count/text/cache qualification. It does not
establish cross-framework token equality or full-model state equality. Official
`/api/chat` queues callers and runs one generation at a time; C8 is HTTP
concurrency, not an eight-row GPU batch.

`serve_probe.specification(out, C, port, key, observed=True)` reuses the pinned
E020 Lithos-none target and existing 2048-capacity pack, exact 128-row prefill,
`--no-draft`, `--no-warmup`, `--keep-warm 0`. It returns a command/environment
for an external direct-process owner. Nothing here owns a service, child or GPU
queue. `observed=False` returns the original `python -B -m monolith.serve` CLI;
the probe is absent from that command. No package files are modified.

The probe command is:

```text
LITHOS_PYTHON -B serve_probe.py --run --out NEW --concurrency 1|8 -- ORIGINAL_ARGS
```

Only `--run` imports installed Lithos modules. Default/`--help` is CPU safe.
The external owner must pin and check the target, pack, Python/runtime, source,
actual environment and original arguments; enforce resource/deadline bounds;
retain logs; and reap the direct server. This module verifies the five named
source seams but does not claim to encapsulate all runtime dependencies.

Three scoped taps call the unchanged original methods once and return their
original objects/strings:

1. `Session.generate`: retains full prompt IDs/cap and returned
   `Generation.tokens`, cache/setup fields and actual session/program route.
   Its per-worker pending record contains Python metadata only.
2. `WireResponse.body`: binds that worker's generation to the actual internal
   wire UUID, official metrics and original final response object.
3. `WireResponse.finish`: retains the exact emitted NDJSON line. Ollama does
   not expose its internal UUID on the wire. The offline join requires a unique
   exact final-line bijection, rejecting ambiguous duplicates or missing lines.
   It never matches by completion order: the worker releases the generation
   lock before its final NDJSON has necessarily reached the client.

The last expected terminal line seals 8/16 complete observer records and
rechecks the five sources. The installed Uvicorn re-raises SIGTERM after graceful
shutdown (`uvicorn/server.py:332–349`), so the seal does not depend on the official
main function returning normally. Client EOF/close and external owner reap are
separate mandatory evidence. Later observer events make the join fail.

`jobs_for(fixtures, C, served_model)` builds the original single-user bodies.
Use the frozen `pareto_realistic.lithos_client.run_group(...,
phase='qualification', qualification=None)` and keep its whole `group.json`.
Pass `headers={'Authorization': 'Bearer '+key}` using the owner's same API key;
the official server enforces it and the client does not invent credentials.
After the owner has reaped the server, call:

```python
result = join.join(group, observer_jsonl_lines, jobs, fixtures_by_id, decode_text,
                   probe_sha256, expected_official_args=official_args,
                   expected_pack_dir=existing_pack_path)
```

`decode_text` must use the pinned tokenizer with `skip_special_tokens=True`,
cleanup disabled, as official `Backend.visible_text` does. No response text is
re-tokenized. The join checks original request-body bytes, full prompt IDs,
actual returned tokens, a unique terminal EOS (including EOS exactly at cap),
no post-EOS IDs, final counts, content, original NDJSON framing and EOF. It also
scans the retained finite, positive POST-to-DONE intervals and requires actual
peak HTTP outstanding to equal C. Tied DONE/POST times retire the old request
first; a serial history labelled C8 fails. This proves observed HTTP concurrency,
not GPU batching or sustained occupancy throughout the run. Raw scope
is `Generation.tokens`, which official code truncates at the output cap; it is
not an assertion about every unreturned ring-buffer token. With no EOS, exactly
512 IDs and `length` are required. On EOS, `eval_count` includes that final ID;
committed count excludes it once. `Generation.decode_tokens` is not this count.
Actual `Session.prefill_exact` is false for chunk 128 because its extra flag
only applies above 128; this does not change the registered 128-exact CLI route.

Formal observer-OFF cache/setup evidence is available only in official logs.
NDJSON has no cache field; `load_duration` mixes load and setup. The owner must
record the same log file's identity and byte offset after all 8 warm responses
have reached EOF/closed and before the first formal POST; record the ending
offset after all 64 formal responses and closes. Reject rotation, truncation,
partial boundary lines, extra traffic and missing/extra Generation records.
Then `generation_log_window(window_bytes, 64, require_setup_zero=True)` requires
all 64 official records to have cache=0 and setup=0. Verify the 8 warm records
separately with `require_setup_zero=False`: cache still must be zero but initial
setup may be nonzero. This proves every record in a complete isolated window is
zero, not a per-wire request-ID association. The function accepts bytes only;
the external owner still has to establish the window boundaries and isolation.
The default exact log prefix is `INFO:__main__:Generation:` because the formal
entrypoint uses `-m monolith.serve`. Imported qualification uses
`expected_logger='monolith.serve'` explicitly; a mixed/unknown prefix fails.
`e020-generation-fixture.log` is the unchanged first Generation line from
`.artifacts/m5-ultra-256gb/external-spec-http-v1/pair2-lithos-none/server.log`.

CPU validation: `python3 -B -m unittest
bench.m5_ultra_256gb.lithos_d512.test_cpu -v` (15 tests). Fakes are explicit.
One test executes the original pinned `WireResponse` AST and string helpers
without package imports, then passes their emitted NDJSON through the taps/join.
The first 11-test run had one inadequate mock decoder (different IDs decoded to
the same fake string); the mock was made token-sensitive. Raw evidence logic
was not changed for that failure. Subsequent runs: 11/11 then 12/12 PASS. Source
self-review found the formal/imported logger distinction before execution;
the thirteenth test uses the saved original E020 line to cover both prefixes.

The two additional interval tests reject serial C8, excess overlap and zero
duration, and verify the tied-endpoint boundary. All 15 CPU tests passed; no
Lithos D512 device qualification or performance run was executed.
