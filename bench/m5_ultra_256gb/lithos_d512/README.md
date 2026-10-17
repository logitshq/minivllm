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

## Single fresh qualification trial

`trial.py` adds the bounded direct-process runner. It does not alter the frozen
probe, join, client, official package, model or pack. Prepare one spec per new
output directory and execute C1 and C8 sequentially through the external owner:

```python
launch = serve_probe.specification(out, C, port, key, observed=True)
spec = trial.prepare(out, C, port, key, deadlines=deadlines,
                     environment=registered_environment)
result = await trial.run(spec, out, checkpoint)
```

`registered_environment` must contain the exact flags/API key from
`launch['environment']`. Only explicitly recorded PATH/HOME/TMPDIR/LANG/LC_ALL/
USER/LOGNAME additions are allowed. The fixed command uses the original local
target and existing 2048 pack, with `cwd` equal to the Lithos repository; no pack
conversion occurs. `deadlines` contains positive seconds for `ready`, `request`,
`group`, `close`, `term`, `kill`, `overall`, and `resource_interval`. TERM and KILL
are 15 seconds each; the probe interval is at most 5 seconds and overall covers
ready plus group. The outer owner must bind these limits in its own budget.

The runner sends the same `Authorization: Bearer <key>` to official `/v1/models`
and the frozen Lithos client. It validates the actual served model/capacity and
the direct PID's localhost listener, then issues only 8/C1 or 16/C8 qualification
requests. There is no warmup inference, formal group, prior fabricated count
receipt or performance point. After client EOF/response/client close, it sends
TERM, escalates to KILL if necessary, waits, and requires raw PID and port absence.
Repeated caller cancellation cannot cancel the independent cleanup task.
An unresolved cleanup remains FAIL and names the child in `cleanup-incomplete`;
an unknown Popen outcome retains its intent without a false release.

`checkpoint(label, pid)` is mandatory and can be async. Labels are `pre-spawn`,
`spawned`, `starting`, `ready`, `qualification`, `after-qualification`, and
`after-reap` or `cleanup-incomplete`. The outer owner must settle any in-flight
resource probe before propagating cancellation, retain its direct child handle
for cleanup reconciliation, enforce exclusive device/service absence and full
runtime/weight/pack pins, then restore the service. This runner never controls
launchd or certifies those outer conditions.

Only after actual reap does it invoke the frozen full raw join, decode IDs with
official `AutoTokenizer` semantics, and check all official Generation log entries
for cache=0 with the imported `monolith.serve` logger. Qualification permits first
setup cost; this is not the later formal setup=0 gate. Logs, observer bytes,
partial client results, launch/readiness, intent/identity/release, joins and all
artifact hashes remain in the new directory. Trial PASS means this observed
route/count/text contract passed; full-state and performance fields stay false.

`trial.py --spec PATH --spec-sha256 SHA --out NEW_PREPARATION_DIRECTORY` validates
a saved spec without starting a process. The real trial requires the explicit
outer-owned `run` call. CPU tests use `httpx.MockTransport` with the real frozen
client and join; no model/server/service is exercised:

```text
.artifacts/omlx-baseline/.venv/bin/python -B -m unittest bench.m5_ultra_256gb.lithos_d512.test_trial_cpu -v
```

Ten tests passed, including C1/C8 raw flows, authenticated requests, missing seal,
wrong readiness/identity, unknown spawn, TERM failure followed by KILL, resource
cancellation, repeated cancellation during reap, and surviving PID rejection.

## Two-stage direct owner

`owner.py` directly awaits the unchanged `trial.run` in its own process, first
`lithos-c1` (8 requests), then a fresh `lithos-c8` (16 requests). Its tracked
backend forwards the registered `cwd` and retains the actual server Popen object.
There is no Python trial subprocess between the owner and server. The owner
settles each resource-probe thread before cancellation propagates, requires
pressure1/AC/thermal and the original swap guards, and independently repeats
TERM/KILL/wait plus raw PID/port-absence checks after the trial closes.

The new exact plan schema is `E023-lithos-D512-owner-plan-v1`, with fields
`schema`, `execution`, `qualification`, `cold_restart`, `http`, `verification`,
`trials`, `deadline_seconds`. `http` contains `port`, `api_key`, `deadlines`,
`environment`; the frozen trial defines those deadlines/environment. Work is
`300 + 2 × (overall + close + term + kill + 12)` seconds, rounded up; postflight
gets its existing separate120-second bound. `plan_template(...)` builds this
CPU-only recipe. The `qualification` reference binds the unchanged D fixtures
and resource helpers; it does not pretend to qualify Lithos numerics.

`verification={python,groups}` explicitly supplies five groups:
`lithos_runtime`, `lithos_source`, `target_model`, `pack`, `outer_runtime`.
Each group maps absolute paths to `{sha256,bytes,mtime_ns}`. Required Lithos
source/runtime and pack inventories inherit E020 hashes; the target inherits
the original model manifest. Resolved Python binaries and the actual outer
environment's httpx/httpcore/transformers/tokenizers are mandatory. The outer
tokenizer runs in the owner, so its runtime must be pinned separately from the
official server environment. Additional transitive dependencies belong in
these supplied groups. Full preflight and postflight hash every supplied file
in1MiB blocks with deadline checks, preserving open-file and pathname identity
before/after. Zero-byte files are valid. No native preparation receipt replaces
Lithos runtime/pack verification; OS shared libraries and stdlib are not fully
encapsulated by this inventory.

The owner requires a new `E023-lithos-D512-cold-journal-v1` STOPPED journal,
binding the exact owner plan/source and immutable restored predecessor. It
does not stop or restore a service. A separate reviewed `controller.py` and
restoration auditor must consume its real two-phase protocol before execution:
canonical `phases/{phase}/{intent,child,release}.json`, inner
`phases/{phase}/trial/{spawn-intent,child,release}.json`, and top-level
attempt/index/final release. Pre-Popen rejection has explicit `no-popen.json`
evidence; unknown spawn remains unresolved. A later-phase failure preserves the
already-reaped prefix. No numerical/performance PASS is required to recover a
failed experiment, but every observed child must be proven reaped.

The CLI is `owner.py --plan PATH --plan-sha256 SHA --out NEW [--prepare-only]`.
Prepare-only checks small metadata and source pins; it does not hash runtime,
weights or pack payloads or start a process. Actual execution is still pending.
Owner PASS means both observed route/count/text trials and cleanup completed;
`performance_qualified` and independent qualification remain false. Formal
observer-OFF timing and its separately bounded warm/formal log windows remain
unimplemented here.

CPU command: `.venv/bin/python -B -m unittest
bench.m5_ultra_256gb.lithos_d512.test_owner_cpu -v`. Final24 tests passed in0.167s;
isolated `-I -B owner.py --help` also passed. The first16-test run was15/16
because a cancellation fixture retained the old phase label. The next22-test
run was21/22 because the AST comparison expected the old error-message text;
both test-only mismatches were corrected. Additional tests cover all five pin
groups, zero-byte files, changed bytes with unchanged stats, path replacement
during reading, deadline interruption and retaining the primary error if an
error-receipt write also fails. All process/service/HTTP behavior in these owner
tests is explicitly fake; no real model, listener, service or device was used.
