# Lithos D512 formal C1/C8

This directory prepares six fresh observer-OFF trials in the fixed order
`r1-c1, r1-c8, r2-c8, r2-c1, r3-c1, r3-c8`. Every process runs eight warm
requests, waits for complete DONE/drain/client-close, then runs 64 formal
requests. Total: 48 warm and 384 formal. Preserve every failed attempt; the
first failure stops the cycle. C8 means eight concurrent HTTP callers to the
original Lithos FIFO route, not an eight-row GPU batch.

The prerequisite is the actual closed qualification archived in
`docs/experiments/m5-ultra-256gb/E023-lithos-d512-qualification-v2`: all 24
observed requests passed the full prompt-ID, actual route, raw count, visible
text and cache checks, with separate resource and restoration audits. This
supports those observed semantics; it is not cross-framework full-state
identity, a new coding-quality test, or qualification of every future response.

## Fixed inputs and identity

`preregister.py --prepare` copies the frozen retry owner plan's complete
`verification`, base workload reference, HTTP environment, port and key. It
changes only the formal plan/source binding, six phases and these deadlines:

| Phase or bound | Seconds |
| --- | ---: |
| Ready / request | 180 / 600 |
| Warm / formal | 720 / 2400 |
| Trial overall | 3600 |
| Client close / TERM / KILL | 10 / 15 / 15 |
| Resource interval | 5 |

The owner work deadline is 22,212 seconds (300 preparation plus six bounded
trial/cleanup allowances); its existing 120-second postflight phase remains.
These are upper bounds, not expected durations or benchmark results.

The original NVFP4 snapshot, existing none pack, max-context 2048,
`128-exact` prefill, no-draft flags and four MLX environment values remain.
Official server command is `python -B -m monolith.serve` with observers OFF.
The localhost port remains 18881. The inherited experiment-only key is
`minivllm-lithos-d512-local-qualification-v2`; it is not an account credential.
Every client request uses `Authorization: Bearer <key>`, exactly matching
`LITHOS_METAL_API_KEY` in the original environment. Do not substitute the current
shell environment for the frozen qualification environment.

Bodies use the original `fixture.messages`. Do not wrap rendered `prompt_text`
in a new user message. The formal body has normal EOS, cap512 and fixed greedy
sampling. The client contract references the real qualification audit SHA and
path; it contains no old output-token oracle and does no response retokenization.

## CPU preparation and execution boundary

Only root runs the following commands after all source placeholders have been
replaced with reviewed final SHAs. Run from `/Users/ryanli/code/minivllm`:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/lithos_d512_formal/preregister.py --prepare
```

The builder creates fresh `prereg-formal-v1/{owner-plan.json,preparation.json,
source-manifest.json,commands.json}`. It does not run a subprocess, query a
service, load a model, hash model/pack payloads, or create an execution receipt.
The generated commands separately identify CPU owner preparation, read-only
controller preparation, execution and offline audits. The five output paths are:

```text
.artifacts/m5-ultra-256gb/lithos-d512-formal-v1-owner
.artifacts/m5-ultra-256gb/lithos-d512-formal-v1-cold-journal
.artifacts/m5-ultra-256gb/lithos-d512-formal-v1-execution
.artifacts/m5-ultra-256gb/lithos-d512-formal-v1-analysis
.artifacts/m5-ultra-256gb/lithos-d512-formal-v1-cpu-prepare
```

The sole accepted predecessor is the archived v2 RESTORED journal/audit with
replacement PID28531. Root executes the generated CPU and read-only preparation
commands, checks their actual outputs, and writes `prereg-formal-v1/receipt.json`
with `owner_plan_sha256`, the complete `source_pins`, and the actual
`prepared_journal_sha256`. Include the source manifest and preparation receipts
in that final pin set. No dummy journal digest or synthetic PASS is acceptable.
Commit the frozen registration before explicitly running `formal/execute.py`.
That separate driver handles the authorized stop→owner→finally-recover cycle.

A prepare failure leaves its new directory and error record for inspection.
Do not overwrite or quietly replace a registered attempt. No service state or
request is changed by importing this builder or asking for `--help`.

## Point definition and publication gate

x = median of the 64 request POST→unique newline-terminated DONE latencies;
p95 uses nearest rank. y = total committed output tokens divided by
`last formal DONE − first formal POST`. Warm/load/hash/log/reap times are outside
that interval. EOF/close is outside latency but must complete successfully.

On `length`, raw `eval_count` must equal512. On `stop`, 1..512 is accepted and
one terminal EOS is subtracted exactly once, including immediate EOS and EOS
at the cap. Wire cache counts remain unknown. Separate complete Generation-log
windows must prove cache0 for every warm/formal request and setup0 for all64
formal requests; no guessed last64-line window or per-wire-ID log association.

Every fresh process retains its own point and complete raw wire/log evidence.
Plot all three points per C and their per-C median. The old native single-process
points remain historical comparisons, not a newly paired three-repeat trial.
Count-eligible producer points alone cannot enter the Pareto plot: the offline
formal audit must also close source/runtime/model/pack, log, pressure1, paging,
AC/thermal, direct-child reap and service-restoration gates. No full-state or
output-quality qualification is asserted by this formal run.
