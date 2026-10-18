# oMLX D512 qualification

Actual C1/C8 qualification is now complete: 24 requests and three independent audits PASS.
See [the closed result](../../../docs/experiments/m5-ultra-256gb/E023-omlx-d512-qualification-v1/README.md).
This adds no performance or cross-framework full-state equality claim. The original preparation notes below remain historical; the executed cycle uses the versioned v2 observer/join, trial and committed owner/controller.

## Original CPU preparation notes


No service, model or GPU execution has been performed for this probe. It adds
qualification observations around the installed official CLI; it does not
change the official package or manufacture a native full-state qualification.

`serve_probe.specification(out, C, port, key, observed=True)` returns the original
official settings, symlink, environment and direct-child command for C1/C8. The
caller owns materialization, input/model/runtime pinning, exclusive resources,
readiness, requests, termination/reap and service restoration. `observed=False`
returns the original `omlx serve` command with identical settings and no hooks.
Nothing is written or started by this builder.

The qualification-only command returned by the builder is equivalent to:

```sh
/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/bin/python -B \
  /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/omlx_d512/serve_probe.py \
  --run --out NEW/probe --concurrency 8 -- serve \
  --model-dir NEW/models --base-path NEW/state --host 127.0.0.1 --port PORT \
  --api-key KEY --headless --no-hf-cache --no-cache \
  --max-concurrent-requests 8 --sse-keepalive-mode off
```

`jobs_for(frozen_fixtures, C)` produces 8/16 normal-EOS D512 requests. The owner
uses the fixed `pareto_realistic/native_cleanup.run_group` client with
phase=`qualification`, qualification=None and
`headers={'Authorization': 'Bearer ' + key}` (the same key passed to the CLI).
Save its original `group.json`.
After client completion and server cleanup, call:

```python
from bench.m5_ultra_256gb.omlx_d512 import join
with open("NEW/probe/observer.jsonl", "rb") as stream:
    result = join.join(group, stream, jobs, fixtures_by_id,
                       decode_text, expected_probe_sha256)
```

`decode_text` must use the externally pinned target tokenizer with
`skip_special_tokens=False`. The outer plan pins this module, the probe SHA,
the fixed fixtures, settings, official package/runtime and target model. The
join raises on missing/failed/extra evidence; callers must retain failure and
partial raw artifacts. A result cannot be obtained from CLI flags alone.

The three observer seams copy only Python metadata. A ContextVar scoped to each
original async-generator advancement binds official response UUIDs to real
engine request IDs without passing a replacement `_request_id`. The scheduler
tap records actual raw tokens before the official EOS exclusion and committed
IDs/count/cache after it. It reads the loaded-model target-only route switches;
it never calls a parser factory, evaluates arrays or changes output objects.
The original generator, method arguments/returns and SSE chunks are preserved;
all three attributes are restored on scope exit. Protocol DONE is recorded
separately from generator EOF because the original client closes after DONE.

The join reparses original SSE, compares server/client wire SHA and bytes,
requires unique response↔engine↔client IDs, full prompt IDs and body binding,
greedy effective sampling, actual target-only routing, cache=0, normal EOS/cap,
committed IDs/text and exact usage. C8 also requires a real multirow response
batch. It reports actual early-EOS coverage (possibly zero) rather than assuming
it. PASS is **observed protocol/route/count qualification only**;
full-state numerical, model/resource ownership, cross-framework quality parity
and performance flags remain false. A separate owner must bind those scopes
before any formal point. Observer runs never provide latency/throughput results.

CPU mechanism tests (Python 3.13, not macOS's old system Python):

```sh
cd /Users/ryanli/code/minivllm
.artifacts/omlx-baseline/.venv/bin/python -B -m unittest \
  bench.m5_ultra_256gb.omlx_d512.test_probe_cpu -v
```

These tests use fake engines and original-format synthetic wire bytes. They
cover real async task interleaving, context cleanup, request-ID association,
unchanged outputs, early EOS versus length, cache/route/count failures, retained
wire tails, incomplete cleanup and default-off formal instrumentation. They do
not load the official model or prove actual server compatibility on device.
