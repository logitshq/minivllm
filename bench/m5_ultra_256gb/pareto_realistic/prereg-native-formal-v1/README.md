# E023 D512 native formal timing: first C1/C8 pair

2026-10-17, registered before GPU execution. Status at registration: PREPARED.

The actual recorded C1/C8 HTTP histories independently passed full-state numerical,
source/runtime, observed-resource and restoration audits in commit 0555f52. This plan
measures the same native baseline with both qualification observers disabled.
It binds the exact model, D fixtures, transport, sampling, source and independent
numerical receipts. Qualification covers the captured representative histories;
it does not claim every subsequent free-arrival schedule was replayed numerically.

Plan SHA: `7b7a563ee832c1f91c2f5c745ec0b3087ef25dae6d8974cbfa6e0cb9a23687bb`.

| Trial | Fresh server | Warm requests | Formal requests | Outstanding requests |
|---|---|---:|---:|---:|
| r1-c1 | Yes | 8 | 64 | 1 |
| r1-c8 | Yes | 8 | 64 | 8 |

Every request uses its frozen full prompt, greedy decoding, normal EOS and cap 512.
There is no cache reuse. The finite queue refills each completed slot immediately.
The count numerator is actual committed output tokens; the denominator is the
whole formal queue's first POST through last SSE DONE. Latency is each full
POST→DONE p50/p95. Raw SSE, original jobs, identity, all errors and failed tails
remain available. Single-pair points are labeled n=1; three alternating fresh
process repeats and same-workload official baselines remain subsequent work.
Historic 19-input / cap32 points are a separate comparison and will not be pooled.

The outer owner serializes the two real direct children and samples resources
outside the HTTP event loop. It keeps the original pressure1, AC/thermal and
sustained-swap gates, exact process identity and original-service absence checks.
Source/runtime/weight hashes are checked before and after timing. Failure stops
the next trial; every actually returned child is waited/reaped, with raw ps/port
absence. Unknown Popen outcomes never certify no child. The first or next trial
may prove a rejection before Popen using exact inner-intent and positive receipt.

The existing explicit user authorization covers cold stop and recovery. Controller
v4 binds the immutable HTTP-v2 RESTORED journal/audit to original PID 90204, preserves
the same service/model/settings and bounded restore checks, and uses its own real
worker CLI/source. A final wrapper always attempts recovery after stop. Original
model reload, fresh MTP/resident PLE logs and physical memory are checked; warm KV
and exact hidden runtime state are not claimed restored.

`receipt.json` pins 20 sources and the actual PREPARED journal. Root 41 controller+
owner CPU tests passed, as did actual owner CPU preparation and read-only service
preparation. Per-trial runner/numerical-audit root 21 CPU checks were already recorded
with the prior archive. No GPU work occurs during these preparations.

Run the fixed wrapper from the repository:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/pareto_realistic/prereg-native-formal-v1/execute.py
```

Each trial has bounded ready 120s / request 600s / warm 720s / formal 2400s / overall 3300s;
client close 10s, TERM/KILL 15s each and resource interval 5s. The outer work budget is
7004s plus separate 120s postflight, not an expected runtime. Successful individual
trials save count-eligible points immediately. Final performance eligibility still
requires independent raw-count/resource/restoration audit after the cycle.
