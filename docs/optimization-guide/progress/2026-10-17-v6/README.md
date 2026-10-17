# Qwen3.8-27B D512: native measured operating points

M5 Ultra 256 GiB; identical fixed NVFP4 weights and eight D coding fixtures. Greedy normal EOS, output cap 512; all qualification observers OFF during timing.

| Concurrency | Fresh processes | Measured requests | Committed tokens | Window (s) | Output tok/s | p50 (s) | p95 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 1 | 64 | 32768 | 722.205195 | 45.372147 | 11.283287 | 11.348184 |
| C8 | 1 | 64 | 32768 | 195.747992 | 167.398907 | 24.431814 | 25.979063 |

X is complete per-request POST-to-DONE p50 latency. Y is the sum of actual committed output tokens divided by the common first-POST to last-DONE window. The client uses a closed loop at the stated concurrency; latency includes server queueing, not waiting before a client slot becomes available.

Each point has eight warm requests and 64 measured requests in one fresh server process. These are single-process observations, not confidence intervals. The dashed connector is a visual guide. The older output-cap-32 experiments belong to another workload and are not plotted here. oMLX and Lithos D512 results are pending; this chart establishes no competitor dominance.

Correctness scope: independent full-logit/state replay of representative C1/C8 recorded HTTP histories, plus raw fixture/UUID/text/EOS/count checks on every new measured request. It is not a replay of every formal schedule. The independent cycle audit also checks source/runtime/model receipts, observed resources, child reaping and service restoration.

Source audit SHA256: `b669350e95ef07f4da5cac3a5cf6fb6fdaf2aea57e886d3ba60b9afd15e950bd`. See [audit](independent-audit.json), [data](chartdata.json), [render source](render.py), [PNG](native-D512-throughput-latency.png), [SVG](native-D512-throughput-latency.svg).

Subsequent E024 long decode graph qualification failed exact logits at its first point. It contributes no chart point; the two native baseline observations above remain unchanged. See the [retained failed experiment](../../../experiments/m5-ultra-256gb/E024-long-decode-cold-v1/README.md).

The separate [four-arm follow-up](../../../experiments/m5-ultra-256gb/E024-dispatch-isolation-v1/README.md) identifies bound change as sufficient to reproduce the rejected candidate difference at B1/used257; same-bound compilation is exact there. It has no performance qualification and does not change these points.
