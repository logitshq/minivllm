# D512 native and Lithos observed operating points

Same eight D coding fixtures, 886–978 prompt tokens, greedy normal EOS, output cap512.

Discrete observed p50/p95–throughput tradeoffs only. Native n=1/C is historical; Lithos n=3/C is a new cycle. No paired trial, CI, maximum-concurrency or global-frontier claim. Lithos C8 is HTTP concurrency with official FIFO GPU execution. Not a cross-framework full-state or output-quality test.

| Engine | C | Repeat | Committed tokens | Window (s) | tok/s | p50 (s) | p95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mini-vLLM | 1 | 1 | 32768 | 722.205195 | 45.372147 | 11.283287 | 11.348184 |
| mini-vLLM | 8 | 1 | 32768 | 195.747992 | 167.398907 | 24.431814 | 25.979063 |
| Lithos | 1 | 1 | 32768 | 729.634050 | 44.910185 | 11.404540 | 11.521853 |
| Lithos | 1 | 2 | 32768 | 729.026220 | 44.947629 | 11.397076 | 11.517165 |
| Lithos | 1 | 3 | 32768 | 729.772161 | 44.901685 | 11.402589 | 11.531824 |
| Lithos | 8 | 1 | 32768 | 730.059929 | 44.883987 | 91.252783 | 91.313418 |
| Lithos | 8 | 2 | 32768 | 728.711273 | 44.967055 | 91.072059 | 91.186858 |
| Lithos | 8 | 3 | 32768 | 729.534247 | 44.916329 | 91.183519 | 91.324838 |

Throughput = committed output / (last formal DONE − first formal POST). Latency includes server queueing and excludes waiting for a client slot. EOF/drain/close must pass but is outside the latency interval.

Black outlines identify nondominated points among these eight observed processes only. Repetition spread is not a confidence interval. No interpolated frontier is drawn. Median summaries do not invent new observed coordinates.

Native numerical eligibility covers representative recorded histories. Lithos eligibility binds actual prior full-input IDs, route/count/text/cache checks; formal wire has counts rather than raw IDs. This comparison does not establish identical logits, state or coding quality across implementations.

[Data](chartdata.json) · [PNG](D512-native-lithos-pareto.png) · [SVG](D512-native-lithos-pareto.svg) · [Native audit](native-independent-audit.json) · [Lithos audit](lithos-independent-audit.json) · [Renderer](render.py)
