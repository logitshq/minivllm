# E035 C1 profile: spend the next experiment on computation, not HTTP plumbing

The actual OFF/runner/full cycle finished, all three server children exited naturally, and the saved independent diagnostic and restoration audits are PASS. This readout uses the audited eight diagnostic requests per mode, each emitting 512 tokens. Warm requests are excluded. It introduces no numerical or performance qualification and supplies no Pareto point.

**Main finding:** the already available `ahead_submit_once` optimization targets a call whose entire observed cost is only about 0.022% of local service time. It cannot plausibly deliver the large gain sought by deleting that measured call alone. The second async submission dominates the recorded wall interval and deserves backend/dependency/computation investigation. The data do not identify it as exclusively GPU work, memory bandwidth, or an avoidable stall.

## Observer comparison

| Mode | Audited aggregate output tok/s | Eight-request window, s | E2E p50, s | E2E p95, s |
|---|---:|---:|---:|---:|
| off | 48.41887 | 84.59512 | 10.54068 | 10.78651 |
| runner | 48.46129 | 84.52106 | 10.56130 | 10.61648 |
| full | 48.44073 | 84.55694 | 10.56509 | 10.61726 |

Runner versus OFF changes the observed common window by -0.08755%; full versus OFF by -0.04514%; full versus runner by +0.04245%. These differences are tiny relative to the per-fixture variation: the first two OFF requests were slower, while the other six requests were slightly faster than their observed counterparts. There is one fixed-order process per mode and no estimate of run-to-run variance. This does not prove negative observer cost or any engine speedup. All three modes also share admission/DONE/drain bookkeeping absent from the original formal server.

## Actual full-mode interval coverage

These are unions clipped to server request entry through DONE flush, totaling **84.543105 s**. Rows overlap and must not be summed. Compiled callables are nested inside decode submissions. Queue-get time includes blocking for output; it is not CPU service cost.

| Recorded wall interval | Union, s | Share of local service window |
|---|---:|---:|
| Engine step | 84.471235 | 99.9150% |
| Runner step | 84.409637 | 99.8421% |
| Existing async eval | 76.157459 | 90.0812% |
| Existing eval | 5.061150 | 5.9865% |
| Decode graph submission | 2.620069 | 3.0991% |
| Compiled callable (nested within submission) | 2.230384 | 2.6382% |
| Text decoding | 0.193766 | 0.2292% |
| HTTP write | 0.106571 | 0.1261% |
| HTTP flush | 0.001547 | 0.0018% |

- Engine outside runner: **0.061598 s**, or 0.0729%.
- Runner outside its nested recorded graph/sample/eval intervals: **0.556411 s**, or 0.6581%. This residual is uninstrumented wall time, not guaranteed removable CPU work.
- Request service outside engine: **0.071870 s**, or 0.0850%.
- Text decode/write/flush combined union: **0.301884 s**. Of this, **0.301373 s** (99.831%) overlaps worker engine spans. Scheduling and serialization changes therefore have little visible unoverlapped budget in this capture.

Both observed modes contain 4,096 diagnostic runner steps: eight prefill steps plus 4,088 decode steps. Every request has 511 decode steps, of which 510 consume the already prepared successor; each makes one current decode call and 510 future calls. There are **no diagnostic graph-creation markers or first compiled-call invocations**. The full mode matches all 512 queue puts to 512 queue gets per request by actual server UUID and message ordinal.

## The two async calls have very different costs

The frozen runner first resubmits the current token/cache roots, constructs a successor, then calls `async_eval` on the successor roots. The observed already-ahead steps break down as follows:

| Full mode, already-ahead steps | Calls | Total wall time | Mean per call | p95 per call |
|---|---:|---:|---:|---:|
| First async call, targeted by `ahead_submit_once` | 4072 | 0.018523 s | 4.549 µs | 6.209 µs |
| Second async call, successor submission | 4072 | 75.821625 s | 18.620 ms | 18.820 ms |

The first call totals **0.02191%** of local service time; removing all its observed duration is an accounting thought experiment, not a speedup bound for changes that alter scheduling. The first decode in each request is distinct: it has a real current forward and its first async call averages about 21 ms. The final decode in each request has no async calls. Neither should be folded into the tiny repeated-current-call category.

The one-pass long-route guard candidate targets the wrapper around `compiled_callable`; the observed submission-minus-compiled interval is only **0.389686 s**, or **0.4609%**. That difference also contains unrelated dispatch/state handling, so it overstates the specific guard-loop opportunity.

## Next experiment

Prioritize correctness-preserving changes to actual model execution and amortized weight traffic: qualify and measure the layer-major/shared-weight verifier path and its end-to-end acceptance economics. In parallel, inspect why the **successor** async call occupies ~18.6 ms on steady steps: backend graph evaluation/encoding, dependency synchronization, resource pressure and actual GPU execution remain competing explanations. A narrow controlled backend probe can distinguish them; this Python profile cannot.

Keep `ahead_submit_once` and the one-pass guard change as small optional cleanup experiments. Their measured target intervals do not explain the remaining roughly 4.77% C1 formal gap and cannot substantiate a large speedup. HTTP work is a still weaker primary direction here. No new claim against oMLX or Lithos follows from this diagnostic.

## Reproduction and evidence

`analyze.py` reads the exact saved audits, immutable raw request groups, specs, closures and profiles; `readout.json` records their SHA256s, per-fixture measurements, interval unions and raw-UUID joins. `async-calls.py` refines the same accepted profile by call ordinal into `async-calls.json`. Both one-use scripts completed successfully; stdout/stderr are retained. Outputs use exclusive creation to preserve an existing result.

Span calculations use only server nanosecond clocks; client POST/DONE latency uses only client clocks. The frozen raw SSE parser independently recovers UUIDs from original wire chunks. Unassigned warm/background events and graph-creation markers are excluded; nested intervals are unioned before subtraction. Distribution p95 is nearest rank (with eight requests, the maximum); there are no confidence intervals or significance claims.

The driver's original `observed_diagnostic_complete=false` is intentionally unchanged: that driver always defers completion to independent auditing. The later independent audit records `diagnostic_complete=true`; neither receipt was rewritten. Source/runtime/model cleanup and restoration remain the responsibility of the already saved independent audits.
