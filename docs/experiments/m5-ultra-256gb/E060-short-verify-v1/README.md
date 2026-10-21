# E060: short verification after E059's mixed result

**Closed: GPU correctness and independent raw audit PASS; no overall performance win.** Run from d12dbdd, M5 Ultra / MLX0.32.3 / original frozen Qwen3.8-27B NVFP4.

| Prompt | native | fixed2 | fixed4 | fixed8 |
|---|---:|---:|---:|---:|
| Math | 733.422 ms | 706.986 ms | 662.175 ms | 822.345 ms |
| Chinese | 734.251 ms | 794.596 ms | 929.213 ms | 1407.145 ms |

Three samples/arm/prompt, medians above. Full generation loop includes draft8, actual accept/reject, target feature capture, selected-state install and synchronization. Prefill and HTTP are excluded; denominator is31 new decoded tokens, excluding the initial prefill token. Native uses the existing decode_model runner through a host-driven loop, not the serving scheduler/lookahead.

Combined two-context throughput relative to native: fixed2 0.979022×, fixed4 0.924252×, fixed8 0.658720×. These are exploratory synthetic-context results, not a production adoption decision. T2 reduced Chinese latency by14.49% vs T4, but remains8.22% slower than native. Math T4 is1.107596× native; there is still no universal winning width.

![All local generation samples](generation-cost.png)

Independent audit:16 raw final files,1040 tensor pairs, all24 timings and token/round/EOS/cap accounting PASS. 6 unique seed files rehashed;1349 target and175 drafter parameter receipts unchanged. Final-state/raw qualification covers the dedicated untimed trajectories, not full per-round raw state for every timed run. All timed outputs match the qualified IDs. CPU: 30 root +13 independent audit checks pass.

Resource observations:27 physical+metadata samples, max system187.26GiB / child physical peak22.44GiB, zero swap/swapout growth. Bounded child reaped and same oMLX PID52818 restored. Final live /health confirms original model remains loaded. MLX active bytes at in-function cleanup were209,409,032 (residual allocations before function return; retained local references are a likely cause), cache0; process exit/reap releases the remainder. No zero-residual-memory claim.

[Independent audit](independent-audit.json) · [all timings/IDs](complete.json) · [resource review](resource-review.json) · [source/receipt manifest](device-evidence-manifest.json). Raw multi-GiB tensors remain under `.artifacts` and are indexed by `pairs.json`; no weights or credentials are committed.

## Original pre-run record


GPU not run at source checkpoint. E059 found fixed4 faster on math but slower on Chinese, while fixed8 was worse on both. Test T2 to reduce rejected-suffix work; this is an exploratory follow-up on the same two synthetic1024-token contexts, not a new held-out benchmark.

Frozen draft block8 is unchanged; verify one anchor+one proposal at T2. Candidate arithmetic already admitted T2, but E058/E059 did not qualify it on device. E060 must first compare real generated IDs, last logits, every target feature and all128 final states, plus a forced-first-rejection trajectory alternating T2/T4/T8. EOS/cap selection now covers widths2/4/8 in3072 CPU cases.

Four timed arms native/fixed2/fixed4/fixed8, one warmup each and3groups×2contexts×4arms=24 samples. No calibration/adaptive policy. Same M1 host-driven native runner; no HTTP/prefill timing/serving lookahead. Fresh outputs and inherited bounded pause/restore owner. A win here is not an oMLX/Lithos Pareto win.

Pre-run failure preserved: first CPU run passed29/30; the expanded enumeration correctly produced3072 cases but the copied assertion still expected2928. Only that count/comment corrected, not selection behavior.
