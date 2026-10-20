# E048: local eager GDN timing, all groups retained

![All 60 group ratios and 12 medians](chart-layout2/gdn-local-timing-ratios.png)

All 12 case medians favor the candidate: **B2 1.071695–1.089616×**, **B8 1.047935–1.066748×** native/candidate. B8 early layer32 group0 is **0.880700×** and remains in the figure, raw data and median calculation. This is a warm-resident local eager GDN layer screen on recorded real-model activations. It is not a compiled model rollout, HTTP C8 result, or engine Pareto point; whole-model cache behavior and end-to-end speed remain unmeasured here.

Each of 12 cases has five groups of ten native/candidate pairs in alternating order, after five warmups per arm: **1,200 measured arm durations +120 warmups**, with all 60 group ratios retained. A group ratio is median(native ns)/median(candidate ns); the reported case ratio is the median of its five group ratios. Spread is observed variation, not a confidence interval. Fresh forward work, output/conv/S materialization and synchronization are inside timing; independent seed reset is outside. Outputs from every measured sample are not separately saved. The qualified original-dtype output/state tensors and producer seed-preservation checks remain distinct evidence.

| Case | Median group ratio | Minimum group | Maximum group |
|---|---:|---:|---:|
| b2-early-layer0 | 1.075262× | 1.055930× | 1.149845× |
| b2-early-layer32 | 1.071907× | 1.056612× | 1.088256× |
| b2-early-layer62 | 1.079556× | 1.076615× | 1.089962× |
| b2-late-layer0 | 1.077415× | 1.066522× | 1.078024× |
| b2-late-layer32 | 1.089616× | 1.062811× | 1.091129× |
| b2-late-layer62 | 1.071695× | 1.067452× | 1.088272× |
| b8-early-layer0 | 1.064586× | 1.051890× | 1.120772× |
| b8-early-layer32 | 1.047935× | 0.880700× | 1.082116× |
| b8-early-layer62 | 1.056157× | 1.047893× | 1.063538× |
| b8-late-layer0 | 1.053764× | 1.030714× | 1.072833× |
| b8-late-layer32 | 1.066748× | 1.056849× | 1.113766× |
| b8-late-layer62 | 1.058262× | 1.046807× | 1.073278× |

Actual owner `e7e79b`/session65941 → `af7b46` exited 0, `PASS_CAPTURE_AND_RESTORE`, 15.352014792 s. Independent original-byte/timing audit `003e46`/session91399 → `0bc50b` exited 0: 48 raw files, 72 output/state byte comparisons, all 1,200 timings, 120 warmups, 60 groups, 672 expected/reported candidate calls and zero reported fallback. Actual per-call dispatch remains pinned producer evidence. Parameter hash receipts are checked; no parameter payloads are saved by this screen or reread by the auditor. All original full-model/HTTP/performance/frontier qualification flags stay false.

Independent saved-resource review `53c011` passed: six physical +six metadata observations, max gap 3.289793 s (baseline-to-first 2.105915 s), maximum system footprint 188,859,121,664 B, child current footprint 16,033,230,656 B and recorded historical peak 16,034,672,448 B. Pressure 1, thermal 0, AC, zero swap growth, Swapouts unchanged at 807478. Final observation/ack/wait/reap and saved original oMLX PID52818 restoration/three health responses were checked. These are sampled accounting observations and saved restoration facts, not hard allocation limits or fresh live checks. Full original process-inventory text and repeated identity history were not retained.

`evidence.tar.gz` retains original execution/owner/lease metadata, every raw timing/warmup row, numerical descriptors, source/preregistration, actual numerical-audit execution, screen CPU6 / owner CPU13 / audit CPU5 receipts, both figure attempts and independent resource/visual reviews. The first figure's left-axis clipping is preserved as `FAIL_LAYOUT`; `chart-layout2` passed actual PNG inspection by both this agent and root. Original `RENDERED_NOT_YET_VISUALLY_REVIEWED` receipts remain unchanged, with reviews in separate files. No narrow-screen/color-vision simulator check was run; faceting and open-dot/dark-median encodings do not require color to distinguish groups.

The **24 new raw tensor files (386,095,428 B)** remain local, alongside the 24 original E046 input/reference files. `raw-file-references.json` records each absolute path, observed size and SHA bound to the completed independent raw-byte audit. This archiver did not reread or copy tensor payloads. All original pending flags are preserved.

Archive execution `5b572d` passed: **307 members**, 3,061,362 uncompressed metadata bytes, gzip 927,168 B; every member was reread and size/SHA checked against its original source. Archive SHA-256 `1f05f1729e0c48042e0a0cdace836aa6e8a66cdcb48fbd0159cdd27e9cccbe03`; manifest SHA `6c7781ba6f76df387c0927384b7e1133d3f5841a29c27d5a25c823ed8a2a2da9`. No requested directory was missing. Exactly 24 new tensor payloads were excluded by design. See `summary.json`, `manifest.json`, `references.json` and `execution-provenance.json` for precise evidence boundaries.

Reproduce the figure from the retained audited JSON using the command in `render-v2-command.json` and a fresh output directory. The static PNG/SVG and full `chartdata.json` preserve all plotted values. Archive integrity does not broaden the experiment's numerical or performance scope.
