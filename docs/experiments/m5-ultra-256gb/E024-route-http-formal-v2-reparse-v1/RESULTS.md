# E024: paired HTTP timing after an explicit offline parser correction

Twelve fresh native servers completed the predeclared OFF/ON order, with eight warm and 64 formal requests per server. All 864 raw requests, counts, full POST-to-DONE timing, ownership, resource records and restoration now pass independent audit. The measured engine sources, weights, runtime and raw trials were not changed or rerun for this correction. Original observer-OFF timings remain intact.

## Result

Each cell below is the median of three process metrics, not a pooled request percentile. Full min/max observations and all twelve points are in `independent-formal.json`.

| Arm | HTTP C | Fresh processes | Committed tok/s | POST→DONE p50 s | POST→DONE p95 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| OFF | 1 | 3 | 45.405789 | 11.270521 | 11.319748 |
| ON | 1 | 3 | 45.835028 | 11.165923 | 11.215184 |
| OFF | 8 | 3 | 167.536857 | 24.418582 | 25.891997 |
| ON | 8 | 3 | 170.619844 | 23.976082 | 25.531040 |

All six adjacent process comparisons have higher committed throughput and lower p50/p95 latency with the graph enabled. Each pair shares fixed fixtures; realized HTTP arrival schedules need not be identical. These three repeats per concurrency do not provide a confidence interval or a claim about all future workloads.

| Repeat / C | Actual order | Throughput change | p50 change | p95 change |
| --- | --- | ---: | ---: | ---: |
| r1 / C1 | OFF→ON | +1.621% | -1.588% | -1.736% |
| r1 / C8 | ON→OFF | +1.777% | -1.699% | -1.799% |
| r2 / C8 | OFF→ON | +1.799% | -1.812% | -1.394% |
| r2 / C1 | ON→OFF | +0.972% | -0.930% | -0.924% |
| r3 / C1 | OFF→ON | +0.847% | -0.806% | -0.908% |
| r3 / C8 | ON→OFF | +3.807% | -3.813% | -1.771% |

Against the previously audited, separately timed oMLX D512 process medians, native ON C8 has 1.34% more throughput, 1.19% lower p50 and 2.29% lower p95. Native ON C1 still has 9.74% less throughput than oMLX. Native ON is ahead of the recorded Lithos C1/C8 medians; Lithos C8 uses its official FIFO execution, so this is not a GPU batch-8 kernel comparison. No global-fastest, maximum-capacity or universal Pareto victory follows.

## Why the first audit failed

The original frozen auditor applied its generic canonical-regular-file rule to the exact registered Hugging Face tokenizer snapshot path. This legitimate snapshot leaf is a symlink to a cache blob. The auditor rejected it before opening HTTP wire records. [The complete original FAIL](../E024-route-http-formal-v2/RESULTS.md) remains unchanged; archive SHA `9174c276c76405651cc351f8e248e98b68aba0c257831a8bc5706e4b2fb52b7b`.

The separately versioned `audit_formal_v2.py` changes only tokenizer reference handling and explicit provenance. It requires the registered snapshot path and model-manifest hash, matching original pre/post target hash/inode/size/mtime, current regular target and actual payload hash, then rechecks the target after the audit. The generic canonical-file gate and every original count, timing, resource and restoration function are unchanged. Historical symlink spelling was not retained, so the new receipt reports the current resolved path without inventing historical path evidence.

Eight targeted CPU regressions passed, including the actual tokenizer reference, payload tampering with size/mtime preserved, a same-byte target with a different inode, wrong registered paths/manifests and unchanged generic symlink rejection. A source AST comparison checks the narrow entry change. Root and an independent agent reviewed the correction. The v12 automatic renderer correctly declined the original FAIL; the later v13 display carries this post-run correction explicitly and preserves v12.

## Resources, recovery and retained evidence

The original resource gates pass for 1,267 samples: pressure 1, zero swap growth, zero Swapouts/Pageouts/Compressions growth; Swapins +82 and Decompressions +33,248. Retained process records cannot exclude all GUI or unknown GPU activity. These resource observations do not establish an allocation cap.

All twelve direct servers and the owner were reaped. Original oMLX PID 43565 recovered as 73594. Independent restoration passed with original model/configuration and fresh MTP/resident PLE observations. Warm KV and exact hidden runtime state were not restored.

The new archive retains both audit attempts, the eight regression tests and their actual complete output, original registration and all raw trial/resource/restoration evidence: 3,843 members, 754,330,855 logical bytes, 38,203,144 compressed bytes. SHA `63e211a8eecaccd7f41e7b690e33fc01964d164c7c45ff5f82d4b3fc2575f7db`. All safe extracted member hashes/sizes, original bytes and tree membership were verified. Model shards and runtime binaries are not bundled or independently rehashed here. This explanatory file is outside the sealed archive.

The graph remains an explicit default-OFF module. Prior captured-history numerical qualification does not prove every new arrival schedule; observer-OFF timing does not observe compiled-call counts. GDN prework, GLM load, E028 composition, cancellation/EOS paths not observed in qualification and higher concurrency remain separate gates.
