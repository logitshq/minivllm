# E024 serving adapter: fixed same-schedule qualification PASS

The default-OFF, per-instance route-preserving decoder passed the original 37-request numerical workload. Independent numerical, recorded-resource and service-restoration audits all PASS. This is **not HTTP capture/replay or performance qualification** and adds no Pareto point.

The candidate retained short compile/fuse and GDN gates compile ON, norm reuse OFF; the reference remained eager with gates OFF. E028 identity refill was OFF. The loaded model had 64 layers and 128 state arrays. Weights stayed fixed during scheduled requests; parameter replacement was tested separately while the engines were idle.

| Actual case | Requests | Steps | Committed IDs | Consumed ahead | Partial retirements | Survivor refills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D-C1 | 8 | 4,096 | 4,096 | 4,080 | 0 | 0 |
| D-C8 | 16 | 1,027 | 8,192 | 1,014 | 7 | 7 |
| Lifecycle | 13 | 40 | 123 | 24 | 8 | 5 |
| Total | 37 | 5,163 | 12,411 | 5,118 | 15 | 12 |

The 24 D requests used the original normal-EOS cap512 contract. All reached their caps; no EOS retirement was observed. The 13 lifecycle requests used the original variable caps, including request-ID reuse. `discarded_prior` was zero in every case. Cancellation, sampling-key changes and outstanding-ahead invalidation are therefore outside this result.

## What the numerical audit establishes

The [independent numerical receipt](independent-numerical.json) reconstructs admissions, FCFS scheduling, slot reuse, retirement and semantic ahead pairing from the complete ledger. It accounts for 6,622,533 recorded original-dtype array comparisons: 10,315 full-vocabulary logits, 6,596,864 state arrays and 15,354 ahead-token arrays. All recorded bit-equality and both-arm finite checks pass. These are array-comparison counts, not scalar-element counts.

The separate idle parameter diagnostic made **42 direct calls**: six baselines, 18 changed-value calls and 18 restored-value calls. It covers `norm.weight`, `layers.0.linear_attn.in_proj_a.weight` and `.scales`, each at used1023/1024 and B1→B8→B1. Eighteen receipts confirm that changed values actually changed eager logits; restored outputs matched the original. It adds 17,068 array comparisons. This does not implement or qualify hot reload while a request or ahead state exists.

Actual compiled invocations were observed in both route families: D-C1 had 723 bound1023 and 3,365 bound1536 calls; D-C8 had 88 and 938. The parameter diagnostic had 21 of each, with 38 long-route fallbacks elsewhere. Calls are bound by case-step/direct label and independently inferred within-step order, not an explicit token key. Two callable families do not imply only two compiled machine graphs. The exercised physical KV capacity was 1536; wider capacity is not qualified here.

Successful tensor payloads were **not** saved. The offline audit checks the source-bound device comparison receipts, their shapes, hashes and complete scheduling coverage; it does not independently recompute GPU arithmetic or reread successful tensor bytes. The raw ledger is 1,628,001,846 bytes and the separate route ledger is 6,012,475 bytes.

## Resources and recovery

[Independent recorded-resource audit](independent-resources.json): 217 samples, pressure1, thermal0, swap growth0 MiB, Swapouts/Pageouts/Compressions growth0; Swapins +12 and Decompressions +6,053. Minimum recorded free pages × page size was 59,895,414,784 bytes. This sampled record does not prove absence of unrecognized runtimes or unrelated GUI GPU activity.

Owner86493 and direct child86527 were reaped. Original oMLX PID85613 was restored as PID92152, with the original model/config and fresh MTP/resident-PLE evidence; [independent restoration audit](restoration-audit.json) PASS. Warm KV and the exact hidden runtime state were not restored. The driver closed at 2026-10-17 23:50:46 UTC with stop/owner/recovery return codes all zero. Diagnostic duration includes loading, observers and verification and is not inference speed.

## Preserved evidence

- [Summary](summary.json), [manifest](manifest.json), [archive verification](archive-verification.json), [restored journal](restored-journal.json).
- [Evidence archive](evidence.tar.xz): 935 regular members, 3,850,220 compressed bytes; SHA256 `dc2458cfa410121ac9c82e4a9eee988f9ab6f3265da91e28de8a2e789e8bbb79`.
- Owner plan SHA `5161400cdfcd2b91610383f0776fc8483003d7b9bc3962c3a6728948004b5298`; child plan SHA `e85f92aa1402025a669de4c2426460c1eced29f0145f128480f1fa9e66349c56`.

The archive contains raw owner/journal/execution/analysis/preparation trees, preregistration and exact registered sources. Member bytes, source files and tree membership were rechecked. Extract into a fresh directory; only relative regular-file members under `raw/`, `preregistration/`, `registered-source/` and `archive-source/` are expected. The manifest retains the original absolute paths. Replaying source-bound audits requires those registered source and plan references to remain available; the archive does not contain model weights or successful tensor payloads. This README is an external index added after archive sealing, not a tar member.

Next: qualify actual C1/C8 route-ON HTTP arrival histories through capture → strict server reap → independent numerical replay and route audit, then measure observer-OFF throughput versus complete POST→DONE latency. No combined E028, larger-batch, unseen arrival-history or new performance claim follows from this result.
