# E024 long decode graph: first numerical point rejected

Actual cold cycle completed 2026-10-17. The unchanged, default-OFF graph candidate
was run against the fixed Qwen3.8-27B NVFP4 model with gates/norm/packed fusion/core
compile OFF. Its first B1/used257 comparison invoked the compiled closure with
attention bound 1536 and failed the original exact BF16 logits criterion.
No tolerance was changed and no performance point is awarded.

The independently reread original safetensors contain finite BF16 logits of
shape `[1,248320]` in both arms. **228,516 elements differ**. First differing
element `[0,1]`: eager 2.96875, candidate 2.859375. Maximum absolute difference
is 0.28125 at `[0,1759]`. The raw tensor payload SHA256 is
`9c7070379c6a3f1cf85d9feffc9aba88c48ef76a5ed347b5fdbab499bf65d320`.

Coverage is **0/32 passing matrix points**, no parameter-replacement decode
checks and no ahead lifecycle. The failure precedes argmax and post-graph cache
comparison, and the failed child did not execute the final source/model recheck.
Earlier 2,381 recorded weight/prefill array checks passed; their successful tensor
payloads were not retained, so the independent audit checks those ledger entries,
not a new byte comparison of the successful prefix. Root read and reran the raw
failure auditor; its result is byte-identical to the independent result.

This rejects the candidate; it does not establish that native eager is wrong or
identify the numerical cause. E024 changed both compilation and attention slice
length. A [separate four-arm diagnostic design](../../../../bench/reviews/2026-10-17-e024-dispatch-isolation.md)
will isolate those variables. Matching MLX tagged source predicts a reduction-route
change at K>=1024 on the recorded g17d device; causality is not yet measured.
The original candidate, plan and failed evidence remain unchanged.

The numerical child ran for 10.248649 seconds (diagnostic duration, not inference
performance), and the owner ran for 11.195151 seconds. All five observed resource
samples passed: pressure 1, zero swap growth and zero Swapouts/Swapins/Pageouts.
Only the owned experiment child was observed among known inference processes;
these snapshots cannot exclude an unknown runtime or GUI GPU activity.

Original oMLX PID 98702 was stopped at 16:20:23 UTC. Child 700 and owner 665 were
reaped/absent before restoration. The original service recovered at 16:21:16 UTC
as **PID 832**, with its original Qwen3.8-Flash-Next-oQ6e-mtp model and fresh
MTP/resident PLE logs. Independent recovery audit PASS checks 19 physical samples
and the 11,299-byte fresh log window; maximum recorded system footprint during
recovery was 176,108,027,904 bytes. Warm KV and exact hidden runtime state are
not restored. The owner numerical failure and successful service recovery are
separate outcomes; the enclosing executor returned 1 for the numerical failure.

Archive: **637 members**, 966,084 compressed bytes, SHA256
`f61e7e5ccdc044b8744a0edf263e6497b0c2955c0c79400b46f4e09260b684ee`.
Every decompressed member size/hash and every original file hash/mtime were
verified, and source tree membership was rechecked. Raw owner, cold journal,
execution, CPU preparation, analysis, preregistration and audit sources are saved.

- [Independent numerical failure audit](independent-numerical-failure.json)
- [Restoration audit](restoration-audit.json) and [restored journal](restored-journal.json)
- [Manifest](manifest.json) and [archive verification](archive-verification.json)
- [Frozen preregistration](../../../../bench/m5_ultra_256gb/long_decode_graph/prereg-cold-v1/README.md)
- [Current D512 measured chart](../../../optimization-guide/progress/2026-10-17-v6/README.md)
