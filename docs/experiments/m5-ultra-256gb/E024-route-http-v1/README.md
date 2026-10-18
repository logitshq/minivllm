# E024: actual route-ON HTTP qualification closes

2026-10-17. Four direct phases completed after preregistration4ec464d: C1 capture/replay, then C8 capture/replay. All four independent audits (C1 numerical, C8 numerical, resources/cycle, restoration) PASS. This qualifies the recorded HTTP histories; no observer-OFF speed result is awarded.

| HTTP C | Requests | Replay steps | Committed IDs | Exact array receipts | Actual graph calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 4096 | 4096 | 5251008 | 4088 |
| 8 | 16 | 1028 | 8192 | 1322478 | 1027 |

Same eight D512 fixtures, original NVFP4 revision, normal greedy EOS/cap512, gatesON/normOFF and E028OFF.24 requests produce12,288 committed IDs; all reach cap512, observedEOS0. Independent auditors reconstruct the actual recorded scheduling, validate raw HTTP count/text/IDs/fully-drained histories and source bindings, and check6,573,486 original-dtype device array-comparison receipts. Full successful tensor payloads are not retained, so this is not offline recomputation of GPU arithmetic. Both recorded graphs and replay route events match; physical KV capacity1536, bounds1023/1536.

C1 observed route hits: `{"HTTP-C1": {"1023": 723, "1536": 3365}}`; ahead/retirement: `{"HTTP-C1": {"consumed": 4080, "discarded_prior": 0, "partial_retirement": 0, "refills": 0}}`.
C8 observed route hits: `{"HTTP-C8": {"1023": 89, "1536": 938}}`; ahead/retirement: `{"HTTP-C8": {"consumed": 1014, "discarded_prior": 0, "partial_retirement": 7, "refills": 7}}`.

C8 includes seven partial retirements and seven survivor refills. Consumed ahead is covered; invalidation/discarded-ahead0, EOS0, cancellation and changed sampling-key paths remain unobserved. Original37-request/42-parameter-call serving prerequisites remain separately bound; this HTTP cycle does not rerun them or authorize E028 combinations, arbitrary future arrivals or larger batches.

The250 raw resource samples pass the original pressure/AC/thermal/sustained-swap gates: pressure1, swap growth0MiB, Swapouts/Pageouts/Compressions0, Swapins+12/Decompressions+5335. All four direct children and owner were reaped. Original oMLX19019 restored as27913, with original visible model/config plus fresh MTP/residentPLE evidence. Independent restorationPASS; warmKV/exact hidden runtime state not restored. Sampling cannot rule out unobserved physical peaks or every possible GUI/unknown GPU process.

## Evidence

Archive:1333 members,1,844,688,963 logical bytes,4,544,516 compressed bytes, SHA `aec7947a4afd39d2b17c560ef45014b7d4b2b2181732d1bff44292437c09f717`. All member hashes, original source bytes and directory membership checked;135 registered source pins remain unchanged. Original raw ledgers retained in full. Packaging verifies retention and does not grant qualification.

- [C1 numerical](c1-independent-numerical.json) / [C8 numerical](c8-independent-numerical.json)
- [Cycle/resources](independent-cycle.json) / [Restoration](restoration-audit.json)
- [Archive verification](archive-verification.json) / [Manifest](manifest.json) / [Audit execution](audit-execution-summary.json)

Next is paired observer-OFF native OFF/routeON timing, three fresh processes per arm/C, same raw count and POST-to-DONE definitions. The v10 three-engine chart remains unchanged until those actual formal results exist.
