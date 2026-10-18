# E029 actual real-activation canary: PASS

The default-OFF B1 GDN prework candidate passed the registered local canary and independent saved-array audit on 2026-10-17. Resource and service-restoration audits separately passed. This establishes local real-input correctness only; it adds no throughput/latency point.

| Actual coverage | Result |
| --- | --- |
| Candidate calls / continuous state chains | 18 / 6 |
| Full output, convolution-cache and recurrent-state pairs | 54 exact |
| Continued-state pairs | 24 exact |
| Reference full-vocabulary greedy checks | 6 |
| Saved raw files / bytes | 60 / 176,734,788 |
| Device comparison arrays | 162 |
| Candidate fallback calls | 0 |

The first, middle and last GDN layers use real activations from the shortest and longest fixed D512 prompts. Each candidate chain carries its own state for three steps; subsequent activations still come from the reference model. This is not candidate whole-model rollout. All 14 selected-layer parameters have unchanged original-byte hash/shape/dtype/finite receipts; parameter payloads are not retained for offline recomputation. E024 graph and E028 refill are not combined with this candidate.

Seven resource samples passed: pressure 1, thermal code 0, no swap growth, Swapouts/Pageouts/Compressions unchanged. Minimum Pages-free bytes were 117,425,848,320; that is not available memory or physical footprint. Child 29163 and owner 29126 were reaped. Original oMLX 27913 was restored as 29328 with the same model/config and fresh MTP/resident PLE verified; warm KV and exact prior hidden state were not restored.

The archive retains 1,040 members, including all raw arrays, plans, frozen sources, execution records and independent audits. Its 185,486,960 logical bytes compress to four parts totaling 166,005,544 bytes. Full stream SHA256: `46ba33137f1729cb42d2ca673211e022c73b81b563ce1828d6057693e0b2d33f`. See [archive instructions](README.md), [summary](summary.json) and [verification](archive-verification.json). This result note was added after archive creation and is not a member of that sealed stream.

Next: run the unchanged 37-request full-model scheduling/logit/cache/ahead comparison with this adapter installed before observation. Only a passing independent result can advance it to actual HTTP qualification and observer-OFF timing.
