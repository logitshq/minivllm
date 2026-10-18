# E029: GDN-only paired HTTP performance

Completed 2026-10-18 01:20 PDT. The twelve fresh server processes, 96 warm and 768 formal requests, independent raw HTTP/count/timing/resource audit, service recovery and archive verification all passed. All actual formal requests produced 512 committed tokens under normal EOS/cap512 semantics. The qualifier did not observe terminal EOS.

Medians of three independent process metrics per configuration/concurrency; request percentiles are not pooled across processes:

| Concurrency | GDN switch | Throughput (tok/s) | HTTP p50 (s) | HTTP p95 (s) |
|---|---|---:|---:|---:|
| C1 | off | 45.372212 | 11.277394 | 11.329765 |
| C1 | on | 47.859981 | 10.694225 | 10.741129 |
| C8 | off | 167.697252 | 24.402386 | 25.862208 |
| C8 | on | 167.419757 | 24.431340 | 26.000035 |

C1 paired throughput improved by 5.5668%, 5.5741% and 5.1914%; paired p50 fell by 5.2038%, 5.2483% and 4.9497%. Against the previously measured oMLX C1 median, GDN-only throughput remains 5.7532% lower and p50 remains 6.0945% higher. Compared with Lithos C1, throughput is 6.5682% higher and p50 is 6.2123% lower. These cross-engine comparisons reuse separately measured process groups, not newly randomized engine pairs.

C8 has no established gain: paired throughput changes are -0.2031%, +0.0923%, -0.1040%; p95 worsens in all three pairs, including +2.5712% in repeat2. Prior recorded C8 qualification used the original multirow fallback with zero candidate dispatches. Formal observers are disabled, so actual formal candidate call counts were not measured. Keep this GDN-only configuration separate from the faster E024-only C8 result (170.619844 tok/s).

Adding all twelve new observations preserves 38 process points and 14 separate configuration/concurrency groups. The discrete joint frontiers for both p50 and p95 remain oMLX C1 plus minivllm E024-only C8. The minivllm frontier can choose GDN-only C1 and E024-only C8 as separate configurations; it does not establish a combined GDN+E024 result. C2-C7 remain unmeasured.

1,252 raw resource samples passed the fixed gates with zero swap-size growth. All twelve experiment children were reaped. The original oMLX service PID90511 was restored as27245, with independent persisted-visible recovery PASS; warm KV and exact hidden runtime state were not restored.

The first independent audit and archive both passed. Every archive member was extracted and checked against original SHA/size, then the originals and tree membership were checked again. Full process ranges, individual paired comparisons, source/runtime bindings and raw file pins are in `independent-formal.json`; all original wire, timing, errors, cleanup and restoration evidence is retained in the archive. These observed process ranges are not confidence intervals.
