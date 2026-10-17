# Independent v3/v4 throughput–complete-request-latency audit

2026-10-17; CPU-only read-only review. This document does not modify historical chart data, source snapshots, acceptance gates or graphs. No server, socket, model, MLX runtime, download or GPU run was started. Existing PNGs were visually inspected; no figures were rendered.

The metric reconstruction passes. **v3 is a historical short-request concurrency scan, not evidence of current universal dominance; v4 is a separate same-round C1 experiment and must not replace v3's C1 while keeping its other points.** For a new presentation, use separate historical C2–64 and current C1 panels; exclude historical external C1 from new superiority claims for the resource-provenance reason below.

## Independent checks and metric definition

- Read both READMEs, summaries, chartdata files and frozen renderer sources; verified every snapshot entry against its manifest: **v3 51 files / 46,205,055 bytes**, **v4 153 files / 4,305,787 bytes**. This includes small recorded source/raw files, not model payloads.
- Independently recomputed **v3 all 48 processes / 2,304 formal requests**, plus each configuration median and its Pareto membership, from archived request timestamps and actual usage; values agree with chartdata to 1e-12 relative tolerance.
- Independently recomputed **v4 all 15 processes / 45 formal requests** from the individual request JSONs; all process latency/throughput medians agree to 1e-12. Reviewed E020's existing independent SSE/route/resource/recovery audit, which reports `PASS_RECONCILED` and `COMPLETE_VALID_CONTRACT` for all 90 warm/formal requests. This review did not rerun its full compressed archive audit.
- Latency = client POST/request start to SSE DONE, excluding cleanup after DONE; it includes queueing, prefill and final output delivery. v3 group latency is request p50, then median over three groups, then median over three fresh processes. Throughput = sum(actual completion tokens) / (last DONE minus first POST) for each group, aggregated with the same hierarchy. It is not the inverse of a decode-only rate or TTFT.
- v4 uses actual completion usage / POST-to-DONE per C1 request. At fixed 32 output tokens, throughput and latency are reciprocal rescalings; this slice does not establish a concurrency tradeoff curve.

## Correctness and comparability scope

All seven native v3 points use the same final E010 source hashes and flags. Its recorded qualification contains 448 layer/cache checks, 22 boundary cases per implementation, and two 16-step fixed-schedule lifecycle traces; logits/cache/output exactness applies to identical inputs and schedule. HTTP raw IDs equal visible roundtrip IDs and count/finish contracts; free-arrival schedules legitimately yield multiple continuations. Formal native streams per C are 1/1/1/1/2/2/3 for C1/2/4/8/16/32/64. C16/C64 remain correctness-eligible; their inferior performance, not a numerical failure, removes them from the median frontier.

The nine external v3 configurations have archived count/visible-text contracts and recorded identities; those HTTP endpoints did not expose live raw token IDs. v3's renderer checks lengths/usage/roundtrip structure rather than independently re-running a semantic correctness oracle. Cross-framework identical tokens, equal downstream quality and every live HTTP schedule's full numeric equality are **not** established. The stronger v4 contract checks external visible IDs against prior same-engine raw-ID qualification; Lithos and native/oMLX still have different qualified continuations. This is a stated performance comparison under each engine's output contract, not a shared-oracle quality-equivalence proof.

Model/workload identity is consistent with the recorded plans: M5 Ultra 256 GiB, `mlx-community/Qwen3.8-27B-nvfp4@5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`, the same 19-token prompt, greedy, normal EOS, cap32, target-only and no prefix-cache hits. E010's native source is final core b33b282; external oMLX is official 0.7.1.dev1 wheel (VCS identity unknown), Lithos upstream5485ad6 plus local M5 profile2236580. Their differing kernels, batching and transport are part of the compared systems. Native uses a 2 GiB allocator pool; external allocator policies remain official defaults. C is outstanding HTTP concurrency, not a claim that actual GPU batch always equals C. Each v3 group is a finite C-request burst; it does not measure a long sustained mixed-request service.

The external points were measured earlier; E010 native was not contemporaneously paired with them. Short fixed length, one prompt, warm operation, different continuation streams and historical resource checks limit the conclusion. Three process replicates and their spread are not confidence intervals or proof of significance. No GLM, long-context, MTP/DSpark concurrency curve or untested continuous-curve interpolation is supported.

## Newly identified historical C1 resource limitation

Parent's physical preflight records establish PID74384 start identity **Fri Oct16 08:01:58 local (America/Los_Angeles), i.e. 2026-10-16 15:01:58 UTC**. The v3 external C1 resource records occur **15:03:12.541121–15:05:18.369638 UTC**, after that start. These use the old `known benchmark command matches only` detector and report zero known inference processes before launch. It did not inventory the later-recognized `omlx-server` process name. The evidence therefore does not establish that PID74384 was absent, suspended or GPU-idle during those external C1 measurements. It also does not establish that it was actively inferring: do not assert causal contamination or attribute the performance difference to it.

Concrete archived samples (the resource arrays are embedded in the result JSONs whose hashes are below):

| Result | Resource entry | UTC timestamp | Old reported known count |
| --- | --- | --- | ---: |
| pareto-mode-frontier-v1/C1-pair1-omlx/result.json | resources[0], before-launch | 2026-10-16T15:03:12.541121+00:00 | 0 |
| pareto-mode-frontier-v1/C1-pair3-lithos/result.json | resources[0], before-launch | 2026-10-16T15:05:05.925924+00:00 | 0 |
| pareto-batch-frontier-v1/C64-pair3-omlx/result.json | resources[-1], after-unload | 2026-10-16T14:59:06.383925+00:00 | 0 |

All v3 external C2–64 resource timestamps span **14:34:32.449459–14:59:06.383925 UTC**, before this PID started. Thus this specific PID overlap does not apply to those points; it does not retroactively turn the old matcher into a complete GPU-exclusivity proof. Native final E010 used the improved inventory and authorized stop/resume leases. Preserve the original measurements; historical external C1 should not support a new strict-fairness all-engines-win conclusion.

## Verified v3 numerical values

Units: complete-request p50 latency ms and total output tok/s/Mac; each is the documented median hierarchy. Frontier here means among the historical plotted medians, subject to the caveats above.

| C | Native ms | Native tok/s | oMLX balanced ms | oMLX balanced tok/s | Native median on joint frontier |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 622.147709 | 51.434731 | 706.494875 | 45.294030 | yes |
| 2 | 781.340396 | 80.805636 | 868.482271 | 72.306662 | yes |
| 4 | 999.266250 | 126.674834 | 1196.767791 | 104.171573 | yes |
| 8 | 1662.854146 | 153.890592 | 2157.480375 | 113.523080 | yes |
| 16 | 3464.325250 | 147.720561 | 4568.812041 | 105.080726 | no |
| 32 | 5286.722729 | 190.444616 | 9262.126791 | 99.512615 | yes |
| 64 | 10718.856625 | 189.827696 | 19126.449604 | 92.463892 | no |

Additional historical C1: oMLX aggressive **719.467666 ms / 44.477329 tok/s**; Lithos **675.674792 ms / 47.360062 tok/s**. Lithos has only C1 support in this measured official endpoint; no invented concurrent curve or implicit zero-throughput point. The joint median frontier is native **C1/2/4/8/32**; C16 is dominated by C8, C64 by C32. The tiny C32/C64 throughput difference is not a significance claim.

## Verified v4 values and separation

| Arm | E2E ms | Output tok/s | TTFT ms (auxiliary) |
| --- | ---: | ---: | ---: |
| native-none | 716.098458 | 44.686593 | 135.450167 |
| omlx-none | 679.323791 | 47.105667 | 113.478208 |
| omlx-mtp | 752.063000 | 42.549627 | 99.762625 |
| lithos-none | 812.920167 | 39.364259 | 278.709000 |
| lithos-dspark | 741.955458 | 43.129274 | 251.650000 |

The same-round C1 best median is **oMLX target-only**, not native. MTP's earlier first text does not offset its worse complete-request latency/throughput on the main axes. All three paired results remain; within-engine throughput geometric means are **0.905527× oMLX MTP/none** and **1.085991× Lithos DSpark/none**.

The new native C1 anchor differs from E010 (716.098458 ms /44.686593 vs622.147709 ms /51.434731). HTTP adapters and their instrumentation/payload differ; no controlled causal explanation is established by these two plots. Do not splice the faster historical native C1 or slower historical external C1 into v4, and do not splice v4 C1 into v3.

Display artifacts, visually checked for readable axes/labels and retained inferior points:

- [v3 historical full scan](../../docs/optimization-guide/progress/2026-10-17-v3/throughput-latency-pareto.png)
- [v4 same-round C1 slice](../../docs/optimization-guide/progress/2026-10-17-v4/e020-throughput-latency.png)

## Exact reviewed file hashes

These SHA-256 values bind this review to the original evidence. The preflight files are parent-owned local receipts; the review read only identity/measurement metadata, no credentials. The two chart manifests bind all 204 source-snapshot files; those files are not duplicated here.

```text
5aa2d3527d45d704715c0924c8bdc09aa052516bb3608c73cfd4961da354a305  docs/optimization-guide/progress/2026-10-17-v3/README.md
09e0dca520c97b330c8f25c70a9ded9c13a67f67f8401e8922dd4fbfccc89f00  docs/optimization-guide/progress/2026-10-17-v3/summary.json
6b6664e0cdf9dc5108779f96b093072e6079f082b4be969facdd14712d6ba43a  docs/optimization-guide/progress/2026-10-17-v3/chartdata.json
1e633f637af827b0dfa7e9f92c79dcc69743cec0ba2809bfb61787bdb8e4e43c  docs/optimization-guide/progress/2026-10-17-v3/source-manifest.json
0c0e618951ee4b04a178b70c29766e7387410645bb2c4cdc7d1692ba8cf86c7d  docs/optimization-guide/progress/2026-10-17-v4/README.md
da3a11366b42c2b00936bc06ce6d44fcb1cacaecc65c88ac50e262fe6a7f8af8  docs/optimization-guide/progress/2026-10-17-v4/summary.json
f5fab1581521f5eac6b3ab13870846abc8f9324dac0529fd8e41931d148795be  docs/optimization-guide/progress/2026-10-17-v4/chartdata.json
7f39ad30ec3fc260aca69a69f8f22f5532732ebd8c59472a13b3c04cf3e94840  docs/optimization-guide/progress/2026-10-17-v4/source-manifest.json
256501986be1bdcc6e74146f5b0056d1955cf7be828170b7b9b45b350a54b408  docs/optimization-guide/progress/2026-10-17-v3/sources/docs/experiments/m5-ultra-256gb/prefill-nax-v2/audit.json
4ad79b88d80f2a913a2e9b2b166940c3b826506bcd872f9a7f6a0bf269942b08  docs/optimization-guide/progress/2026-10-17-v3/sources/docs/experiments/m5-ultra-256gb/prefill-nax-v2/frontier.json
33c9398f045405b493cc1e7cc5b68999c4cf9c999d98da3f710ae56d7dae680e  docs/optimization-guide/progress/2026-10-17-v3/sources/bench/m5_ultra_256gb/progress/latency_pareto.py
253674f814ef2ba138857ae07454a4302fc5124055cd19e1ecd822ed6d724fa8  docs/optimization-guide/progress/2026-10-17-v4/sources/bench/m5_ultra_256gb/progress/external_spec.py
19a18db40341914bb60c0daa937eb88b20bf7aa6f4e283cc53d1507a1c6179ed  docs/experiments/m5-ultra-256gb/E020-external-spec-http-v1/summary.json
4bbf533e6e16a627065f15f190a0a2e9a0ac41b9a6081c945308a691b7300b1b  .artifacts/m5-ultra-256gb/glm-flash-physical-preflight-v1/raw.json
dcb3ebdef6410de9a313645db3b2ef5e1ce53aa9c91ee361cf95ac896755e6b9  .artifacts/m5-ultra-256gb/glm-flash-physical-preflight-v1/service-location.json
85f67bd0bef90e546998bdaf62abd86060806a2d8a05d2432e17069a835c42c1  docs/optimization-guide/progress/2026-10-17-v3/sources/raw-artifacts/m5-ultra-256gb/pareto-mode-frontier-v1/C1-pair1-omlx/result.json
9b9b95193f2449dae69dca0fb4ddbd36f1502b723ecf8236e663b747da2a316e  docs/optimization-guide/progress/2026-10-17-v3/sources/raw-artifacts/m5-ultra-256gb/pareto-mode-frontier-v1/C1-pair3-lithos/result.json
bf86c4ad92499521fb9db8300cbb03a660b9becb5f7c66c4b757019a9a038491  docs/optimization-guide/progress/2026-10-17-v3/sources/raw-artifacts/m5-ultra-256gb/pareto-batch-frontier-v1/C64-pair3-omlx/result.json
```
