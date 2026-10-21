# Device Timing Analysis

Gate: PASS_FEATURES_LOCAL_COST and PASS_CAPTURE_AND_RESTORE verified.

| width | arm | median ms | min | max | n |
|---|---|---|---|---|---|
| 4 | native | 93.00 | 92.42 | 97.67 | 10 |
| 4 | shared_without | 59.52 | 58.32 | 60.50 | 10 |
| 4 | shared_with | 59.76 | 58.63 | 60.55 | 10 |
| 8 | native | 185.65 | 183.74 | 186.18 | 10 |
| 8 | shared_without | 109.76 | 108.24 | 114.88 | 10 |
| 8 | shared_with | 110.12 | 108.55 | 111.22 | 10 |

Ratios native/shared_with per group (median of 2): T4 [1.538,1.609,1.565,1.548,1.560] median 1.56; T8 [1.661,1.688,1.684,1.684,1.699] median 1.684. Capture overhead with/without 1.004 (T4) 1.003 (T8).

Scope: 10 samples/arm/width, no CI, T=verifier input width not concurrency; native uses ORIGINAL MODEL CAPTURE on supplied teacher tokens (not device-dependent AR). Does not measure draft, actual acceptance, full serving throughput or Pareto. Isolated latency ratio is not end-to-end speedup.

Hashes: source 7d4c0c349164 result 0ae0ec0c4e02
