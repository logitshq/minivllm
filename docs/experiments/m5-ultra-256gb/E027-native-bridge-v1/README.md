# E027 native-route bridge v1: direct replacement rejected

The preregistered M5 device test stopped on its first native-route difference.
It completed 18 cases and two checks in case 19: 55 comparisons passed, one
failed, and the remaining 14 cases were not run. Numerical qualification is
**FAIL**. The independent audit of this failed record is **PASS**; that does
not turn the candidate into a qualified implementation.

The earlier E027 screen compared the candidate with independent per-row M1
operations. Current M5 `Linear.__call__` instead sends the complete row matrix
to `mx.quantized_matmul`. For the tested T2/4/5, K5120 shapes on pinned MLX
0.32.3 / g17d, native dispatch selects `fp_qmv_wide`, which already reuses
weights across rows. Neither M1 equality nor a reduced count of source-level
weight reads establishes native-route equivalence or a speed advantage.

| Observed failure | Value |
|---|---|
| Case | synthetic-N10240-random-T2 |
| Output | BF16, shape 2 × 10240 |
| Differing elements | 5 / 20,480 |
| First difference | byte 13,142; coordinate [0, 6571] |
| Independent per-row M1 | 0xc263, −56.75 |
| Native multirow | 0xc264, −57.0 |

Both root and the independent auditor read the full original output bytes.
[All five differences](root-difference.json) and [independent audit](independent-audit.json)
are retained. Different rounding does not establish that native MLX is wrong.
It rejects the proposed exact substitution of this entire three-geometry route.
The failed case stopped before calling the shared candidate; we do not describe
that missing output as a measured candidate failure. Passing smaller geometries
are not promoted into production from this interrupted prefix.

The plan was committed as `28b67d0` before execution. Plan SHA:
`f52346375b8a2ccecb861240c283d1215d8fee2880f41ae38116ac5b9c3ada59`.
The auditor verifies all 580 execution pins, 139 tensor files, 64 actual GPU
descriptors, 39 self-physical phases and the three external resource samples.
Pressure remained 2 under the separately registered small numerical-screen
policy; no swap-use or Swapouts increase was observed. MLX historical peak was
29,802,632 B; maximum physical peak observation was 386,024,240 B. These are
bounded observations, not performance results or OS hard caps.

Direct child 52256 exited 1, was waited/reaped and passed strict PID absence.
The outer lease restored the same original service 74384 from T to S. The
last bank has original W/scales bytes and unchanged after hashes, but no final
full after-bank files. Child terminal source/bundle rechecks and final-release
phase were not reached. Parent pins were rechecked. These early-stop limits
remain explicit in the audit and original receipts, which were not rewritten.

Root reran the independent auditor and its six corruption refusals; the saved
root and independent JSON receipts are byte-identical. The archive contains
all saved owner, lease, input-plan and review evidence, including the failure.
Identical payloads use tar hardlinks; every member was subsequently reread and
verified against its SHA. The archive does not invent artifacts from unrun
cases. Its manifest is [manifest.json](manifest.json).

Archive: 31,005,640 B; 942 members; 61,630,988 logical raw bytes.

SHA256: `4e8b2ccd0423d2f79d845b4d22aa81facaedcc348d8c5a51704c5436b7149aa8`.

No full-model, HTTP, throughput, latency or production qualification resulted.
The [throughput–latency plot](../../../optimization-guide/progress/2026-10-17-v5/README.md)
is unchanged. Keep the candidate default OFF. The next arithmetic experiment
must first preserve the actual native multirow route, or independently prove
full-model equivalence under the established correctness contract; changing
the baseline to forced M1 would not establish an optimization of the old path.
