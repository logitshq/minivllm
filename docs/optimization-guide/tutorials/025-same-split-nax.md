# E025: keep split-K fixed, then test NAX against native arithmetic

The first M5 Ultra device screen rejected this candidate at the partition-output
comparison: **11 of 233,472 BF16 elements differed**. Default-OFF and reconstructed
SIMD-reduction controls had already matched native QMM exactly. NAX compiled and
produced finite output, but its final reduction was not run. This experiment adds
no point to the throughput–complete-request-latency Pareto curve.

## Hypothesis and the smallest useful experiment

For the three selected NVFP4 projections at T19, pinned MLX 0.32.3 selects a
SIMD split-K QMM path. The proposed optimization was to use the M5 NAX matrix
hardware inside each existing partition while keeping the partition count,
partition boundaries, BF16 partial-output storage and final reduction fixed.
This isolates a potentially useful hardware change without loading a model or
changing scheduling, quantization or the amount of contraction work.

The private helper preserves full-K strides for activation, packed weight and
scale rows. For partition start `k_start`, the offsets remain `k_start`,
`k_start / 8` packed UINT32 words and `k_start / 16` scale bytes. The NAX helper
changes its contraction bound to `K_eff`; it does not substitute `K_eff` for the
original row stride. Each partition writes its own `[T, N]` BF16 output.

| Registered projection | T | K | N | Partitions | K per partition |
|---|---:|---:|---:|---:|---:|
| gdn_z | 19 | 5,120 | 6,144 | 2 | 2,560 |
| mlp_down | 19 | 17,408 | 5,120 | 2 | 8,704 |
| out | 19 | 6,144 | 5,120 | 3 | 2,048 |

SIMD uses BM/BN/BK 32/32/32; NAX uses 32/64/64. All registered K partitions
satisfy the NAX tile boundary. The original helpers are expanded from pinned
Apple MLX headers in separate libraries, retaining their licenses. The candidate
is isolated in a benchmark module and remains default OFF; production core was
not patched. Source and the original scope are in the
[candidate README](../../../bench/m5_ultra_256gb/prefill_nax_same_split/README.md).

Equal mathematical sums are not sufficient evidence for equal floating-point
results. Changing matrix instructions, tile geometry or accumulation order can
change rounding. This is why the experiment checks complete original BF16 bytes,
including signed zero, before considering speed. The present evidence identifies
a difference, not the precise instruction responsible for it.

## Fix the reference before judging the candidate

The v2 screen retained the original seed `20261017`, synthetic construction and
six fallback lengths (1, 11, 12, 32, 33, 128). It planned 33 comparisons across the
three shapes, with no tolerance or post-hoc exception:

1. Default-OFF output versus native `mx.quantized_matmul`.
2. Reconstructed SIMD partials reduced with the original expression versus native QMM.
3. Every NAX BF16 partial versus the corresponding SIMD partial.
4. NAX final reduction versus native QMM.
5. The actual enabled callable versus native QMM, followed by the six fallback checks.

The second comparison establishes that the exposed SIMD reference agrees with the
actual native operator for this input. Skipping it could make two custom kernels
agree with each other while both fail to represent the path being optimized.

V2 changed evidence collection and bounds, not the candidate arithmetic: every
comparison saves both full original-dtype tensors, including successful prefixes;
X has an explicit finite check; complete X/W/scales snapshots are saved before
and after execution, including the failure path. The small diagnostic uses
cache0, a 120-second child alarm plus an external direct owner, and separately
registered MLX 1 GiB / physical 2 GiB observation limits. These are observed
limits, not OS hard caps or a relaxation of the full-model performance policy.

## What actually ran

Plan commit `4d97a8d02b2d20022ee314ddb9406b486e3d7b62` was created at
2026-10-17 12:29:38 UTC, before the recorded preflight at 12:30:12 UTC.
The fixed plan SHA256 is
`3beae0672fd90d6a83a3869f66164273229751151d99868259dcb3f1c4b16a12`.

| Actual check on gdn_z T19 | Compared elements | Result |
|---|---:|---|
| Default OFF versus native QMM | 116,736 | Exact, all finite |
| SIMD partial reduction versus native QMM | 116,736 | Exact, all finite |
| NAX partials versus SIMD partials | 233,472 | FAIL: 11 differing elements / 11 differing bytes |

The first difference is byte 4,700, BF16 element `[0, 0, 2350]`:
SIMD `0x3d29` = 0.041259765625; NAX `0x3d2a` = 0.04150390625.
Six differing elements are in partition 0 and five in partition 1. The largest
absolute difference among these partials is 0.125. These magnitudes describe the
failure; they do not introduce an error tolerance. Both complete partial arrays
are finite, and all X/W/scales bytes match their original snapshots.

The strict gate stopped immediately. **NAX final output, enabled callable,
fallback checks, mlp_down and out were not run.** A partial mismatch does not by
itself demonstrate a difference in a final reduced output or model logits. It
does reject this registered candidate under its unchanged partition-equality gate.

Six self-resource phases and three external resource samples were retained. The
last self-phase is **before** the failing comparison, not after its evaluation.
Observed MLX historical peak was 45,219,886 B and maximum physical peak reading
was 335,659,752 B. No final-release phase or child terminal source recheck was
reached. These samples cannot establish the peak of the missing completion phase.
System samples stayed at the separately admitted pressure level 2, with swap use
445.06 MiB and Swapouts 316,235 unchanged; minimum real free memory was
67,817,275,392 B. There is no performance conclusion from this brief screen.

Owner 55019 directly owned child 55031. The child exited 1, was waited/reaped and
passed strict PID-absence verification. The outer lease restored the same original
service 74384 from T to S. Parent before/after pin checks remained intact.

## Keep the failed attempt reproducible

The independent CPU auditor reads the safetensors headers and all payload bytes
without MLX or NumPy. It checks finite BF16 encodings, every byte, all 11 difference
coordinates, complete input preservation, physical raw readings, the lifecycle and
570 source/runtime pins. It also reconstructs both recorded Metal sources from
frozen headers. Four CPU negative tests cover signed zero, equal NaN bits,
malformed tensor layout and invalid physical readings. Root repeated the audit;
the two audit receipts are byte-identical. Audit PASS means that the failure is
supported, not that the candidate passed.

The [experiment archive](../../experiments/m5-ultra-256gb/E025-same-split-nax-v2/README.md)
contains the executed plans, raw tensors, owner and lease records, source copies,
CPU preparation, root raw review and both audits. Each tar member, including
hardlinked duplicates, was reread and checked against its SHA256.

A future arithmetic attempt needs a new preregistration and must retain the native
reference and exact gate. The retained 11 coordinates provide a small diagnostic
set for locating the accumulation difference, followed by the entire registered
matrix; they must not become a handpicked passing workload. Only after the
operator and required full-model correctness gates pass can a controlled HTTP run
measure complete-request latency and aggregate throughput. Until then, leave this
candidate OFF and keep the [qualified Pareto plot](../progress/2026-10-17-v5/README.md)
unchanged.
