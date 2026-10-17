# E027 · M5 shared-weight M1 · numerical screen v1

Status before execution: UNRUN. Default OFF. The candidate is not installed in
an engine and creates no GPU work on import. M4 evidence is source provenance,
not M5 qualification. Original M4 files remain unchanged.

The hypothesis is that 2/4/5 activation rows can reuse each packed NVFP4 weight
and its group scale while preserving the original M1 per-row accumulation,
SIMD reduction and BF16 store. Saving source-level loads is not proof of lower
DRAM traffic or end-to-end speed: caches, registers and occupancy can erase it.

This attempt is numerical only: 27 synthetic cases (three shapes × three banks
× rows 2/4/5), plus six cases using two actual Qwen3.8-27B layer-0 projections.
The shapes are (N,K)=(48,5120),(1024,5120),(10240,5120). Real weights cover N48
and N10240 only; activations remain deterministic synthetic inputs. Native M1
row-by-row is the oracle. Both the original-expression control and shared-W
candidate must be finite and exactly equal in original BF16 bytes, including
signed zero: 33 cases, 66 comparisons. Preserve every input/output and both
successful and failed cases; stop on the first mismatch.

A fresh child gets 120 seconds, MLX cache0 and observed active+cache/peak ≤1GiB;
all three observed process physical readings must be ≤2GiB. The owner gets the
existing 450s work +30s cleanup bound. Reuse E026's fixed resource policy only
for this small numerical screen: initially pressure1 or stable existing2,
no new Swapouts, swap growth ≤16MiB, ≥50GiB actually free pages, AC and thermal
0/1. These observations are not OS hard limits. Full-model/performance gates
remain unchanged. The user-authorized outer SIGSTOP/SIGCONT lease alone pauses
and restores original omlx-server74384; this does not unload its memory.

Before launch, commit the exact plan/jobs/source/runtime/bundle pins in
prereg-v1/. Raw artifacts use new shared-w-m1-v1-* directories. An independent
raw-file audit is required after the child's PASS. No diagnostic elapsed time,
memory saving, or local matrix qualification is a throughput–latency point.

The final objective remains correctness-gated throughput vs complete-request
latency on the same model and workload, compared with oMLX and Lithos. TTFT,
ITL and memory explain results; they do not replace those axes. The next gates
are propagated whole-model states/outputs, then uninstrumented same-round HTTP
concurrency sweeps. Only those results can update the Pareto curve.
