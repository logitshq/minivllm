# Actual-model multirow head numerical handoff

Plan: `plan.json`, SHA256 `f90d416fba5b883d4c02a072df4284a0d20ce2e9cd2e06fffff6aa23b8d17be8`.
Status: **CPU_VALIDATED**, device execution not performed by this task.

`commands.json` contains the exact argument arrays for the owned numerical child
and its later independent auditor. It also records the exact inherited MLX
variable set; any additional MLX_* variables must be absent. Root owns stop,
resource admission, direct child handle, hard deadline/reap and service recovery.
No new owner or controller was introduced. Run the audit only after the actual
child has closed and its original evidence is stable; keep failed runs unchanged.

- Child output: `/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/rowwise-head-reuse-v1-child`.
- Child deadline: 1,800 seconds; outer root cleanup remains independently bounded.
- Audit output: this directory's fresh `independent-audit.json`.
- Model: pinned Qwen3.8-27B NVFP4 snapshot `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`.
- Actual head: untied, K5120/N248320, weights U32[248320,640], scales U8[248320,320].
- Cases: T2/4/8 at used257/1023/1024; 42 full prefixes, 5,418 tensor pairs, 96 raw files, nine evaluated head calls.
- Preserve all raw prefix states, including padding; budget tens of GiB of disk as in the original qualifier.

40/40 focused CPU tests passed in `../cpu-head-reuse-v2/`; the initial guard-only
failure in `../cpu-head-reuse-v1/` is retained. No native arithmetic, compile,
acceptance/rollback, HTTP or speed qualification is granted by these tests.

`preparation.json` records all current namespace source hashes, inherited fixed
runtime pins, model/source manifest and fixture refs. `prepare_head.py --run`
produced this plan and successfully ran the actual read-only `qualify.validate`
path. Existing output paths must not be reused; for a new registration invoke
that generator with fresh absolute `--out`, `--child` and `--audit-out` paths.

The implemented kernel is only the head. Its 4.9621% logical projection-weight
share is not a measured DRAM or runtime share. The immediate target is actual
full-prefix numerical PASS; it is not evidence of a Pareto improvement.


## Decision after E038, 2026-10-19

**PENDING / NO ADOPTION: defer this v1 head full-model qualification.**
[E038 independent audit](../../../../docs/experiments/m5-ultra-256gb/E038-real-projection-screen-v1/independent-audit.json)
passed all six actual head/up projection cases (T2/4/8) numerically, using real
weights with deterministic synthetic BF16 activations. However, all 12 observed
median baseline/candidate latency ratios were below 1. Head ratios against
native B1 were 0.9315 / 0.9222 / 0.8209 and against custom B1 were
0.8766 / 0.7995 / 0.7698. The first kernel is therefore rejected for speed;
this is **not a numerical failure**, nor a full-model numerical result.

The plan, pinned source, 40 CPU PASS results and initial failed CPU attempt remain
unchanged and available. Do not run this expensive full-model plan automatically.
The next local screening candidate is the separate default-OFF
`nvfp4_multirow_shared/candidate.py`; its device qualification is still pending.
