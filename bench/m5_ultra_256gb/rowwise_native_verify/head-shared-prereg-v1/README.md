# Shared-kernel full-prefix numerical handoff

Status: **CPU_VALIDATED; actual full-prefix device run pending.**
Plan SHA256: `f6aeddb2af9e0958e16b45c03c1360cd96f5662ad65c43700809382f2ff3df92`. Exact child/audit argument arrays and the unchanged four MLX
variables are in `commands.json`; `preparation.json` records the actual source,
model, fixtures and runtime bindings. Actual preparation command and result are
preserved in `../prepare-head-shared-attempt-v1/`.

Only two literals in each existing `qualify.py` and `audit.py` changed: HEAD path
and HEAD SHA now select `nvfp4_multirow_shared/candidate.py` at
`d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e`. The rowwise candidate, native bridge,
parameter handling, nine cases, all prefix/state comparisons, rejection cases,
1800-second deadline, runtime environment and resource assumptions are unchanged.
No new qualifier, owner or model loader was added.

The unchanged focused suite actually passed **40/40**, with zero errors,
failures/skips, source before/after hashes equal and no MLX/NumPy/Torch imports.
`../cpu-head-shared-v1/` retains its runner, source copies, command and stdout/stderr.
The previous `head-prereg-v1/plan.json` and 40-PASS source snapshots in
`../cpu-head-reuse-v2/sources/` remain unchanged; the old executable path now has
this new source identity, so its old plan must not be passed to the current entry.
The first kernel's performance rejection is not a numerical failure.

The root reported E040 six real-projection cases completed with numerical PASS;
T4/T8 head baseline/candidate ratios were 1.1692/1.2581 against native B1 and
1.0955/1.1360 against custom B1. T2 was slower; up projection mostly had no gain.
This motivates the next actual-prefix numerical check only. It does not establish
a whole-model, C8 native-batch, speculative-acceptance or HTTP speedup. The present
nine-case plan retains T2 as correctness coverage, not an adoption recommendation.

## Scope and rough resource accounting

One fresh Qwen model is loaded; the native teacher and candidate share its original
parameters. There is no second model or derived full-model packed weight copy.
Gates are ON, norm reuse OFF, native original compiled256 path ineligible at all
registered positions. Reference uses the shortest D fixture (886 IDs); used257
uses its first256 IDs as a separately identified diagnostic prompt.

- Teacher decode calls: `(257-256+7)+(1023-886+7)+(1024-886+7)=297`.
- Candidate: T2/4/8 at used257/1023/1024 = nine calls, 42 retained prefixes.
- One complete state payload is
  `16*2*4*1536*256*2 + 48*(3*10240*2 + 48*128*128*4)`
  = 254,607,360 bytes = 242.8125 MiB.
- Eight complete prefix states total 1.89697 GiB logically; KV sharing may reduce
  physical residency, while canonical inputs, comparison temporaries, lazy graphs,
  allocator/driver storage and host file cache add costs. This is not a peak bound.
- All96 raw snapshots contain approximately 22.7942 GiB of state/logit payload
  (`96*254607360 + 66*248320*2` bytes), plus small integer arrays/headers/JSON.
  Keep at least30 GiB fresh disk for the producer; auditor reads the same raw files.
- The qualifier records MLX memory but has no process-physical allocation cap.
  Two-GiB cache setting limits allocator cache, not total residency. The external
  owner must retain its existing physical/system/pressure/swap/deadline/reap gates.

No measured runtime or physical peak exists for this nine-case path. A planning
allowance of minutes rather than seconds is reasonable because full parameters
and approximately23 GiB raw states are written/read/hashed repeatedly; the
1800-second child deadline remains the actual hard work bound. Do not interpret
that deadline, logical bytes or the previous local projection speed as an ETA.

A roughly143 GiB paused service plus one Qwen model is not by itself evidence
that cold stop is required, nor evidence that SIGSTOP is safe. SIGSTOP retains
resident allocations. Root may use the already authorized pause lease only if
its unchanged live admission/physical/system/swap gates pass with sufficient
headroom and actual child lifecycle proof. Otherwise use the authorized cold
cycle. No resource gate is weakened by this preparation.

Only `forward_prefix_qualified` can become true after the actual producer and
independent original-byte auditor pass. All96 raw files, 5,418 output tensor pairs,
all42 full-vocabulary/prefix states, input preservation and nine evaluated head
receipts remain required. Keep failures and partial runs unchanged. Acceptance,
rollback, EOS retirement, compiled blocks and performance remain unqualified.
