# Future owner fix: finish lease cleanup after a waited child

E044 v1 remains a failed parent run. Its saved numerical outputs, timings, source, plan, cleanup failure and later restoration reconciliation are not changed by this revision. No v2 plan has been prepared and no real run is requested merely to replace v1's status.

The observed race was between the numerical child being reaped and the lease finishing its `finally` restoration. V1 treated an absent child as an identity error and sent TERM to the still-running lease, interrupting restoration. The v2 owner recognizes absence only when a prior complete child identity exists and a matching `job-0.json` proves `waited=true`, an integer return code, the exact PID and command, the expected lease PID/PPID and job index. It then stops live child sampling and lets the lease finish under the existing 160-second main deadline.

A nonzero waited return code permits cleanup to finish but never qualifies the run. A failed in-flight physical observation, changed identity, unreadable process or mismatched/missing receipt remains a failure. In `finally`, an independently matched already-waited child suppresses TERM to the restoring lease; the existing 30-second cleanup wait and bounded timeout failure path remain. No physical/resource limit or started-observation rule was weakened. This is a narrow completion-race fix, not a hard resource cap or a general PID-reuse proof.

Executed fake-only checks:

- `cpu-v1`: 5/5 PASS; wait proof validation, full normal-exit race, failed observation retained as FAIL and nonzero child exit retained as FAIL.
- `cpu-v2`: 7/7 PASS; adds changed-live-identity rejection and a stalled drain that reaches the unchanged 160-second deadline. The failed cases still let the already-waited lease close without TERM.

Both attempts preserve command, runner, sources, unchanged v1 owner, stdout/stderr and result. The test guard forbids MLX/NumPy/Torch, real subprocesses, network and signals. No GPU/service operations occurred. The v1 owner, candidate and projection probe hashes remain unchanged.

Owner SHA256: `9d07896bde97159be5d766f3e7cf7a73ea2900270f0a842a503a9bdfd1e7052d`.
