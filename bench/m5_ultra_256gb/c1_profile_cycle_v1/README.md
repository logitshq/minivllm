# C1 profile cycle v1

Prepared on 2026-10-19. This is one bounded diagnostic cycle with **three actual direct server children**, ordered `off`, `runner`, `full`. Each uses the same qualified E033 combined-ON C1 specification and receives eight warm plus eight diagnostic D8 requests. The 48 requests are not new Pareto points.

The implementation reuses the frozen owner `trial_one`, tracked `spawn`, resource `probe`, strict `release`, deadlines and outer cleanup without changing those nested functions. Only plan/spec acceptance, diagnostic result publication and terminal naming differ. The fixed cold-controller stop/reap/recover algorithms receive the actual three-phase list. No old source, completed journal, formal result or donor module dictionary is patched.

The actual E034 GLM v3 experiment failed its swap-growth observation. Its normal RESTORED journal and independent restoration PASS are the predecessor here. The manifest binds those existing bytes and the complete GLM model/read-command reap chain; an experiment failure is not relabeled as success. The predecessor validator rejects UNSEALED, LOAD_INTENT, incomplete restoration and unresolved handles.

## Current preparation

- `prereg-v1/owner-plan.json` SHA256: `f70b5cf3015f3c382696ce24e4f2891e66e56916c68b0617ec81f3ab31911343`.
- 90 owner source/input pins, plus the original native runtime/model pre/post verification.
- Owner work bound: 10,356 seconds, followed by the existing bounded cleanup/postflight procedure.
- Actual owner `--prepare-only` CLI: return code 0; command and stdout/stderr retained under `prereg-v1/`.
- CPU attempts: `.artifacts/m5-ultra-256gb/c1-profile-cycle-cpu-v1/` and `c1-profile-cycle-cpu-v2/`, both 13/13 PASS. The second also validates the actual E034 predecessor and real CPU registration. Source hashes match before/after.
- No controller/service preparation, stop, model/server launch or device execution has been performed by this implementation task.

All `.py` sources and the predecessor manifest are frozen for this registration. Editing any of them requires a fresh registration; retain this attempt.

## Root-owner operation

Use the exact argument arrays from `prereg-v1/commands.json`.

1. `owner_prepare` already completed successfully. Do not rerun it into its existing output.
2. Run `controller_prepare`. This performs the existing read-only service identity/settings checks and writes the actual PREPARED journal.
3. Run `preregister.py --seal-prepared <absolute prereg-v1 directory>` to bind those actual PREPARED bytes. The seal also requires the real owner preparation receipt.
4. Review and commit/push the sealed sources and preparation before stopping the service.
5. Run `execute.py --run --registration <absolute prereg-v1 directory> --out <fresh absolute execution directory>`. The driver executes stop, the actual `--run` owner, then recovery in `finally`. Unknown owner launch does not receive a false `--no-experiment` flag.
6. After recovery, run `restoration_audit` and `closed_audit` from the saved commands. Retain failed captures and audit outputs unchanged. The independent diagnostic binder checks all raw request counts, ready/listener identity, 16 DONE/closure/profile joins per mode, actual PID/port reap, resource history and restoration.

The registered future output paths are under `.artifacts/m5-ultra-256gb/c1-profile-cycle-v1-{owner,journal}`; diagnostic audit output is `analysis-v1/independent-diagnostic.json` in this namespace. No output path is recycled from an earlier cycle.

## Interpreting the diagnostic

OFF/runner/full share admission/DONE/drain bookkeeping. Report their request-latency overhead before interpreting spans. Python runner, graph submission, existing eval, queue, decoding and HTTP spans overlap; their sum is not exclusive GPU time. Natural exit can race a drain checkpoint, so an unknown PID is accepted by the offline resource binder only for `draining` with the exact successful empty `ps` absence result. Unknown request-phase identity remains a failure.

Use the result to choose the next actual optimization: material uncovered host work supports testing the prepared `ahead_submit_once` or `long_decode_onepass` candidates; predominantly overlapped host work supports prioritizing numerical qualification and actual cost of layer-major weight reuse. Source structure alone does not establish a speedup.
