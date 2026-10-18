# GLM-5.3-Flash first cold-load attempt

The GLM load did not complete. No request was generated and no model correctness, fit, quality or performance qualification is granted.

- The owner retained 37 live physical observations, then stopped on `resource checkpoint exceeded observation cadence`. The final seven-command checkpoint took 1.598319 s, exceeding the fixed 1 s post-physical cadence. It passed its separate 5 s checkpoint budget. The preceding five-command physical probe took 0.244151 s. Individual command times were not recorded; attributing the delay to a particular command is unsupported.
- The direct child 71812 was sent TERM, then KILL after the bounded wait, and strictly reaped with code −9. The original four-request diagnostic never reached its first request.
- Restoring the original persisted Qwen service reached a real replacement, PID 72953. Its HTTP-load monitor failed after swap increased by 849,347 B from that monitor’s baseline. The HTTP worker was reaped with code −15. The original controller journal remains `LOAD_INTENT`, `post_monitor_failed=true`; driver recovery return code remains 1.
- A separately authorized read-only check subsequently verified the actual replacement PID/job/port, idle loaded model, unchanged persisted settings/model metadata, resident PLE and MTP completion log markers, old child/worker absence, and two stable fresh physical observations. Its status is **OPERATIONALLY_RESTORED_WITH_MONITOR_FAILURE**. It does not convert the failed historical monitor or HTTP response into success.

`service-baseline.json` and `operational-recovery.json` are exact copies of those separate receipts. Any future lifecycle must explicitly accept this observed-baseline contract; the original RESTORED prerequisite is not met. The saved old `closed_audit` command was not run because its original gate was not met.

`raw.tar.xz` preserves preparation-execution, preregistration, owner, original cold journal, driver execution, analysis, and the three supplied input-preparation trees. `manifest.json` records every member and SHA; `archive-verification.json` records streamed tar comparison and a full original-tree recheck. The one `probe/models` symlink is retained only as link metadata. **Do not use unrestricted extraction or follow that link**: it points outside the archive to the model snapshot. No model weights or runtime binaries are included.

Verify without extraction or service access:

```sh
python3 archive.py --verify .
```

`archive.py --run` is a one-shot creator bound to the original paths and refuses an existing destination. It never calls a model, service mutation or recovery command.
