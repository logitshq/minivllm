# GLM dual-cadence owner v3 — prepared, not executed

This version keeps the original GLM model/runtime checks, direct model Popen,
220 GiB process / 240 GiB system ceilings, zero swap growth, physical parser,
TERM/KILL/wait and postflight gates. The model probe itself is unchanged.
Only live monitoring is split: a physical lane with maximum 1 s end-to-start
gap, and a metadata lane with maximum 5 s gap. Both callbacks have 5 s execution
deadlines. The scheduling targets are half those maximum gaps.

`owner.py` reuses the SHA-bound original functions in its actual new namespace;
its explicit `RUN_DELTAS` preserve preparation, spawn and cleanup code. A normal
model exit cannot turn a failed observation into a success. Failed-but-settled
monitoring allows recovery; unknown probe Popen, unreaped read commands or live
workers suppress the canonical `all_children_reaped` release.

The owner CLI is unchanged: `owner.py --run --plan PLAN --plan-sha256 SHA --out NEW`.
Omitting `--run` describes without executing. The plan has the original eight
fields plus `monitor_policy`, exactly `owner.MONITOR_POLICY`. Its schema is
`GLM-cold-load-owner-plan-v3`; the journal and result schemas end in `v3` too.
The separately supplied controller must be the real adjacent `controller.py`.

`controller.py` reuses the original stop/recover/file-worker algorithms. Its
normal restoration protocol also checks every retained monitor command intent,
returned PID and final wait receipt, including failed-but-reaped commands.
`audit_restore.py --journal CLOSED --out NEW_JSON` reuses the original raw
physical/log/load-worker restoration checks and that same file-only reap chain.
Neither tool grants model correctness, model fit or performance eligibility.

Execution is deliberately blocked while `predecessor-manifest.json` is
`UNSEALED`. The current source accepts only a real `cold_observed_cycle` normal
`RESTORED` single-child journal, successful recovery worker and independent
restoration PASS. Its older observed `LOAD_INTENT` branch is explicitly excluded.
The root is planning another HTTP cycle first: accepting that future four-phase
predecessor needs a reviewed, explicitly pinned adaptation after it actually
closes. Nothing here discovers a replacement PID or invents future evidence.
Seal the actual predecessor manifest, controller, restoration auditor and closed
auditor in that order before creating the next execution plan.

The adjacent `audit_closed.py --journal CLOSED --plan-sha256 SHA --out NEW_JSON`
keeps the frozen physical, metadata, token and inventory checks. It replaces the
old 1:1 count relation with independent lane inventories, sample/command ranges,
raw command equality, full child identity and actual gap/deadline checks. Each
physical callback has two identity commands around the original five probes;
each metadata callback has the original seven commands. Redacted process args
remain hash/size evidence, not independently available raw bytes. The result
schema is `GLM-closed-load-audit-v3`; failed experiments do not become diagnostic
PASS merely because restoration passed. There is no execution registration or
real GLM result in this directory.

CPU mechanism evidence is retained separately: the monitor's first 13-test run,
its single incremental-history regression, the v3 owner's first 9-test run,
6 lifecycle tests and 8 independent-resource tests all passed under
subprocess/socket/signal/backend guards. Full stdout/stderr, runner and source
snapshots are under `.artifacts/m5-ultra-256gb/`:

- `glm-dual-monitor-cpu-v1` — first 13 tests, before the later narrow changes.
- `glm-dual-monitor-cpu-v2` — one new incremental-history regression; current
  helper module has 14 tests, not all rerun after that change.
- `glm-v3-owner-cpu-v1` — 9 tests.
- `glm-v3-lifecycle-cpu-v1` — 6 tests, tool chunk `59db97`.
- `glm-v3-resource-audit-cpu-v1` — 8 tests, tool chunk `dbe1e4`.

They are fake-only; they do not demonstrate macOS timing, physical headroom or a
successful load. The source-only predecessor/auditor seals remain pending.
