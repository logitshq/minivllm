# GDN-only HTTP qualification cycle

This new adapter runs C1 capture → direct-server reap → C1 numerical replay → C8 capture → direct-server reap → C8 replay. It changes no frozen candidate, HTTP producer or serving source. Default invocation does nothing; execution requires `--run` and an exactly bound STOPPED journal. GDN-only is enabled in the qualification child; E024 route graphs and E028 identity refill remain OFF. This is not timing or Pareto evidence.

`owner.py` retains the tested four-phase 5700-second lifecycle, per-replay 1830-second bound, original runtime/source/model preparation, pressure1/AC/thermal/swap gates, settled async resource probes, direct Popen ownership, TERM/KILL/wait and strict PID/port absence. The only numerical interface changes are the frozen GDN capture/replay schemas, dispatch events, and the actual GDN37 plan/audit/complete prerequisite. Capture cleanup failure or a numerical replay failure stops subsequent phases.

`controller.py` directly reuses frozen route-cycle code objects with this module's explicit namespace. Donor globals and paths remain untouched. The actual new controller path is used for the recovery worker. First/next capture no-Popen proofs, unknown spawn rejection, all phase receipts, original MTP/resident-PLE restoration and monitored load bounds remain unchanged. The only accepted predecessor is the archived E024 route formal-v2 restoration, including its exact source and receipt hashes; a failed experiment can be a valid predecessor if independently restored.

`audit_restore.py` verifies restoration independently of experimental success. `audit_cycle.py` calls the frozen resource/preparation algorithms and binds the separate C1/C8 GDN numerical audits; it does not grant performance, combined-candidate or future-arrival correctness.

## Operator sequence

1. Run the guarded CPU suite before registration. Tests use synthetic ownership/recovery fixtures; they do not establish actual device or HTTP qualification.
2. Run `.venv/bin/python -I -B bench/m5_ultra_256gb/gdn_prework_http_cycle/prepare.py --run`. It writes fresh `prereg-http-v1`, CPU-prepares only, and saves exact commands. It does not inspect or operate a service.
3. Root reviews and runs `commands.json`'s `readonly_service_prepare` command. Only then run `prepare.py --seal-prepared`, which binds the actual PREPARED journal bytes and creates the driver receipt.
4. After preregistration review, root runs `execute.py --run`. It always attempts recovery after a changed service state; an unknown owner launch is never labelled no-experiment. All actual outputs use `.artifacts/m5-ultra-256gb/gdn-http-v1-{owner,cold-journal,execution,cpu-prepare}`.
5. After actual RESTORED, create a fresh analysis directory. Run `audit_restore.py --journal ... --out NEW_JSON`; run frozen `gdn_prework_http/audit.py --child <owner>/phases/c1-replay/qualification --plan-sha256 <actual replay plan SHA> --out NEW_JSON`, then the C8 equivalent. Finally pass both independent numerical references to `audit_cycle.py --owner ... --journal ... --plan-sha256 ... --c1-audit ... --c1-audit-sha256 ... --c8-audit ... --c8-audit-sha256 ... --out NEW_JSON`.

Failures and partial prefixes must remain intact. Missing replay plans are incomplete numerical evidence, not successful qualification. Restoration may independently pass after an experimental failure. No inference process is started by import, default invocation, CPU preparation or offline audit.
