# GLM v3 future formal predecessor: three CPU tests passed

This is a narrow continuation of the [dual-monitor CPU archive](../glm-dual-monitor-prepared-cpu-v1/README.md).
It changes only the next predecessor accepted by the unregistered v3 controller:
the future `E032-gdn-graph-combined-formal-v1` immutable journal/audit pair,
twelve fixed phases, and the exact reviewed owner/controller/restoration sources.
The original formal-restoration checker is retained with one exact journal-schema
literal changed. Exact archive and owner-source checks are additional. The
manifest is still **UNSEALED**, with no fabricated future PID or result SHA.

A failed experiment may restore successfully; this permits that case only when
normal `RESTORED`, independent restoration PASS and HTTP recovery worker code zero
are proven. Unknown launch or incomplete recovery receives no exception. The
owner, monitoring deadlines/budgets and stop/recover algorithms are unchanged.

First guarded command:

```sh
.venv/bin/python -I -B /tmp/run-glm-v3-formal-predecessor-cpu-v1.py
```

Actual result: **3/3 PASS**, zero failures/errors/skips/forbidden operations,
source pre/post equality, 0.062957542 s; tool `0b8464`. Tests use synthetic archive
bytes and real pinned source checks, not process/service observations. They
check restored-failed-experiment acceptance, malformed ownership/recovery
rejection, exact source/paths, and the single-literal AST delta. The earlier
six-test lifecycle module was not rerun for this change.

The archive retains full stdout/stderr, runner, results, all 20 run-time source
snapshots, and the current source/documentation. **31 regular members** contain
328,399 raw bytes; compressed size **66,804 bytes**. SHA-256:
`da2ec7addcdb1c9c27f52561711b7025a12e55bf39bbc0462294c45ee6dd5993`.
Packing and a second safe-extraction/SHA verification both passed (tool `0f2d6f`).

```sh
python3 docs/experiments/m5-ultra-256gb/glm-formal-predecessor-prepared-cpu-v1/verify.py
```

No model loading, GPU use, live monitoring, service action, actual GLM v3
execution or resource/performance qualification occurred.
