# Fixed single-child cold cycle

CPU preparation only so far. The controller, owner and restoration auditor are
one fixed lifecycle version; new experiments change an immutable run plan, not
these sources. This version does not manage HTTP multi-phase experiments.

`lifecycle.json` binds exactly the three adjacent Python sources. Every entrypoint
checks the real paths and bytes. The reviewed lifecycle SHA and run-plan SHA are
the registration trust anchors: a different directory or arbitrary auditor path
is rejected. Freeze this directory after review; a future lifecycle change needs
a separately reviewed migration, not replacement of a predecessor's manifest.

The run-plan schema is `cold-single-child-run-plan-v1`, with exactly:

- `lifecycle`: `{path, sha256}` for this directory's `lifecycle.json`.
- `predecessor`: archived RESTORED `{path, sha256, audit: {path, sha256}}`.
- `experiment`: `{phase, qualifier: {path, sha256}, plan: {path, sha256}, source_pins: {absolute_path: sha256}}`.
- `inherited_owner_plan`: the unchanged `4dff5f46…caf6a4` E024 owner-plan reference.
- `outputs`: distinct absolute `{journal, out}` directories under this repository's `.artifacts/m5-ultra-256gb`.
- `budgets`: `{work_seconds: 1830, child_seconds: N, term_seconds: 15, kill_seconds: 15}`, where `0 < N <= 1800` and the child plan's `deadline_s` equals N.

The experiment plan must bind `qualifier_sha256`; other experiment-specific
fields are interpreted only by the qualifier and its independent numerical
auditor. `source_pins` includes the qualifier and its reviewed dependencies.
It cannot replace a fixed lifecycle/support source. The owner does not import
experimental code and constructs only this command, using the Python and
environment returned by the unchanged original preparation:

```
python -I -B QUALIFIER --plan EXPERIMENT_PLAN --plan-sha256 SHA --out OWNER/phases/PHASE/qualification
```

The original preparation and its actual CLI receipt are retained before and
after execution. The owner checks child elapsed time itself, in addition to the
1830-second work alarm. Existing pressure1, AC/thermal/swap, service absence,
source/runtime/model preparation, unknown-Popen, TERM/KILL/wait and strict raw
PID-absence checks remain. A returned PID with failed identity lookup can be
reaped; an unknown spawn outcome cannot become `NO_CHILD_STARTED`.

CLI remains explicit: controller `prepare`, `stop`, `recover` use the fixed
proposal SHA and existing approved-contract argument, with `--owner-plan` and
`--owner-plan-sha256` at preparation. The owner takes `--plan`, `--plan-sha256`,
`--out`. The offline auditor takes `--journal`, `--out`. No command here performs
a live action merely by importing the module or requesting help.

The first migration accepts only the archived E024 route-canary v7 journal SHA
`b8a209b3ea6b888ccf4331b695881a85962f1097d51c46da84338ca084669ce2`
and restoration-audit SHA
`732e5c19b4afde075387907cb2de89026ae42510392fb9ec260e643cee06561e`.
Later runs require the same lifecycle reference and source hashes. An experiment
failure may still have a valid restoration receipt.

Owner PASS means execution/resources/reap completed; it grants no numerical or
performance qualification. Restoration audit checks raw restoration and child
absence evidence, while final management status and owner absence remain
source-pinned controller receipts. It does not independently audit the experiment's
mathematics or performance. Hidden runtime state and warm KV are not claimed restored.

Run the fake-only regression suite:

```
.venv/bin/python -B -m unittest bench.m5_ultra_256gb.cold_cycle.test_cpu -v
```

The two-cycle test uses synthetic restoration-audit fixtures and fake service
operations; it tests the protocol chain, not a real restoration or raw-device
qualification. Other tests exercise the actual owner/controller receipt contract,
rejection paths and original preparation arguments. AST comparisons preserve
the v7 OS/resource/stop/monitored-load/recover implementation.
