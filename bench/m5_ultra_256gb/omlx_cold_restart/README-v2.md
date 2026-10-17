# Fixed-service controller v2 — preparation only

The frozen `controller.py` remains support code for its OS/status parsers.
The new [controller_v2.py](controller_v2.py) adds a distinct lifecycle; use a
**new** journal, never the earlier v1 PREPARED diagnostic journal. The
[CPU/read-only review](../../reviews/omlx-controller-v2-cpu-v1/README.md)
contains verified scope, source pins, tests and the remaining execution work.
Root has run 16 controller tests and 30 service-absent owner tests successfully.

The CLI for read-only preparation additionally requires `--owner-plan` and
`--owner-plan-sha256`, binding the final service-absent wrapper plan. Its owner
is [qualify_absent_owner.py](../pareto_realistic/qualify_absent_owner.py). The
owner's `qualification` entry must pin the unchanged original 37-request plan;
its `cold_restart` entry pins this controller, the fresh journal path and the
fixed persisted-visible recovery proposal. Runtime journal identity is bound
by the locked STOPPED snapshot, avoiding a circular plan/journal hash.

Caught failures before a durable spawn intent may produce a source-pinned
NO_CHILD_STARTED terminal. Recovery independently validates the terminal,
attempt and STOPPED snapshot, absence of spawn evidence, and current owner
absence. After spawn intent, unknown outcomes cannot use this proof. Forced
termination before a terminal is recorded still needs manual reconciliation.
The lock is released between watchdog retries so the owner can publish its
attempt/intent without deadlocking recovery.

Every physical sample keeps raw command output and footprint JSON; bounded
reload log windows retain original bytes, offsets and SHA. A fixed POST worker
is the controller's only child, with a 600-second deadline and continuous
bounded physical sampling. Terminating that worker cancels only the HTTP wait:
the service loader may continue. Failure retains LOAD_INTENT and refuses an
automatic success or second POST. No automatic service kill is added.

**Not yet ready for a cold experiment.** A separately registered wrapper plan
and service-absent independent auditor remain to be completed; actual v2
prepare/stop/bootstrap/load have not run. Explicit cold-restart authorization
remains pending. The user-approved temporary SIGSTOP/SIGCONT lease does not
cover this new cold restart. Full-model/performance pressure-1 gates remain.
