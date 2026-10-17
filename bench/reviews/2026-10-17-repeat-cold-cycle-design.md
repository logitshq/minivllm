# Repeat the authorized cold-stop / experiment / restore cycle

2026-10-17. Read-only source review; no live journal/ledger, service query,
model operation or runtime hash was performed. No implementation is included.
The user has authorized the cold stop and subsequent restoration, including
ongoing experiments. Repeating that same contract does not require another
permission question. Current priority remains completing E028 and restoration.
Implement v3 only after the actual new restored identity and evidence exist.

## Actual assumptions that prevent a second cycle

- [controller.py](../m5_ultra_256gb/omlx_cold_restart/controller.py),
  `original_identity` (208–213), hardcodes PID 74384 and its full original
  process identity, launchd PID and listener PID. `absent_service` (237–238)
  checks that PID; `replacement` (331–335) excludes that same old PID.
  These checks must instead use the current cycle's original identity.
- [qualify_absent_owner.py](../m5_ultra_256gb/pareto_realistic/qualify_absent_owner.py)
  has a fixed `ORIGINAL`, a STOPPED-journal equality against it, and
  **`absence(system, pid=74384)`**. Updating a privately loaded module's
  `ORIGINAL` alone does not change this Python default argument. The new owner
  adapter must explicitly bind the current cycle's PID at every absence check.
- [model_owner.py](../m5_ultra_256gb/identity_refill/model_owner.py) pins the
  v2 controller path/SHA. A future owner needs its new controller/predecessor
  binding; the original run, resource and bounded reap logic can remain intact.
- The frozen owner auditors likewise bind `ORIGINAL`, the `ps -p 74384`
  absence command, replacement `pid != 74384`, and the v2 controller/schema.
  A new-cycle auditor needs those narrow identity/binding changes. Preserve
  the old auditors and their historical evidence unchanged.

## Minimal separate v3 design

Use a narrow v3 derived from
[controller_v2.py](../m5_ultra_256gb/omlx_cold_restart/controller_v2.py), retaining
its raw physical/log evidence, resource gates, terminal/reap protocol and
monitored single POST. Do not introduce a generic service lifecycle framework.

The new owner plan's cold-restart binding adds only a predecessor path/SHA:
an **immutable byte-for-byte snapshot of the preceding completed RESTORED
journal**. Its restored-service contract and associated evidence must already
be verified, including absence of a failed POST monitor. Do not keep a mutable
live journal as the predecessor identity oracle or rewrite any old journal.
An experiment's numerical failure does not forbid another cycle if restoration
was independently completed and verified.

New `prepare` derives `original = predecessor.replacement`. It then checks that
exact full `{pid, ppid, start, comm}` identity against the actual process, fixed
launchd label/plist and port 8001, plus the same idle, loaded-target status.
It must not discover an arbitrary listener and silently accept it as the
authorized original. Save the predecessor binding in a fresh journal.

Every stop/absence/replacement check then uses that cycle's original. Preserve
the existing conservative refusal of an unexpected process/PID reuse. The next
cycle repeats this procedure from its own verified RESTORED snapshot.

Do not evade source identity by changing a privately loaded module's `HERE` or
`__file__`. V2's `owner_plan`, `authorize`, `load_worker` and `monitored_load`
use their real source path for SHA checks and the POST-worker CLI. A real new
v3 file and worker entry point must own those bindings; replacing one module
constant cannot make the frozen v2 implementation repeatable safely.

## Keep the contract and guards

Keep the fixed proposal/evidence as the service-contract genesis, its plist
and model-settings pins, launchd label, model/path, version and variant checks,
resident PLE / Lightning MTP log checks, pressure-1 requirement, 40 GiB cold
and 240 GiB loaded physical bounds, swap/stability gates, deadlines, direct
ownership and reap rules. No resource algorithm needs to change for a new PID.

Restoration still means the persisted-visible variant. It does not restore the
old PID, warm KV or exact historical runtime signature. A failed/unknown load
retains its intent and evidence; no repeated POST or automatic service kill is
added. This note neither asserts that the current restoration has completed nor
changes the currently running frozen E028 plan.
