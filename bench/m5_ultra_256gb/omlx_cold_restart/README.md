# Fixed-service cold restart controller

**CPU-reviewed preparation; not ready for a cold-service experiment.** The
[independent interface audit](../../reviews/omlx-controller-interface-audit.md)
identified two required additions: a service-absent owner/auditor protocol and
a durable pre-spawn-failure recovery receipt. Reload also needs an external
physical-memory monitor and complete raw evidence retention. Do not consume
the pending approval or execute stop/recover until these are resolved.

Preparation only. No stop, bootstrap, model load, login, settings write, or GPU
operation has been executed during development. The user's pending answer is
not authorization. Root must review this code and receive an explicit trusted
approval of [the recovery contract](../../reviews/omlx-cold-restart-plan.md)
before invoking either mutating phase.

`controller.py` is stdlib-only and exposes `prepare`, `stop`, and `recover`.
Only `System` contains system/HTTP operations. There is no general command
runner, authentication setup, kill fallback, model unload, settings editor,
experiment launcher, or model inference route exposed through its CLI.

The fixed proposal SHA is:

```text
e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8
```

`--approved-contract cold-restart-persisted-visible-v1` is an explicit operator
assertion that the human approved the reviewed contract. The string itself is
not a substitute for that authorization. Both mutating phases require it and
the fixed plan SHA. The journal pins the controller's actual source SHA too;
source changes after prepare require independent review, not editing a live
journal to bypass a failed check.

## Proposed invocation sequence — do not execute without approval

Use a new journal directory whose parent already exists and one **new,
predeclared guarded-v2 owner directory**. Read-only prepare checks the live old
PID/start/launchd/listener and exact physical model set, records the original
log identity, and pins model config/index metadata. It reads only global
settings file stat metadata, never its credential-bearing contents.

```text
python3 controller.py prepare --journal /absolute/new/journal \
  --owner-directory /absolute/new/experiment-owner \
  --plan-sha256 e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8
```

After approval and review, the stop phase rechecks everything before the exact
single-job bootout. The original 5-second launchd termination deadline remains;
the controller does not promise a complete SSD cache flush.

```text
python3 controller.py stop --journal /absolute/new/journal \
  --approved-contract cold-restart-persisted-visible-v1 \
  --plan-sha256 e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8
```

`STOPPED` proves the original PID, launchd job and port listener are absent.
It does **not** admit a GLM load. Root must separately validate physical memory
reclamation, stable pressure and the experiment-specific loader budget. Root
owns the only GPU producer and must launch solely the predeclared guarded-v2
owner while this journal remains `STOPPED`. No experiment may start once
recovery is requested. The controller is not a system-wide GPU admission lock.

The independent recovery process may wait for that one owner to close. Its
absolute watchdog bound is 0–3,600 seconds; it retries only `NotReady` resource
or owner-reap conditions, never unknown bootout/bootstrap/POST outcomes.

```text
python3 controller.py recover --journal /absolute/new/journal \
  --approved-contract cold-restart-persisted-visible-v1 \
  --plan-sha256 e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8 \
  --watchdog-seconds 3600
```

For a deliberately cancelled experiment, use `recover --no-experiment` with
the same mandatory contract/SHA arguments. It works only if the predeclared
owner directory never appeared. Once it exists, a missing or corrupt reap
receipt is a refusal requiring independent investigation; `--no-experiment`
cannot override it. The operator must also cancel any pending producer before
selecting this option.

## Durable states and refusal behavior

Every operation intent is fsynced before the external action. Journal updates
use a same-directory temporary file, file fsync, atomic replace, then directory
fsync; an exclusive `flock` prevents concurrent controllers on the same journal.
Journal permissions inherit a mode-0700 parent. States are:

`PREPARED -> STOP_INTENT -> STOPPED -> BOOTSTRAP_INTENT -> JOB_RESTORED -> LOAD_INTENT -> RESTORED`.

The journal also retains event timestamps, command return codes including an
unknown timeout, selected resource samples, closed-owner receipt SHA values,
the replacement process identity and final restoration evidence. Errors leave
the last durable intent intact and append `controller_incomplete`. They do not
change an unresolved operation to a false success.

- A replayed `STOP_INTENT` only reconciles complete absence; it never repeats
  bootout. A still-live job after an uncertain stop needs operator review.
- A replayed `BOOTSTRAP_INTENT` waits for a replacement job/listener and checks
  it. It does not issue a second bootstrap. If no job materializes, it refuses.
- Before bootstrap **and again before POST load**, recovery requires guarded-v2
  `attempt.json`, `child.json`, and `release.json`, created after removal,
  matching child PID, integer return code, `waited_reaped=true`, and
  `strict_absence=true`. Both child and owner PIDs must currently be absent.
  Even a reused PID causes refusal. A failed experiment is acceptable after
  verified reap; a missing success summary does not justify skipping cleanup.
- A replayed `LOAD_INTENT` polls the existing loader or verifies the completed
  model. If there is neither an active loader nor a loaded model, it refuses
  instead of posting again. A network timeout cannot cause duplicate loading.
- A new service PID is expected and bound to the original job and listener.
  Any subsequent replacement during recovery is a refusal. The old PID is
  never blindly resumed/signalled. Every status endpoint must show idle work;
  other physical loaded models or other pinned models fail the contract.

The absolute process deadline is 90 seconds for prepare/stop and
`1800 + watchdog_seconds` for recover. Each individual subprocess has a bound,
each GET has a two-second timeout, and the single model-load POST has a
600-second socket timeout. A 120×5-second follow-up poll handles an unknown
load outcome. The absolute alarm bounds slow HTTP body delivery as well.

## Physical recovery admission and log evidence

Each recovery admission uses two `/usr/bin/footprint --sysFootprint` samples,
targeting only the controller PID for process details, separated by at least
four seconds. It records system physical footprint, actual RAM, pressure and
swap; it does not substitute RSS or the model's tracked memory estimate.
Both samples must report 256 GiB RAM, pressure exactly 1, system footprint
at most 40 GiB before loading, no increasing swap, and footprint movement no
larger than 1 GiB. The stable baseline gate is conservative operational policy,
not a guarantee of future loader peak. The original service's recorded peak
was about 184.8 GiB; 40 GiB baseline leaves about 31.2 GiB below installed RAM.
The process and driver can still exceed a historical peak, so root retains the
external resource monitor and recovery responsibility.

After load, physical checks require pressure 1 and at least 16 GiB headroom
under the same stable-sample rule. The controller reads only new stderr-log
bytes since its pre-bootstrap offset, capped at 4 MiB, and requires a single
new target load transaction proving resident PLE, Lightning MTP, resident
table fusion and completed VLM/model load. It records only fixed matched
markers, never arbitrary request/log text. Rotation, truncation, missing
markers, SSD/mmap PLE or a changed model/config produces incomplete recovery.

Successful completion is named `restored_persisted_visible_variant`, with
`exact_runtime_signature_restored=false` and `warm_kv_restored=false`. It is
not an inference qualification or performance observation.

## Required evidence and practical limits

The actual backend has not yet been run. Before authorization is consumed,
root should run read-only `prepare` and inspect the selected live metadata.
The launchctl parser accepts only top-level tab-indented fields and explicit
exit code 113 for an absent job; an OS-format difference fails closed. It does
not print raw launchd argv/environment output or HTTP error bodies.

Current `guarded_v2.py` gives PID + explicit wait/reap receipts, not an original
experiment start-time receipt. This controller compensates conservatively by
requiring a fresh, predeclared owner directory and strict absence of **both**
PIDs, refusing even PID reuse. It does not claim to know the prior start time.
If the owner dies before writing its release receipt, an independent operator
must verify and document cleanup; this implementation deliberately cannot
guess or synthesize that evidence. A foreign GPU producer outside the declared
owner remains the root owner's responsibility.

The public API cannot export the old hidden runtime signature or force the
original residency. Existing admin access could allow graceful unload, but
this controller does not obtain a session or implement that different
transaction. A whole-process or machine crash can leave the service stopped;
the durable journal tells the independent recovery operator what was intended.
The controller is recoverable, not a claim that its own process can survive a
machine reboot or unconditionally restore a model under pressure.

CPU verification: `python3 -m unittest discover -s
bench/m5_ultra_256gb/omlx_cold_restart -p test_cpu.py -v`. All tests use fake
service/resource backends or mocked subprocess outputs. No sockets, real
process probes, MLX, model tensors or GPU operations are involved.
