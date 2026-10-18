# GLM one-child cycle — prepared source, not registered

`owner.py` runs the existing official GLM load probe as its **direct child**. It does not stop, load, restore or send HTTP to the persisted service. Adjacent controller, CPU input builder, registrar and offline audits are now prepared as source only. They have not been imported, tested or registered during the active route formal measurement. No actual admission, PREPARED journal or GLM success receipt has been created. The original owner/probe/physical guard remain frozen.

The 12 pure-fake tests in `test_owner_cpu.py` all **passed on 2026-10-17** under a guard that denies real subprocess/network/signals and model/backend imports. [Complete CPU evidence](../../../docs/experiments/m5-ultra-256gb/glm-owner-prepared-cpu-v1/README.md). They cover CLI admission, raw failure retention, missing/foreign process inventory, strict file-stat types, missing probe completion, pre-intent failure, unknown Popen outcome and returned-PID cleanup after an identity-read error. They do not qualify a model, device or restoration cycle.

## Exact integration boundary

The future driver uses an outer stdlib Python to invoke:

```text
python -I -B owner.py --run --plan OWNER_PLAN --plan-sha256 SHA --out NEW_OWNER
```

Empty argv only describes; partial arguments reject. The owner itself starts exactly:

```text
<fixed oMLX venv>/bin/python -I -B <frozen glm_flash/load_probe.py> --run
  --plan PROBE_PLAN --plan-sha256 PROBE_SHA
  --admission OWNER/admission.json --admission-sha256 ACTUAL_ADMISSION_SHA
  --out OWNER/phases/glm-official-load/probe
```

This preserves the existing official EnginePool route, fixed offline environment, original mixed-quantized snapshot, plain prompt and normal-EOS caps `[8,32,32,32]`. There is no Qwen preflight, HTTP server, offload or model conversion.

`GLM-cold-load-owner-plan-v1` has exactly these fields:

| Field | Required content |
| --- | --- |
| `execution` | Actual owner `{path,sha256}` and `source_pins(plan)` exact map, including the actual future controller and probe's listed sources. |
| `probe_plan` | Original `GLM-official-load-plan-v1` path/SHA. No dummy qualifier fields. |
| `verification` | Path/SHA of `GLM-model-runtime-inventory-v1`, described below. |
| `physical_policy` | Exact `physical_guard.policy` result, including explicitly chosen child ceiling and observation cadence. Child work equals the original probe deadline, at most 1800s. |
| `cold_restart` | `journal_path`, actual `controller` path/SHA, immutable `predecessor`, unchanged `approved_contract`, unchanged `proposal_plan_sha256`. |
| `outputs` | Exact owner directory and direct child path shown above. |
| `budgets` | Work1830, prepare300, postflight120, TERM15, KILL15, checkpoint5 seconds. Preparation is inside owner work. Cleanup/postflight are additional bounded work. |

The controller must expose `CONTRACT`, `PLAN_SHA`, `JOURNAL_SCHEMA='GLM-cold-load-journal-v1'`, and `predecessor(binding) -> (prior, raw_pair)`. That function must independently verify an actual immutable RESTORED predecessor and its raw restoration audit. It may not accept an experiment PASS as proof of restoration. The owner verifies the returned replacement identity against the actual STOPPED journal.

The new controller's journal must contain the existing stable fields `id`, `removed_at`, `state`, `plan_sha256`, `approved_contract`, `controller_sha256`, `original`, `owner_directory`, `owner_source`, `owner_plan`, `predecessor`, and `trials=['glm-official-load']`. `original` is the predecessor's actual replacement. State and all stable bindings are checked throughout preparation, sampling and postflight; recovery beginning causes failure.

## Receipts the future controller must actually consume

- `attempt.json` and `stopped-journal.json` bind the actual owner PID/start identity, owner source/plan, journal ID and removal time. The attempt precedes preparation.
- Only failure strictly before the durable root/phase spawn intent writes `terminal.json` with `NO_CHILD_STARTED`. There is no fabricated child or release. Empty `children.json` and an empty phase directory can exist in that case.
- After intent, an exception without a returned Popen handle is unknown: retain intent, emit FAIL, **no no-child or release proof**. Recovery must refuse until a separately authorized reconciliation establishes what happened.
- A returned handle is immediately recorded in `phases/glm-official-load/child.json` and `children.json`, initially with `identity:null` and `ownership='direct-Popen-returned'`. Identity-read failure still triggers cleanup of that exact handle.
- Known-child cleanup retains `child-absence.json` with actual `ps` command/rc/stdout/stderr, `reap.json`, phase `release.json`, updated `children.json`, and root `release.json`. Recovery independently checks these hashes, actual wait status and current exact-PID absence, including when numerical/load work failed. Identity-null is not proof of absence.
- Cancellation is shielded until the bounded cleanup and durable root terminal/release/result writes finish. Restore the persisted service in the **external driver's `finally`**, not inside this owner.

The future restoration adapter must use these real paths and source bindings. Do not point the old Qwen/controller parser at a fabricated directory layout. Its own independent restoration verdict is separate from this owner result.

## Inventory and physical observation scope

`GLM-model-runtime-inventory-v1` has exactly `schema`, `snapshot`, `download_receipt`, `payload_scope`, `shards`, `runtime`. Each `shards[name]` and `runtime[canonical_absolute_path]` is `{sha256, stat:{bytes,mtime_ns,inode,device}}`. The 62 shard names, sizes and original SHA values must agree with the fixed download receipt and metadata manifest (203,992,076,296 bytes). The only accepted payload scope is:

```text
retained-download-full-SHA256-and-registered-stat;no-new-payload-hash
```

Every shard stat is rechecked. Supplied runtime files are freshly hashed before/after, with the actual resolved inner/outer Python, listed GLM/vendor sources and three MLX native libraries required. Listed model metadata is freshly hashed. This is a supplied inventory, not a complete dependency closure or fresh 204GB payload hash. A different payload policy requires explicit new preparation evidence.

The owner preserves raw successful **and failed** service/process/power/thermal checkpoints. Legacy `ps args` is parsed, then retained as SHA/byte count to avoid recording unrelated credentials; native titles and exact command/error results are retained. This inventory detects the declared known inference processes, not every possible GPU user.

Two owner-PID physical observations precede admission. After Popen, the pinned physical guard samples the actual child PID; its cold40GiB/system240GiB ceilings, pressure1 and zero swap/Swapouts growth gates remain unchanged. All intervening checkpoints count toward its at-most-five-second observation gap. A slow probe/checkpoint or unreadable just-exited child fails closed. Sampling and termination do not impose a hard allocation cap, and no independent child peak history exists after the last live sample.

`complete.json` can therefore claim only a short observed official-load lifecycle diagnostic. `full_model_correctness_qualified`, `performance_qualified` and `physical_resource_safety_qualified` remain false. Independent raw resource and restoration audits are still required before any final closed-cycle claim.


## New source-only cycle wiring

The new controller accepts only an explicitly supplied future archive pair,
`restored-journal.json` and `restoration-audit.json`, under the experiment archive
root. Both exact hashes are required; the journal must use the frozen twelve
route formal phase protocol and the independent restoration audit must pass.
A previous experiment may have failed. Its service must actually be RESTORED.
There is no guessed future PID or blanket acceptance of another lifecycle.

The original stop/recover code receives private dependency dictionaries. The
new `load-worker` argv points to the actual GLM controller, and strict single
child validation rejects nested spawn exceptions. Restoration still reloads
the original persisted **Qwen service**; that is distinct from GLM experiment
preparation. No Qwen 157-file numerical preflight is called for GLM.

The following are instructions for **after** the current formal cycle closes
and its real archive/audit hashes exist. They have not been executed. Use the
same explicit outer Python entry for input preparation, registration, owner and
controller, so the recorded runtime set includes the actual interpreter. The
invocation preserves a venv entry; the inventory uses resolved file identities.

```text
OUTER_PYTHON -I -B bench/m5_ultra_256gb/glm_flash_cycle/prepare_inputs.py --run
  --prompt ABS_PLAIN_UTF8_PROMPT --prompt-sha256 ACTUAL_PROMPT_SHA
  --outer-python ABS_OUTER_PYTHON_ENTRY --out NEW_INPUT_DIRECTORY
```

This creates `probe-plan.json`, `model-runtime-inventory.json`,
`model-metadata.json`, and `preparation.json`. It freshly hashes only listed
source/runtime files and fixed small metadata. It retains the original 62
payload SHA values and reads current shard stats; **it does not rehash 204 GB**.
The supplied runtime set includes listed official GLM/vendor sources, resolved
inner/outer/current Python binaries and three MLX native libraries. It is not a
recursive dependency closure. No backend package is imported. Prompt IDs are
first obtained by the real official tokenizer during the future probe; a plain
prompt exceeding 128 tokens will fail the original probe contract.

```text
OUTER_PYTHON -I -B bench/m5_ultra_256gb/glm_flash_cycle/preregister.py --run
  --probe-plan NEW_INPUT_DIRECTORY/probe-plan.json --probe-plan-sha256 ACTUAL_SHA
  --verification NEW_INPUT_DIRECTORY/model-runtime-inventory.json --verification-sha256 ACTUAL_SHA
  --predecessor-journal ABS_ARCHIVED_RESTORED_JOURNAL --predecessor-journal-sha256 ACTUAL_SHA
  --predecessor-audit ABS_ARCHIVED_RESTORATION_AUDIT --predecessor-audit-sha256 ACTUAL_SHA
  --child-limit-bytes EXPLICIT_AT_MOST_240_GIB --interval-seconds 1 --probe-seconds 5
  --python ABS_OUTER_PYTHON_ENTRY --out NEW_REGISTRATION_DIRECTORY
  --owner NEW_OWNER_DIRECTORY --journal NEW_COLD_JOURNAL_DIRECTORY --analysis NEW_AUDIT_JSON
```

All four outputs must be distinct and have no ancestor overlap. The explicit
child ceiling is a reviewed observation policy, not a conclusion that the
weights or model fit. Original cold40/system240 GiB gates, pressure1, zero
swap/Swapouts growth, owner1830/preparation300/postflight120 and TERM15/KILL15
budgets are unchanged. The registrar generates `owner-plan.json`,
`commands.json`, `source-manifest.json`, and `plan-build.json`; failure preserves
`plan-build-failure.json` in that new directory. It does not write a success
receipt or invoke any command.

Next, root executes **only the saved `controller_prepare` command**, a real
read-only service preparation. Only after that succeeds:

```text
OUTER_PYTHON -I -B bench/m5_ultra_256gb/glm_flash_cycle/preregister.py
  --seal-prepared NEW_REGISTRATION_DIRECTORY
```

That separately writes `receipt.json` with `owner_plan_sha256`, `source_pins`,
and the **actual** `prepared_journal_sha256`. A missing/non-PREPARED journal,
changed plan/source or appeared owner is rejected. The eventual root driver
uses the saved `controller_stop`, then the saved owner command including
`--run`, and invokes `controller_recover` in `finally`. The new `execute.py` is also source-only and requires an explicit `--run`; it reads the sealed commands and restores in `finally`. Unknown Popen outcomes cannot use
`--no-experiment`; only proven non-start permits that existing recovery flag.

`audit_restore.py --journal DIR --out NEW.json` independently reuses the original
retained raw recovery parser and binds the GLM owner/phase/ps/reap paths. It may
report restoration PASS even when GLM load failed. It does not fabricate a live
`process(None)` check: owner absence is explicitly attributed to the pinned
controller's receipt, while child absence has actual retained ps bytes.

`audit_closed.py --journal DIR --plan-sha256 SHA --out NEW.json` first requires
that restoration proof, then independently reparses every retained Apple
footprint and command output, checks sample cadence/PID/lifetime, original
source/model/runtime proofs and the four observed token/count/reset streams.
It freshly rechecks the supplied runtime set, but no model payload. It does not
load a tokenizer or re-decode text. The legacy process argv output was
purposefully redacted by the original owner: its SHA/size and matcher receipt
are retained, but that portion cannot be independently reparsed. All full-model,
chat-quality, physical-safety and performance qualification flags remain false.
Missing terminal evidence is INCOMPLETE; altered or contradictory evidence is
FAIL. No raw sample can establish an unobserved inter-sample or final child peak.

Prepared additional tests: `test_controller_cpu.py` (9),
`test_preparation_cpu.py` (6), `test_audits_cpu.py` (8), `test_execute_cpu.py` (10). These 33 fake tests are
**source-only and unexecuted**. They target empty pre-intent output, unknown
Popen, known-PID identity-read failure, raw four-field absence, output/interpreter
binding, saved command SHA, non-mutating source maps, explicit predecessor
hashes, no payload hashing, raw corruption and EOS-at-cap/immediate-EOS counts.
Only AST parsing has been performed during the active GPU timing interval.


The eventual root execution entry, only after actual read-only preparation,
seal, review and commit, is:

```text
OUTER_PYTHON -I -B bench/m5_ultra_256gb/glm_flash_cycle/execute.py --run
  --registration NEW_REGISTRATION_DIRECTORY --out NEW_EXECUTION_DIRECTORY
```

The execution directory is also required to be fresh and not overlap any of the
four registered outputs. The driver verifies its own source pin, source closure,
exact commands (including the GLM owner `--run`), owner-plan SHA and original
PREPARED bytes before any subprocess. A normally waited owner that never made
its directory can use explicit no-experiment recovery; an exception with unknown
launch outcome cannot. Errors and the final state are retained, including a
failed recovery. The controller's independent absence/reap gate remains the
service mutation authority; the driver does not manufacture it. The driver can
only attempt recovery in its `finally`; forced process termination, unreadable
journals or unresolved child ownership can require manual reconciliation.
