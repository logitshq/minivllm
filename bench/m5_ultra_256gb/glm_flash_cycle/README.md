# GLM one-child owner — prepared source, not registered

`owner.py` runs the existing official GLM load probe as its **direct child**. It does not stop, load, restore or send HTTP to the persisted service. No controller, registration, execution driver, actual admission or success receipt is created here. The adjacent `controller.py` is deliberately absent; `--run` fails closed until a separately reviewed real controller and immutable plan exist.

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
