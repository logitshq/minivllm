# GLM official load: concrete owner and registration boundary

2026-10-17, source-only. This supplements [owner-next.md](owner-next.md). No tests, backend imports, live process probes, model hashes, downloads, or service actions were performed. Nothing here is a registered execution plan or permission to claim that GLM fits.

The next implementation should be **one GLM owner and one direct `load_probe.py` child**, phase `glm-official-load`. Reuse the existing service stop/recovery algorithms and their raw evidence, but do not run a Qwen preflight or hide a second child behind a launcher.

## Existing interfaces that can be retained

| Interface | Actual reuse |
| --- | --- |
| `load_probe.py:130–181,585–602` | Existing `GLM-official-load-plan-v1`, external admission, fixed oMLX interpreter, offline environment, original official EnginePool route, four requests with caps `[8,32,32,32]`. The direct command **requires `--run` and both admission arguments**. |
| `physical_guard.py:85–185` | Retain original footprint JSON and five command results; two cold samples of the owner PID; then samples of the actual returned child PID. `cold_admission` and `observe` already enforce physical byte units, pressure1, zero swap/Swapouts growth, system ceilings and an explicit process ceiling. |
| `physical_guard.py:187–267` | `reap_owned` consumes the real `Popen` object and checks `wait` plus strict raw PID absence. `diagnostic_report` keeps successful shutdown, resource observations and reap separate; it is not an independent raw-resource audit. |
| `pareto_realistic/qualify_absent_owner.py` | Pinned stdlib durable-write and journal-lock helpers can be reused. The new owner must still bind the actual GLM plan, journal, owner source and PID. |
| `resources_v2.py:9–51` | Process inventory includes an exact supplied PID even when its executable is Python. Use no suspended-PID exception after cold removal. The recognized-process inventory remains narrower than proof that all GPU users are known. |

The existing `System.read` in `qualify_absent_owner.py:177–184` drops stderr. It cannot directly provide `physical_guard.reap_owned`'s required `{command,returncode,stdout,stderr}` absence receipt. A small GLM backend method should retain all four fields; do not insert a fabricated empty stderr.

## Why the shared owner is not a drop-in launcher

`cold_cycle_v3/owner.py:83–95` executes fixed Qwen/E024 preparation and emits only `--plan/--plan-sha256/--out`. Feeding it the GLM probe would print the contract and exit0, without loading. Its `load_resources(numeric)` and original model inventory are also Qwen-specific.

`cold_cycle_v3/controller.py:107–147` requires the fixed inherited Qwen owner plan and child `qualifier_sha256/deadline_s` fields. GLM instead has `deadlines` and a runtime-issued admission. Its first predecessor is currently a specifically sealed route-HTTP migration, then same-version cycles; an arbitrary restored oMLX journal is not accepted. Do not patch `HERE`, pretend that a GLM source is an old source, or pad the probe plan to bypass this mismatch.

The smallest correct change is a **model-specific plan/prepare/child-command adapter** plus explicit controller/restore bindings to that new owner. The service being restored can remain the original persisted Qwen service. Only the experiment's model preparation and admission change. Reuse service methods only after checking their module-global dependencies and ensuring the monitored recovery worker runs the actual new controller entry. This boundary should be agreed before writing another full controller copy.

## Proposed small owner-plan contract

This is an interface proposal, not a JSON file with placeholder SHA values:

- `schema: GLM-cold-load-owner-plan-v1` and one fixed phase `glm-official-load`.
- `execution`: actual owner, probe, physical guard and shared helper path/SHA refs; exact fixed oMLX Python and environment.
- `probe_plan: {path,sha256}`: the unmodified `GLM-official-load-plan-v1` with plain UTF8 prompt, original snapshot, fixed download/header receipts and required small source pins.
- `verification`: GLM metadata/62-shard inventory, server runtime/source and outer stdlib Python inventories. Each manifest explicitly states its coverage and when payload SHA verification occurred.
- `physical_policy`: the exact result shape of `physical_guard.policy`; `child_limit_bytes` must be chosen explicitly, not inferred from checkpoint size. Work, interval and aggregate probe bounds must be finite and within that helper's limits.
- `cold_restart`: actual reviewed controller/source, authorized contract, **future closed immutable** predecessor journal/audit refs and separate fresh journal path.
- `outputs`: new owner and probe directories; `budgets`: child work at most1800s, finite preparation/postflight budgets and TERM15/KILL15. Do not charge a fresh 204GB payload rehash to a silently enlarged model-execution deadline.

The shard inventory is exactly the fixed revision `fa04cbc3622b16aa7475404060b05e506d81bd4e`: 62 files,203,992,076,296 bytes. The original download receipt and header inventory are already pinned by the probe. Revalidate their exact agreement with the metadata manifest and every current file's size/stat. If a future policy reuses the original full hashes, label that as **historical full hash plus current stat**, not a fresh payload hash. If fresh full hashing is required, perform and retain it in explicit preparation before the cold work budget. This task did neither.

The old oMLX owner inventory is a useful source of runtime file names, not a callable GLM verifier: `omlx_d512/owner.py:111–156` also binds the Qwen target and HTTP/tokenizer outer packages. The new minimum includes all `load_probe.REQUIRED_SOURCES`, the actual resolved Python binary, relevant package metadata and native MLX libraries; supplied coverage must not be described as a complete dependency closure without proving it.

## Concrete execution order

1. Registration creates immutable **probe plan**, **owner plan**, source/verification manifests and a plan-build receipt. It does not create admission, STOPPED evidence or a prepared-journal SHA. Actual controller read-only prepare later supplies the latter.
2. After a real authorized stop, the owner holds the journal lock long enough to verify STOPPED stable identity and durably create its attempt. Verify exact removed PID/job/8001 absence, AC/thermal and no recognized foreign inference process. Unknown process evidence rejects admission.
3. Run two `physical_guard.measure(..., owner_pid)` observations,4–60s apart, and `cold_admission`. These are additional GLM physical prerequisites, not a replacement for the service and process checks.
4. Create the three real admission evidence refs: `cold_release`, `resources`, `model_runtime`. Issue `GLM-official-load-admission-v1` with actual owner PID, probe/plan SHA, snapshot, exact output and finite validity. Under a short journal lock, recheck STOPPED and write durable spawn intent immediately before `Popen`.
5. Invoke exactly:

   ```text
   <fixed oMLX venv Python> -I -B <load_probe.py> --run
     --plan <probe-plan.json> --plan-sha256 <actual SHA>
     --admission <owner/admission.json> --admission-sha256 <actual SHA>
     --out <new probe directory>
   ```

   Keep the returned handle first and immediately write its real PID/PPID binding; a failed identity read may retain `identity:null` with known direct-Popen ownership. A spawn intent without a returned handle is unknown, never `NO_CHILD_STARTED`.
6. Immediately sample the actual child PID and repeat `measure→observe`. Check PID/start identity around each sample and continue service absence/resource checks. Existing `wait(5s)` followed by multiple independent ten-second resource commands is incompatible with this guard: `observe` bounds the gap from the previous sample's end to the next start. Use an explicit monotonic schedule; all intervening checks consume that gap. Overrun or unreadable live/just-exited PID is a failed observation, not permission to omit a sample. Probe duration is separately bounded and inter-sample physical peaks remain unknown.
7. On success, failure, timeout or cancellation, finish known-child TERM/KILL/wait/strict-absence cleanup, shielding that bounded cleanup from repeated cancellation. Keep unknown-Popen recovery blocked. Only failures strictly before durable spawn intent can issue no-child terminal proof.
8. Preserve child return code, actual complete result, shutdown and four caps. Independent resource review reparses original command/footprint bytes, not only `diagnostic_report`. Then the external driver executes recovery in `finally`, and the restoration auditor checks the new owner's real receipt shape. Experiment failure is compatible with successful, independently verified restoration.

## Minimal tests before any real launch

Pure fake cases: exact GLM argv includes all admission arguments; Qwen-only/short shard inventory rejected; absent or expired admission rejected; real returned PID is the sampled/reaped PID; gap/pressure/swap/physical violations abort; repeated cancellation still reaps; unknown spawn cannot produce no-child proof; describe-only exit0 or missing/incorrect complete cannot pass; diagnostic failure can still permit restoration after exact reap. Registration must reject unclosed or mismatched predecessor refs.

The only successful scope would be an observed short official GLM lifecycle diagnostic. No HTTP, D512/C8, full-state numeric, throughput/latency, guaranteed physical fit, offload, lazy-load or altered quantization claim follows. The final observed live footprint is not an independent physical-peak history after process exit.
