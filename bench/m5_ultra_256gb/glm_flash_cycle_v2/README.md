# GLM cold-load cycle after GDN formal restoration

No v2 registration, service preparation, model load or GPU operation has
occurred. The new 16-test fake CPU module has now run twice: first 15 PASS and
one negative-fixture ERROR, then 16 PASS after a fixture-only correction. The
original production algorithms and all eight adapter production sources were
unchanged. These mechanism checks cannot establish model correctness or fit.

The experiment remains the original official GLM5.3Flash `glm5_next` load probe,
with one direct child, the unchanged plain prompt contract, normal EOS and caps
8/32/32/32. Recovery restores the original persisted Qwen3.8-27B service. The GLM
experiment does not use Qwen model preflight or claim to restore warm KV/runtime
state exactly.

## Exact change and reuse

`_reuse.py` verifies seven frozen original GLM sources and binds their original
Python code objects into the actual v2 module dictionaries. Imported helpers and
classes retain their original identities. The original modules are not patched;
no source paths or evidence are impersonated. Both original and adapter sources
are retained in the owner/registration source inventory.

- `owner.py` retains original direct Popen, physical admission/monitoring,
  prepare/postflight, cancellation and strict reap algorithms. Its source,
  controller path and frozen-source copies refer to this real namespace.
- `controller.py` changes only the predecessor allowlist and namespace bindings.
  It requires an explicit archived `GDN-prework-http-formal-cold-journal-v1`
  RESTORED journal and its independent restoration audit, matching the pinned
  GDN controller `38a6e904…4dcd35` and auditor `8ece5748…21fc0`. The twelve GDN
  trial names describe predecessor history; this experiment has only
  `glm-official-load`. A failed predecessor experiment may have valid restored
  service evidence; missing or incomplete restoration is rejected.
- `audit_restore.py` and `audit_closed.py` reuse the original raw physical,
  checkpoint, four-field child-absence, reset/token/count and shutdown checks.
  Only their actual v2 source entry bindings change. The original limits and
  negative conclusions remain: no full-state reference, quality, HTTP,
  performance, fit, or hard memory-safety qualification.
- `preregister.py` and `execute.py` reuse the original registration and
  stop→owner→finally-recover algorithms. The owner receives `--run`; unknown
  launch outcomes never gain `--no-experiment` evidence.
- `prepare_inputs.py` is an optional explicit forwarder to the original CPU
  builder. Its output truthfully names that original builder. The already
  prepared v1 inputs can be used directly; rebuilding them is not required.

The shared journal/owner/result schemas remain the existing GLM v1 protocols;
exact source and path bindings distinguish this new namespace. No old journal is
migrated or edited. `support_pins` is an explicit reviewed source set, not a claim
of recursive dependency closure. The model inventory retains 62 prior full
shard hashes and rechecks file statistics; no new 204 GB payload hash is implied.

## Existing actual CPU inputs

The first successful CPU preparation is archived at
`docs/experiments/m5-ultra-256gb/glm-flash-inputs-v1`. The original live input refs
are:

- `.artifacts/m5-ultra-256gb/glm-flash-inputs-v1/probe-plan.json`, SHA-256
  `700f0a785138a75ebb4554d6eddee7dfbd1375c5b92bd596f8bd2e3e16f9be87`.
- `.artifacts/m5-ultra-256gb/glm-flash-inputs-v1/model-runtime-inventory.json`,
  SHA-256 `bdf025b9fdfd5cb22eb1991c9b3e7c20c73b962a3e86e4306a0af9c33aca58be`.

No actual GDN formal restoration refs are supplied in this source preparation.
Registration must receive both future immutable archived files and their actual
SHA-256 values. It rejects a live journal, non-RESTORED state, incorrect source,
wrong phase list, mismatched raw-log/replacement/worker receipts, or files outside
the archive tree or in different archive directories.

## Explicit future execution sequence

All paths below must be absolute. `GLM_CHILD_LIMIT_BYTES` is an explicitly
reviewed physical observation policy, not a predicted safe model size. Keep the
venv interpreter entry rather than replacing it with its resolved executable.
These commands are examples only and have not been run.

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -I -B \
  /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/glm_flash_cycle_v2/preregister.py --run \
  --probe-plan "$GLM_PROBE_PLAN" --probe-plan-sha256 "$GLM_PROBE_PLAN_SHA" \
  --verification "$GLM_INVENTORY" --verification-sha256 "$GLM_INVENTORY_SHA" \
  --predecessor-journal "$ACTUAL_GDN_FORMAL_RESTORED_JOURNAL" \
  --predecessor-journal-sha256 "$ACTUAL_GDN_FORMAL_JOURNAL_SHA" \
  --predecessor-audit "$ACTUAL_GDN_FORMAL_RESTORATION_AUDIT" \
  --predecessor-audit-sha256 "$ACTUAL_GDN_FORMAL_AUDIT_SHA" \
  --python /Users/ryanli/code/minivllm/.venv/bin/python \
  --out "$GLM_REGISTRATION" --owner "$GLM_OWNER" --journal "$GLM_JOURNAL" \
  --analysis "$GLM_ANALYSIS_JSON" --child-limit-bytes "$GLM_CHILD_LIMIT_BYTES"
```

Registration writes `owner-plan.json`, `commands.json`, `source-manifest.json`
and `plan-build.json`, not a PREPARED receipt. All output paths must be distinct
and non-overlapping. Root then reviews and explicitly invokes the exact saved
`controller_prepare` command (read-only service check). Only after that actual
PREPARED journal exists may root invoke:

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -I -B \
  /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/glm_flash_cycle_v2/preregister.py \
  --seal-prepared "$GLM_REGISTRATION"
/Users/ryanli/code/minivllm/.venv/bin/python -I -B \
  /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/glm_flash_cycle_v2/execute.py \
  --run --registration "$GLM_REGISTRATION" --out "$GLM_EXECUTION"
```

The root receipt binds `owner_plan_sha256`, `source_pins`,
`prepared_journal_sha256` and the actual `outer_python`. The driver verifies the
saved commands, waits the real owner, and attempts recovery in `finally` even
when the owner fails. Owner work/prepare/postflight/TERM/KILL budgets remain
1830/300/120/15/15 seconds; the original probe has its own 1800-second work
budget and unchanged request/shutdown deadlines. Future execution must be
exclusive and supervised by root. After closure, run the saved `closed_audit`
command; a failed probe may still have a separate PASS restoration receipt.

## Actual guarded CPU evidence

`.artifacts/m5-ultra-256gb/glm-cycle-v2-prepared-cpu-v1` retains the first complete
stdout/stderr, result, runner and source snapshot. The damaged external-plan
negative fixture originally replaced the entire plan with an unrelated object;
the real code rejected it with a missing-field error before reaching the SHA
check. The fixture now retains all required fields and adds a changed value,
then verifies the original SHA gate rejects it. No production guard was relaxed.

The separate `glm-cycle-v2-prepared-cpu-v2` retains all 16 PASS results, the prior
failure's exact bytes, source-before/after, old/new tree membership and imported
source snapshots. Both attempts had zero denied real operations and no backend
modules loaded. The guard forbids real subprocesses, sockets, signals/timers,
backend imports and model/device payload opens. Existing old suites were not
rerun. Documentation was updated after the successful run; the tested README
bytes remain in each attempt's frozen-source tree.
