# Future v3 registration — source prepared, not executed

The new `preregister.py` and `execute.py` have not performed real registration or
execution. The earlier E032-bound source snapshot passed seven focused fake CPU
tests on its first guarded run. The current E033 adaptation is source-only and
has not been tested.
The original input builder and lifecycle are reused under explicit source pins.
No new framework, operational bridge, model payload hash or future evidence is
introduced. Existing GLM CPU inputs from `glm-flash-inputs-v1` remain eligible
only if the real owner revalidates their exact supplied hashes and statistics.

E032 ended with an owner-CLI admission error before any model child or request;
its independent restoration passed. It is retained history, not the next GLM
predecessor while retry-v2 is planned. The registrar explicitly requires the E033 target. After the retry-v2 sources
were frozen and its actual formal owner started, the controller's source-only
adaptation selected those exact controller/owner/auditor pins and v2 journal
schema. Its predecessor manifest and dependent auditor seals remain UNSEALED.
No actual E033 result is assumed and no registration can run before closure.
The earlier E032 controller and seven-test source snapshots remain in the
committed CPU archive; the new adaptation has not yet been tested.

Before registration, root must have the actual immutable
`docs/experiments/m5-ultra-256gb/E033-gdn-graph-combined-formal-v2/` pair:
`restored-journal.json` and `restoration-audit.json`. Normal RESTORED, zero-code
recovery worker and independent restoration PASS are required; formal performance
success is not required for safe recovery. Nothing in these scripts creates that
archive or performs the source sealing.

Seal the actual predecessor manifest and its controller SHA, then the restoration
auditor's controller SHA, closed auditor's restoration SHA, and driver's registrar
SHA. Preserve the currently unsealed/tested source history. Review and CPU-test
the two new adapters before operational use. The root records the actual values
and commits the source/registration before stopping the service.

Future explicit commands (absolute paths and actual hashes required):

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/glm_flash_cycle_v3/preregister.py --run \
  --out "$GLM_REGISTRATION" --owner "$GLM_OWNER" --journal "$GLM_JOURNAL" \
  --analysis "$GLM_ANALYSIS_JSON" --python /Users/ryanli/code/minivllm/.venv/bin/python \
  --probe-plan "$GLM_PROBE_PLAN" --probe-plan-sha256 "$GLM_PROBE_PLAN_SHA" \
  --verification "$GLM_INVENTORY" --verification-sha256 "$GLM_INVENTORY_SHA" \
  --predecessor-journal "$ACTUAL_E033_JOURNAL" --predecessor-journal-sha256 "$ACTUAL_E033_JOURNAL_SHA" \
  --predecessor-audit "$ACTUAL_E033_RESTORE_AUDIT" --predecessor-audit-sha256 "$ACTUAL_E033_RESTORE_SHA" \
  --child-limit-bytes 236223201280 --interval-seconds 1 --probe-seconds 5
```

The builder adds the owner's exact dual-lane monitor policy to the original
plan; it does not reclassify the 5-second metadata lane as 1:1 physical samples.
Ceilings stay 220 GiB child / 240 GiB system, zero swap growth, and the original
bounded work/cleanup policy. A new registration contains actual source manifests,
plan and exact commands, but no PREPARED claim. Root then separately invokes the
saved `controller_prepare` read-only command. Only its real PREPARED output can
be sealed:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/glm_flash_cycle_v3/preregister.py \
  --seal-prepared "$GLM_REGISTRATION"
.venv/bin/python -I -B bench/m5_ultra_256gb/glm_flash_cycle_v3/execute.py \
  --run --registration "$GLM_REGISTRATION" --out "$GLM_EXECUTION"
```

Root alone runs these future operations. `execute` preserves the original
unknown-launch guard and always attempts recovery in `finally`; only a proven
non-start can use `--no-experiment`. After actual closure, run the saved
`restoration_audit` and `closed_audit` commands. Failed-but-settled monitoring may
permit restoration without granting the GLM diagnostic a PASS. No model fit,
full-state numerical, HTTP or performance qualification is implied here.

## Root handoff: exact API and late-bound fields

No real registration command has run. Root temporarily allowed a focused CPU
check between formal attempts; all seven tests passed before retry timing. No
further tests or compression will run once root resumes the timing restriction.

| Entry | Arguments / output |
| --- | --- |
| `preregister.py` without mode | Describe only; operational arguments without `--run` are rejected. |
| `preregister.py --run` | All paths in the example are required; four matching input/predecessor SHA arguments are required. Writes fresh `owner-plan.json`, `commands.json`, `source-manifest.json`, `plan-build.json`. Does not start `controller prepare`. |
| `commands.json:controller_prepare` | Root runs the actual adjacent controller `prepare`, with exact owner path/plan SHA and proposal SHA. This is a separate read-only live-service check. |
| `preregister.py --seal-prepared FOLDER` | Accepts no build arguments. Rechecks source closure and actual PREPARED journal, then creates exclusive `receipt.json`. |
| `execute.py --run --registration FOLDER --out NEW` | Checks the exact receipt/journal/source/command bindings and fresh output, then invokes stop → direct owner → finally recover. Empty argv only describes. |
| `commands.json:restoration_audit` | Actual adjacent `audit_restore.py --journal CLOSED --out NEW`; output is sibling `restoration-audit.json` beside the diagnostic analysis JSON. |
| `commands.json:closed_audit` | Actual adjacent `audit_closed.py --journal CLOSED --plan-sha256 OWNER_SHA --out NEW`; requires a distinct analysis filename. |

Root must supply these actual facts **after** E033 closure; nothing discovers or
fills them in automatically:

1. E033 `restored-journal.json` and `restoration-audit.json` exact bytes/SHA. The
   controller verifies the journal schema, twelve ordered phases, actual owner
   source, normal RESTORED state, replacement identity, matching log window,
   preserved settings/model variant and zero return code from recovery's worker.
   A formal numerical/performance FAIL does not by itself prohibit restoration.
2. Set `predecessor-manifest.json` to `status: SEALED`, exact
   `binding: {path, sha256, audit: {path, sha256}}` and the actual twelve `trials`;
   then put that manifest's SHA in `controller.BOUND_MANIFEST_SHA`. The required
   archive basename is fixed to `E033-gdn-graph-combined-formal-v2`.
3. Seal in order: `audit_restore.CONTROLLER_SHA` = final actual controller SHA;
   `audit_closed.RESTORE_SHA` = final actual restoration auditor SHA;
   `execute.REGISTRAR_SHA` = final reviewed registrar SHA. The builder checks all
   these before creating output. A changed source requires a newly reviewed pin,
   not reuse of an old preparation receipt.
4. Supply the existing actual GLM probe-plan and model/runtime inventory refs,
   explicit outer `.venv/bin/python` entry, and fresh mutually non-overlapping
   registration/owner/journal/analysis/execution paths. Preserve the venv entry
   rather than replacing it with the resolved binary. Original inventory/stat
   checks still run; no claim of a new 204 GB model-payload hash is made.
5. After source review and focused CPU validation, run CPU registration, actual
   read-only controller preparation, and receipt sealing; commit these actual
   source/plan/receipt bytes before any `execute --run`.

The new plan is `GLM-cold-load-owner-plan-v3` with exact nine top-level fields:
`schema`, `execution`, `probe_plan`, `verification`, `physical_policy`,
`cold_restart`, `outputs`, `budgets`, `monitor_policy`. Build and receipt schemas
are `GLM-cold-load-plan-build-v3` and `GLM-cold-load-prepared-receipt-v3`; the actual
journal is `GLM-cold-load-journal-v3`. Original four-request caps remain
8/32/32/32. Physical max gap/probe deadline are 1/5 seconds, metadata max gap is
5 seconds, target gap is half the maximum, and worker join budget is 7 seconds.
These are separate observed lanes; neither proves a hard instantaneous memory
limit. Any live/unsettled read command blocks canonical release and restoration.

## Actual focused CPU check after the E032 CLI failure

Whole-module first attempt: **7/7 PASS**, 0.103372875 s, zero
failures/errors/skips/forbidden operations, unchanged pre/post source snapshots.
Command `.venv/bin/python -I -B /tmp/run-glm-v3-registration-cpu-v1.py`, tool
`d143a6`; retained directory `.artifacts/m5-ultra-256gb/glm-v3-registration-cpu-v1`.
It contains complete stdout/stderr, runner, results and all 24 explicit source
snapshots. No controller/System backend was called. Actual argparse entry paths
were exercised with only operational handlers/signals replaced by explicit fakes.

- Generated owner argv is accepted by real GLM `owner.main`; omitting its
  required `--run` is rejected. GLM differs from the failed formal owner's CLI.
- Real controller parsers accept the generated prepare/stop/recover commands and
  reject an added `--run`; real audit parsers accept the saved offline commands.
- Current E032 controller is rejected by the E033-only registrar before output
  creation. Source/predecessor seals are still absent.
- Real driver preflight binds v3 PREPARED bytes and exact generated commands; a
  removed owner `--run` is rejected even after rebinding the command-file SHA.
- Unknown owner launch still attempts finally recovery, without fabricated
  `--no-experiment`. This fake mechanism test is not recovery evidence.

Remaining readiness gates: final reviewed retry-v2 controller/owner/auditor
source pins, actual E033 archived journal/audit pair, corresponding explicit
controller adaptation, all seal values, actual CPU registration, read-only
PREPARED check and sealed registration receipt. None has been fabricated.
