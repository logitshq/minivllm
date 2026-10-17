# E024 serving launch: read-only review

Reviewed the current source only; no tests, hashes, model imports, services or live records were executed/read. No new source-level blocker found in the preparation → single-child owner → recovery connection. This is a numerical serving qualification, not an HTTP or performance run.

## Required real predecessor

Wait for the complete Lithos driver to return and its independent restoration audit to pass. Then use these **new archive files**, not the live journal or a synthesized receipt:

- `/Users/ryanli/code/minivllm/docs/experiments/m5-ultra-256gb/E023-lithos-formal-v1/restored-journal.json`
- `/Users/ryanli/code/minivllm/docs/experiments/m5-ultra-256gb/E023-lithos-formal-v1/restoration-audit.json`

Hash their actual final bytes. The second file is the independent restoration object, not the whole `independent-formal.json`. `archive_formal_v1.py` either copies the separately executed restoration receipt or serializes the formal audit's actual `/restoration` object as a sidecar; that sidecar therefore has its own byte SHA. The archive can be valid after an experiment failure if restoration independently passed. Do not hardcode the replacement PID: `controller.predecessor()` reads it from this actual RESTORED journal, and `controller.prepare()` subsequently checks that it is still the live service.

The v2 controller explicitly accepts the frozen Lithos formal journal schema and controller `5c13010c…73c2` / restoration auditor `41d260e6…d66f`. It checks exact archived bytes, approved protocol, journal/owner-plan/source binding, restoration replacement/log/worker receipts and the persisted-visible service variant. It does not require the preceding experiment's numerical or performance PASS.

## Shortest root sequence after recovery

Use the preparer's defaults: `execute_serving.py` has fixed paths and does not accept custom `--out`, `--journal` or `--owner` values.

1. Finish the pending focused CPU checks/source freeze, then build new plans with the real two archive SHAs:

   ```sh
   /Users/ryanli/code/minivllm/.venv/bin/python -I -B \
     /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/route_decode_graph/prepare_serving.py --run \
     --predecessor /Users/ryanli/code/minivllm/docs/experiments/m5-ultra-256gb/E023-lithos-formal-v1/restored-journal.json \
     --predecessor-sha256 ACTUAL_ARCHIVED_JOURNAL_SHA \
     --restoration-audit /Users/ryanli/code/minivllm/docs/experiments/m5-ultra-256gb/E023-lithos-formal-v1/restoration-audit.json \
     --restoration-audit-sha256 ACTUAL_ARCHIVED_RESTORATION_SIDECAR_SHA
   ```

   This creates `route_decode_graph/prereg-serving-v1/{serving-plan.json,owner-plan.json,source-manifest.json,plan-build.json}` only. It does not create a service journal, owner, analysis directory or final receipt.

2. Execute `plan-build.json.commands.read_only_service_prepare` verbatim. It calls the actual `cold_cycle_v2/controller.py prepare`, checks current service identity/status/settings/model metadata against the archived predecessor, and writes a fresh PREPARED journal. This read-only service check can legitimately reject if the service/settings changed meanwhile.

3. Root creates **new** `prereg-serving-v1/receipt.json` with these actual fields:

   ```json
   {
     "owner_plan_sha256": "plan-build.owner_plan.sha256",
     "source_pins": {"absolute paths from plan-build.source_pins, plus execute_serving.py": "their final actual SHAs"},
     "prepared_journal_sha256": "SHA of the newly PREPARED route-decode-serving-v1-cold-journal/journal.json"
   }
   ```

   These are field descriptions, not valid placeholder values. Recheck the pins before sealing; the driver requires its own file, actual controller, actual owner and outer owner-plan among them. Do not pin receipt.json to itself. Do not use the archived predecessor SHA as the PREPARED SHA. Commit/push the frozen preparation and real receipt before the device run.

4. Launch the sole existing driver:

   ```sh
   /Users/ryanli/code/minivllm/.venv/bin/python -I -B \
     /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/route_decode_graph/execute_serving.py --run
   ```

   It verifies all receipt pins and exact PREPARED bytes, writes execution evidence, then performs stop → direct owner → finally recover. Do not manually stop first: this driver requires the original PREPARED bytes. Ordinary owner/numerical failure still reaches recovery. Unknown launch is deliberately not resolved using an absent directory alone; `--no-experiment` is supplied only when no owner was attempted or its synchronous launch returned, and the owner directory never appeared.

5. After the driver has returned and actual recovery is terminal, create the analysis directory and run the two saved `plan-build.json.commands.{restoration_audit,numerical_audit}` commands, plus the existing resource reparse below. All outputs must be new files. The numerical command takes the **child serving-plan SHA**, not the outer owner-plan SHA or the fixed cold-restart proposal SHA. Keep any failure receipt; do not silently replace or repair it.

   ```sh
   /Users/ryanli/code/minivllm/.venv/bin/python -I -B \
     /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/resource_reparse.py \
     --owner /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/route-decode-serving-v1-owner \
     --out /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/route-decode-serving-v1-analysis/independent-resources.json
   ```

   Add this resource auditor's actual verified SHA to the root receipt before execution. It was already registered by the timing prereg as `5948ccbf7baeb21d1e71d9e8f7169b1bd9f84e65eb492746aba86fe377806c23`; that is a recorded source identity, not a hash recomputed during this review. The preparer currently omits this third command/pin; root can preserve it in the final receipt/root review without changing the frozen lifecycle or exact seven child sources.

## Actual integration and remaining boundaries

- `prepare_serving.py:prepare()` makes an exact five-field child plan with seven child source pins and deadline 1800; auditor/preparer pins live only in the outer plan. It calls real `serving_qualification.validate()` and `controller.run_plan()` before writing the final build receipt.
- `cold_cycle_v2.owner.prepare()` reuses the frozen original E024 adapter and original E023 environment/model/source preparation, records the genuine inherited launch, then substitutes only the new pinned qualifier/plan argv. The actual direct child output is `route-decode-serving-v1-owner/phases/e024-route-serving/qualification`.
- Child 1800 s / owner 1830 s / TERM 15 s / KILL 15 s remain unchanged. The actual owner enforces the pressure-1 resource guard and service-absence checks, retains durable spawn intent, waits/reaps its known child and writes absence evidence. The driver calls recovery with a bounded 3600 s watchdog; this is not an additional numerical time budget.
- The existing restoration auditor independently checks owner reap and recovery physical/log/HTTP-worker evidence. `audit_serving.py` separately reconstructs the original 37-request schedule/full-state comparison ledger and route/parameter receipts. Neither driver exit 0 nor restoration PASS alone grants numerical qualification; the numerical auditor also does not independently certify all owner resource samples.
- `resource_reparse.py:audit()` is already compatible with the new v2 single-child owner: it reads the real children index/release and derives `<phase>:running`, with the same before/after-prepare and postflight labels. It reparses saved pressure/swap/VM/power/thermal and exact owned-PID ps bytes and recomputes the original guards/deltas. No adapter is needed for that scope. It neither binds the experiment plan nor independently proves continuous service absence; combine it with the real restoration/reap audit, retain the owner's absence evidence, and keep its stated inventory/unsampled-interval limitations. The complete raw process inventory was not saved, so inventory entries remain producer receipts and `sole_gpu_qualified` remains false. Old `audit_absent_qualification.owner_resources()` is **not** directly compatible: fixed `child-running`, PID 74384, flat paths and `absence_probes` differ. Do not fake those fields to reuse it.
- Numerical scope remains gates ON / norm OFF, original37 plus 42 parameter calls, E028 OFF, full logits and 128 states. Parameter equality is a device comparison receipt, not offline tensor-byte recomputation. An observed zero discarded-ahead count explicitly leaves that invalidation path uncovered. HTTP/performance remain false.
- The fixed default execution directory `route-decode-serving-v1-execution` must also be absent; the preparer checks prereg/journal/owner freshness, while the driver checks execution freshness. Any failed preparation leaves its partial new directory for inspection.
- The driver gives Python `finally` recovery, not protection from SIGKILL, host loss or unresolvable unknown process creation. Those remain fail-closed with durable evidence; there is no basis to fabricate child absence.

Lifecycle values read from the current manifest (not independently rehashed during the active measurement): controller `8566b06a48596f04d91b9cade472db150f59879b32b05fc894c0977bcfa572a2`; owner `4c2cc6f305de3ecf0c6495e45838baae885efbf7a8f479ed2043dc0ffb67f31c`; restoration auditor `01b01117e1b4dfac346b1231bb0e36528604fcb2e0d54c9e61461d59b88c2dec`. The preparer will verify them when root explicitly runs it after the measurement.
