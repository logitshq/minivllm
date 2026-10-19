# Run the existing C1 profiler with one direct owner

2026-10-19. Read-only source review; no new execution scaffolding, device run or performance claim. The GLM v3 cycle is owned separately by the root agent. Its future restoration must not be inferred from E033 or from an owner exit code.

The smallest useful next experiment is **three fresh C1 servers: off, runner, full**, each with eight warm and eight diagnostic D8 requests. Keep the E033 combined-ON model/configuration and original client bodies fixed. This produces 48 total requests and a host submission/delivery profile; none is a new Pareto point. The current C1 deficit is roughly 4.77% versus the prior oMLX baseline. This diagnostic should decide whether host work can explain that gap before adding more wrapper optimizations.

## The owner seam is smaller than the existing glue suggests

`route_http_formal/owner.py::run` already has the required direct-Popen ownership. Its nested `spawn`, `probe`, `release`, no-Popen proof, settled callback cancellation and outer `finally` are reusable. Its nested `trial_one` also already works with the capture API: `system.formal_backend(module)` returns `module.System()`, and both the formal runner and `c1_pipeline_profile/capture.py` implement `run(spec, out, checkpoint=..., backend=...)`. Supply the real capture module and keep `trial_one` unchanged. The existing tracked backend then owns the actual server PID, and the existing checkpoint attributes memory to that PID.

Therefore **do not insert `c1_pipeline_profile/owner.py::run_mode` inside that owner**. Its spawn/reap glue duplicates the callbacks already present in `trial_one`. `prepare_modes` remains useful for assembling the three real specs, while `accepted(result, mode)` can validate the drained capture result after the ordinary inner/outer release checks.

The new owner protocol needs these changes only:

1. `validate_plan` accepts exactly `trials=['off','runner','full']`, their three pinned capture specs, the actual predecessor, and the complete executable/runtime/model pins. Reconstruct every capture spec with `capture.validate`; use one immutable E033 C1 combined-ON formal input for all three. The existing `full_prepare` still performs the original native runtime verification, full model hashes and adapter copies before and after the cycle.
2. Keep the owner work bound derived from the actual capture deadlines; a three-mode run is not the old twelve-trial bound. Keep preparation and cleanup budgets. Preserve the existing signal/unknown-launch behavior.
3. Replace the **inline formal-result acceptance and `point.json` publication block** after `trial_one` with `profile_owner.accepted(result, phase)`, actual spec/mode equality, and a `diagnostics.json` entry referencing the capture `complete.json`, both raw groups, raw-count checks and server closure. Keep the preceding actual PID and inner/outer reap checks. Do not fabricate `arm`, `repeat`, a 64-request point or `count_eligible` formal result.
4. Give the owner terminal a diagnostic schema and `diagnostic_modes_complete` field; retain the actual children, planned phase list, errors and release records. Numerical/performance qualification remain false.

A source-pinned adaptation can retain the frozen run algorithm without copying its entire file: bind its dependencies into a separate real namespace and replace only those explicit acceptance/terminal blocks with structure-checked source adaptation. Emit the resulting adapted source for review and source binding. There is no need to patch the imported donor dictionary, change a donor's `HERE`/`__file__`, synthesize old-schema results, or rewrite any completed experiment. If avoiding source adaptation is preferred, extract the run body once into a versioned shared phase runner; that is a larger change and is not on the shortest path to this diagnostic.

## Reuse recovery; register the actual three phases

The reusable recovery algorithms already accept a supplied list:

- `gdn_graph_combined_formal/controller.py::registered_trials` returns `plan['trials']` after checking the real owner plan and journal.
- Its `_bound('phase_reaps', {'registered_trials': ...})` uses the frozen donor's ordered **observed prefix** of that list; it does not require twelve children. Preserve first/late no-Popen, known-null identity and unresolved intent behavior.
- Bind the same existing `stop`, `reaped` and `recover` functions to the new `authorize`, `registered_trials` and `phase_reaps` callbacks. Reuse `monitored_load`, raw service identity/resource checks and the actual load-worker entry. No new stop/restart algorithm is needed.
- Use the existing `owner_plan`, `prepare`, `authorize` and `source_pins` shapes with the new diagnostic schemas/source references. Their plan contains the real three phases. Do not reuse `cold_cycle_v3`'s single-child owner: that would make the model server a grandchild and would require different resource/reap evidence.

`predecessor` must bind the **actual completed GLM v3 restoration** once its archived bytes exist: `GLM-cold-load-journal-v3`, the exact GLM controller/owner/auditor hashes and registered single phase, RESTORED status, independent restoration PASS, replacement identity, settings/model metadata, raw log binding, successful load-worker return, and the GLM monitor's settled read-command/reap proof. A failed GLM experiment can still be a valid predecessor if its independently audited restoration succeeds. An open `LOAD_INTENT`, unknown child, unjoined probe or absent archive is not an acceptable predecessor. Preserve the existing explicit false claims for exact runtime signature and warm KV restoration.

The new restoration audit can reuse `gdn_graph_combined_formal/audit_restore.py`'s `binding`, `owner_reap` and raw restoration helpers, changing only the real module/source/schema bindings and owner terminal schema. These paths already compare `data['trials']` with the owner plan and call the controller's actual `phase_reaps`. Retain exact raw `ps`/port receipts and each adapter copy. It does not need a new performance auditor.

Use one explicit driver with owner CLI preparation **before stop**, then stop, direct owner, and recovery in `finally`. Reuse the proven unknown-launch guard: recovery cannot be based only on an absent owner directory. Source-seal and commit the actual plan/commands after GLM closure and before stopping the replacement service.

## Keep implementation proportional to the experiment

A realistic adaptation is about **250–350 new executable lines total**: approximately 90–120 for owner plan/spec/acceptance bindings, 80–110 for the controller and exact GLM predecessor, and 80–120 for restoration binding plus preparation/driver plumbing. The existing roughly 370-line owner run algorithm and stop/recover algorithms remain reused. Focused adapter tests are additional. A fully runnable, source-bound cold cycle is not honestly a sub-150-line change; a sub-150-line isolated helper would only create another unfinished seam. No such helper was added by this review.

The offline analysis can be a short one-use script over the existing raw captures. First report off→runner→full request latency overhead and confirm all 16 DONEs, natural exit0 and profile closure per mode. Then inspect the retained event intervals: `runner_step`, decoder/compiled-call submission, existing async-eval/eval, worker scheduling, queue delivery, token decoding and HTTP writing. Calculate interval coverage/overlap, not a sum of nested spans; existing eval wall time is not exclusive GPU time. Separate warm/graph creation from measured decode.

If uncovered host gaps or repeated root handling are material, test the already prepared default-OFF `ahead_submit_once` and `long_decode_onepass` changes, one at a time, with numerical qualification followed by unobserved HTTP pairs. If submission/delivery is mostly overlapped, stop polishing wrappers and prioritize **weight reuse per committed token**: rowwise B1 verifier equivalence first, then actual verifier cost and accepted-token yield. Layer-major ordering is a hypothesis; extra prefix/KV materialization may erase any cache benefit. Do not predict a large speedup from the source arrangement.

For the eventual per-batch claim, C1 and C8 endpoints are insufficient. After choosing a winning path, measure C2/C4 and the next admitted batch sizes at the same model/quantization/request distribution; report throughput and per-user interactivity/latency separately. Connecting endpoints is a visual guide, not evidence of wins at the unmeasured batch sizes.
