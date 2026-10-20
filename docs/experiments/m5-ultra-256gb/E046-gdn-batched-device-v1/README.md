# E046 — B2/B8 local GDN device canary

The default-off batched GDN candidate passes its **local real-activation byte canary**. The producer completed 36 calls; the independent saved-byte auditor passed 140 raw files, 108 complete output/state pairs, 48 continued-state pairs, 80 full-vocabulary greedy-row checks, and 324 device comparison array receipts. This does not qualify the candidate full model, compiled graph composition, HTTP behavior or speed, and does not diagnose E045's C8 token variance. The formal engine frontier is unchanged.

The fixed cases are B2/B8 × early/late histories (starts 1 and 129), GDN ordinals 0/24/47, and three consecutive calls per candidate state chain. Activations come from the native batched full model; each candidate chain starts independently and then consumes its own previous state. Original BF16 output/conv and FP32 S bytes are checked. B3–7, row removal/refill/permutation and candidate full-model rollout remain outside this result.

## Actual closure and saved resource review

Root's device invocation `2a413a`, session 16488, completed as `deea7d` with exit 0. The original producer elapsed time is 34.993563 s; the complete outer cycle is 39.526630 s. These are qualification-run durations, not inference timing measurements. Original owner `PASS_CAPTURE_ONLY`, outer `PASS_CAPTURE_AND_RESTORE`, pending-audit fields and the historical `E045-config-driver-result-v1` schema are preserved verbatim.

The independent numerical audit was run by root as tool `edb9ec`, session 12053, completed by `acfa34`, exit 0. The reported stdout was `{"status":"PASS","canary_qualified":true,"raw_files":140}`. No separate original numerical-audit stdout/stderr files were retained. `execution-provenance.json` explicitly attributes those observations to root; it stores the actual relative-path argv separately from the prepared absolute-path argv and does not invent an independently retained log. The actual saved numerical result is included in the archive.

The independent resource review was actually run offline with exit 0 (`resource-command.json`, logs, and `resource-execution.json`): **PASS_SAVED_OBSERVATIONS**, errors empty. It reparses raw footprint/pressure/swap/vm_stat and metadata-probe stdout, checks source copies, and binds numerical PID 87235 to owner PID 87217 and the original lease. It verifies the numerical-done → final physical/metadata sample → hash-bound exit acknowledgement → actual wait/reap sequence, rather than assuming that an exited process was successfully sampled.

- 13 physical and 13 metadata observations, plus the admission baseline.
- Maximum observed system footprint: 201,556,377,600 B. Child current: 28,110,909,200 B; child auxiliary historical peak: 28,744,347,504 B. Process/system accounting is not added together.
- Maximum observation gap: 3.323250 s; admission-to-first gap: 2.107868 s. Observed ceilings remain 64 GiB child / 240 GiB system.
- All pressure=1, thermal=0 and AC; swap growth=0 B, Swapouts=807478 throughout.
- Saved lease closed with return code 0, original oMLX PID 52818 and identity retained, and saved before/after three-endpoint health contracts pass.

These are saved observations, not a fresh live health check, hard allocation cap or bound on inter-sample peaks. The complete raw process-inventory text and repeated start-time probes were not retained; their source-bound saved rows have narrower evidence scope. No model/GPU/service operations or tests were rerun for this resource review or archive.

## Retention

All **140 original tensor files / 1,777,163,588 bytes** remain under `.artifacts/m5-ultra-256gb/gdn-batched-v1-numerical`. `raw-file-references.json` stores their absolute/relative paths, observed sizes, modification times and original audited SHA values. Descriptor/sidecar equality is bound to the completed numerical audit's captures, event ledger and complete-result hashes. The metadata archiver does **not** reread tensor payloads; a referenced digest is not presented as a new payload scan.

The archive includes execution, owner, lease, numerical metadata/sidecars, source snapshots, numerical-audit result, original CPU/preparation evidence, and the independently executed resource review. All original outcomes and pending fields are retained. No selected directory was missing. Runtime binary/model weight payloads are not copied; original pin references and verification receipts remain.

`archive.py` created `evidence.tar.gz` and independently reread every compressed member, checking size and SHA against the manifest and original source: **402 members / 3,381,113 uncompressed metadata bytes**, all PASS. Archive size is **649,500 bytes**.

- Archive SHA256: `ec41ba39432f7a0ae3850eddd9f3b43fb9353dce930b712d58d661cf6e806a95`
- Manifest SHA256: `fe889c346ecbd346d67706eee676d0fdac0a4645a16daeac8bc8dd3c92ec3150`

`summary.json` separates numerical scope, resource review and original closure. `archive-verification.json` covers archived-member integrity; `document-integrity.json` covers the final archive-side files. Integrity PASS never upgrades an experiment's qualification scope.
