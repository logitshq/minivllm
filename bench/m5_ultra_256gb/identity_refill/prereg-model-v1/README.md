# E028 full-model qualification: frozen CPU preparation

No model was loaded, GPU run, child spawned or service changed in this stage.
This is preparation, not a numerical PASS or a throughput–latency point.

- [Child plan](qualification-plan.json): SHA `bc4890e6d302141597eae8e0e9347c093307a8492c09f958f42bdb8d1c7aa73c`.
- [Owner plan](owner-plan.json): SHA `02a2cc98d068d037c6ddd5cda24fe2c5be13152013b919b5a94ddcdfdd0290f2`.
- [CPU test receipt](cpu-tests.json): 66 PASS, including 18 new adapter tests,
  7 integration tests and 41 unchanged qualifier/owner tests. Raw stdout/stderr
  is saved beside this receipt.
- [Actual CPU prepare](cpu-prepare.json): 157 original runtime/core/manifest/
  fixture pins and 5 adapter source copies verified; original four MLX variables
  retained. Model weight shards are checked by the child, which has not run.

The private qualifier adapter installs E028 only on each candidate runner before
observers. It reuses the unchanged 37 normal-EOS requests, full logits/raw IDs,
staging/ahead, retirement and model/source checks. Immediately after actual
eligible synchronization and before decode, it retains independent handles to
all 64 layers/128 arrays and uses the existing original-dtype comparator. A fresh
identity branch counter increment is required; stale last_path is insufficient.
D-C8 must exercise at least one such refill and C1 none. Restore removes the
instance hook and all extra model/state references before the next model pair.

The resource adapter keeps the frozen service-absent owner's pressure-1 gate,
1800-second child / 1830-second owner deadline and durable direct-child cleanup.
It genuinely prepares the original E023 plan under e023-input-preflight/ before
freezing the new sources and preparing the new CLI. Existing journals were not
changed; the new journal path in owner-plan.json is only reserved, not created.
Cold removal/recovery authorization is still pending; the old pause/resume
permission does not become cold-restart approval through this plan.

Before any performance qualification, independently audit the new nested base
ledger and refill witnesses. The old E023 auditor rejects the extra comparison
groups and old owner/path bindings; it must not be advertised as compatible.
Then run full-model qualification under the resource owner, followed by separate
HTTP qualification and uninstrumented fresh-process OFF/ON measurements. All
new performance gates remain pending. Candidate/core defaults are unchanged.
