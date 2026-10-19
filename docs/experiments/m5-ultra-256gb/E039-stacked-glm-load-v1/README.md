# E039: stacked GLM official load — failed diagnostic, restored service

Actual driver exit was 1: stop 0, owner 1, recovery 0. The journal is RESTORED.
Independent restoration audit PASS (exit 0); closed diagnostic FAIL (exit 1,
load diagnostic did not complete). The replacement is PID 52818, PPID 1,
Mon Oct 19 09:22:08 2026, omlx-server. Restoration does not qualify the GLM load.
The original execution-final.json is copied unchanged, including its original
observed_diagnostic_complete field.

The actual E036 58-shard stacked checkpoint entered the official oMLX loader.
At physical sample 0018 the unchanged zero-swap-growth gate failed: reported
swap bytes increased from 1,978,925,056 to 1,980,172,861 (+1,247,805 bytes,
about 1.19 MiB), and cumulative Swapouts increased 782,442 to 782,518 (+76).
These are whole-system counters, not process attribution. Child footprint was
148,261,492,112 bytes and system footprint 171,148,001,280 bytes, below the
220/240 GiB ceilings; pressure was 1. The observed failure is the swap gate,
not evidence of an allocation OOM. No memory gate was relaxed.

The original plan had four plain requests capped 8/32/32/32. Actual completed
requests: **0**; token batch events: **0**. There is one official_vlm_load:before,
no official_vlm_load:after or after_pool_load, no loaded-engine receipt, no
probe complete.json and no cooperative shutdown receipt. The shutdown-start
event after cancellation does not prove loader return or successful owner await.
The owned child exited -9 and the complete direct-child/read-command reap chain
was retained. Monitor read commands settled; the diagnostic remains FAIL.

The E036 raw-byte proof and independent header proof remain valid provenance;
this attempt establishes neither loaded-model memory fit nor inference
correctness. It contains no mini-engine GLM run, speed, HTTP result or Pareto
point. The original official loader, remaining MTP sanitize and gate/up fusion
were not replaced by another engine or offload fallback.

The archive retains owner, journal, exact execution and registration, both
saved audit commands/outputs/exit codes, raw resource observations, input
inventory, explicit registered text sources, and small linked E036 proofs.
The registered model-directory alias is retained only as literal target/lstat,
never followed. No weight payload is copied or rehashed. Every tar member and
source is reread and SHA-verified; manifest.json maps each member to its source.
There was no failed archive attempt before this archive.

See summary.json for unchanged FAIL/PASS statuses, partial-observations.json for
actual request/load boundaries and raw counter deltas, and archive-verification.json
for retention integrity. File-backed cache versus MLX allocation overlap remains
a next-experiment hypothesis; these global samples do not prove its cause.
