# E024 route-preserving 52-point matrix preregistration

This is a new numerical diagnostic, with the frozen default-OFF candidate and
the unchanged Qwen3.8-27B NVFP4 target. It grants no HTTP or performance point.
Core compile, fuse, gates compile and norm reuse remain OFF. Physical K/V
capacity remains1536; the candidate chooses callable bound1023 below1024 and
bound1536 at or above1024 without changing the native arithmetic policy.

The52 points are fixed in `preparation.json`:40 combinations of B1..8 and
used257/1023/1024/1280/1536, a3-step uniform crossing1023→1024→1025, a3-step
mixed-length crossing, and6 B1→B8→B1 reuse points at1023 and1024. The fixture
history repeats and truncates frozen D token IDs as a diagnostic extension;
it is not generated output. All points require original-dtype finite and bit
equality for logits plus128 state arrays, and unchanged131-array inputs.
Parameter values are checked by original-dtype producer hash/finite receipts;
handle IDs are recorded without requiring identity preservation.

Every point retains complete device comparison and full-byte hash receipts.
Only ten selected full safetensors snapshots are preregistered: canonical
input at matrix-B1-used257, and canonical input plus both outputs at
matrix-B1-used1023, matrix-B1-used1024 and mixed-used1024. Independent offline
byte verification covers those saved payloads, not every successful matrix
tensor. At the first failure, available original-dtype outputs and input
handles observed at failure are retained; those dumps are not historical
before-input payloads. Genuine before-input hashes are separate evidence.
No later points execute after failure. Parameter replacement, ahead-state,
scheduler lifecycle, full-model workload and performance qualification remain
separate and are not granted by this matrix.

`route-plan.json` binds the exact five-field matrix plan. `owner-plan.json`
uses the frozen stable single-child lifecycle and its independently pinned
three-source manifest. The predecessor is the archived v7 route-canary
RESTORED journal `b8a209b3…669ce2` and restoration audit `732e5c19…561e`.
The phase is explicitly `e024-route-matrix`; the owner retains the original
4dff preparation and original CLI receipts before adapting to the new fixed
Python qualifier. Source/runtime/model preparation runs before and after the
child. Pressure1, AC/thermal/swap checks,1830-second work budget, independently
enforced1800-second child budget and bounded TERM/KILL/wait are unchanged.
An owner PASS means execution/resources/reap only, not numerical qualification.

`prepare-plan.py` performs CPU validation and preparation only. Its result is
saved under `.artifacts/m5-ultra-256gb/route-decode-matrix-v1-cpu-prepare`.
It does not prepare, stop or restart the real service, import MLX or run a
device. The root separately reviews and runs the read-only controller prepare,
then writes `receipt.json` binding the real PREPARED journal and exact sources.
Only after that may the root run `execute.py`; its existing finally block
attempts bounded service recovery regardless of the experiment outcome.
The registered outputs use `route-decode-matrix-v1-{owner,cold-journal,execution}`.
Restoration checks persisted configuration and fresh visible model/MTP/PLE
evidence; warm KV and exact hidden runtime state are not claimed restored.

The future matrix raw auditor is deliberately absent from the execution plan
until separately frozen and reviewed. Producer success, numerical audit and
restoration audit are separate evidence. No GPU or service action has been
performed by this registration step.
