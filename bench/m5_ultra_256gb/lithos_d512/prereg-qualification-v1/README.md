# Lithos D512 C1/C8: qualification preregistration

Run the existing official target-only Lithos model/pack after temporarily stopping
the original oMLX service under the existing user authorization. A two-phase owner
holds each server Popen directly; C1 must complete and reap before C8 starts.
Restore the original model/config, then independently verify MTP and resident PLE.

Same eight D coding fixtures, full original prompt IDs, greedy normal EOS, cap512,
Qwen3.8-27B NVFP4 revision5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6.
C1 has8 requests, C8 has16. Official `/api/chat` accepts concurrent HTTP callers
with FIFO GPU execution; C8 is not GPU batch8. Retain every raw request, NDJSON
chunk, Generation tokens, EOS/count/text join, actual backend/pack route and
per-request zero cached-prompt count. Observers are ON: timings are diagnostic
and cannot supply a formal Pareto point. Stop on any qualification/transport or
resource failure; no data-dependent retry or outlier removal.

Use the existing capacity2048 target pack; no download, repack, weight change,
sampling override, or draft route. Exact runtime/source/model/pack identities
are in verification.json; inherited large payloads are not rehashed during CPU
preparation. The owner hashes all7,520 supplied files before and after execution,
outside requests, and samples pressure/power/thermal/swap while each server runs.
Only the outer HTTP/tokenizer runtime is current; official Lithos runtime remains
at its previously pinned versions. OS shared libraries/stdlib are not encapsulated.

Next formal comparison, after qualification: fresh observer-OFF official servers,
8 warm +64 measured requests, planned3 process repetitions per C and same actual
committed-token / common-window rate versus full POST-to-DONE latency. Keep all
process observations and the median. Native currently has only1 repetition per
point; do not imply matched three-repeat evidence until actually measured.

CPU preparation only until the source-bound owner/controller plans and receipts
are finalized. No D512 Lithos result or competitor dominance is asserted here.

Frozen owner plan: `d4e0edad9491cb17d4c18451850bf334e67376ad9eb6eba710d23c5b6bdfdc3f`.
Readonly service prepare passed once; current originalPID19870 remains running.
Root61 focused CPU tests pass across owner/controller/restoration/raw resource/raw
qualification checks. Work budget2804s; ready180s, request600s, group900s,
overall1200s per trial, close10s, TERM15s, KILL15s, resource interval5s.
The exact execution driver always attempts bounded restoration in `finally`.
See [source-bound receipt](receipt.json), [root review](root-review.json) and
[exact plan](owner-plan.json). Expected next action is the single actual run.
