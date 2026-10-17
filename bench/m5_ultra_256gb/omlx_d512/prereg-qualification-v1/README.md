# oMLX D512 qualification v1 registration

Registration snapshot: builder, owner `--prepare-only`, and real read-only
controller preparation all returned rc0. Controller preparation completed
2026-10-17T23:52:54.081116+00:00; journal state was **PREPARED**, original service PID
92152. At this snapshot no experiment owner/execution directory
existed: no cold stop, model request or qualification result is implied.
Root seals the actual PREPARED journal receipt and commits before `execute.py --run`.

Plan SHA256: `e5294522149dfccccab4e26189280d77ddb6f3eb0b54fe78034646c59a71d4ef`.
PREPARED journal SHA256 at this snapshot: `53a59c3d52efbc58914cd41ae3b468884d571ace5dca0be1f3cb247946fabd73`.
Predecessor is the immutable E024-serving-v1 restored journal plus independent
restoration audit, both exact-SHA-bound in the owner plan.

Two fresh directly owned official oMLX servers: C1 ×8, then C8 ×16 = **24 D512
requests**, frozen original prompts, greedy sampling and normal EOS with cap512.
Port18882; explicit local benchmark dummy token. Actual VLM engine/runtime/model
class, full prompt, raw/SSE count/text/EOS, cache disabled/no hit, request seal,
C8 actual batch≥2, raw resources and real TERM/wait/reap must all be checked.
Seal is a request boundary, not process EOF or complete lifespan health proof.

Deadlines: ready180/request600/group720/overall1200 seconds per trial;
close10/TERM15/KILL15, resource interval5. Owner work budget2804 seconds plus
bounded postflight120. Four supplied inventories contain oMLX runtime5245,
observed official source8, target13 and outer runtime4309 entries (groups can
overlap). Preparation inherited large payload SHA/stat evidence and checked
current metadata; actual owner preflight/postflight must rehash all payloads.
No complete recursive runtime-closure claim is made.

[57/57 CPU evidence](../../../../docs/experiments/m5-ultra-256gb/omlx-outer-prepared-cpu-v1/README.md)
checks protocol/fakes only. The pending device cycle grants at most observed
route/count/text/cache qualification, never full-state or output-quality
equivalence, formal performance, or a Pareto point. Three independent closed-cycle
qualification/resource/restoration audits remain required. Actual stdout/stderr,
unknown launch intent and every failed attempt are retained.

[Commands](commands.json) · [Preparation](preparation.json) · [Source pins](source-manifest.json).
Use the driver after sealing its receipt; it preserves unknown-launch handling
and always attempts recovery. Do not treat the standalone stop/owner commands as
an alternate execution protocol.
