# GDN projection micro AC attempt2 — STOP-MICRO-GATE
- One explicitly authorized actual-unsandboxed require_escalated run; 2026-10-15 07:29:54–07:30:03 UTC. No third attempt.
- Decision: strict math/activation and selected3 memory gates PASS; performance consistency FAIL (3/5 wins <4). No resource or numerical failure.
- Frozen candidate manifest `0b9c4645ef7321ddb501f416062af7430df63408f4ca820d3cb24883186ba162`; exact26 sources reverified.
- Frozen parent manifest `2a990456522fdd00d0f331011964be4e9b755e47b2b6eeccd293b1a8c435baad`; exact25 files unchanged.
- Driver SHA `e7c5a76738ead787a7dda812c85a2c4ced5ba9135da4bf29026754ea2928a5cc`; final-v2 CPU receipt SHA `2f811fc2d84d777c3b7e114bb7dc86933f4b8f31b02dab2fc62f5d03d772d686`.
- Source/driver/raw evidence were not edited; fixed4→1 only, default off; no automatic2→1, V5/V6/idlequeue, full48, or E2E.

## Authorized app cleanup
- Fresh NSWorkspace inventory selected main bundles; helpers belonged to the same app. Each eligible app received one bounded normal quit.
- Arc `company.thebrowser.Browser`: observed mainPID12715; quit rc0; exact ps rc1/empty, absent. No repeat.
- WeChat `com.tencent.xinWeChat`: mainPID14819; quit rc0; exact ps still present RSS106992KiB at cleanup; left untouched, no dialog response/retry.
- Earlier Slack13978 was absent after prior cleanup; earlier Granola86747 and NotionCalendar37907 had quit rc0 but remained at prior snapshot and were not retried.
- Chrome/Claude/Codex/editors/terminals/unrecognized apps protected; no forcequit, settings, system/fan/daemon changes.
- Cleanup raw directory: `gdn-projection-app-cleanup-before-ac-attempt2`; inventory/raw resource/RSS/quit receipts preserved.
- Cleanup 07:29:30.903→07:29:31.511UTC: AC46% charging, thermal0, pressure1; free70→71%; swap1990.5MiB unchanged.
- RSS sum change−2301072KiB (~2.19GiB), shared pages double-counted; this is not physical RAM released. Pageouts/Swapouts/Swapins/Compressions unchanged; Decompressions+3509.

## Actual model/resource boundary
- MLX0.32.3/GPU0 Apple M4 Max/applegpu_g16s; backend_supported=true. Tool session36936; supervisor16124; sole owned child16125; prior15259 absent.
- Child/controller exit2; controller waited/reaped and exact ps rc1/empty proves child16125 absent; cleanup.json records allocation release on process exit.
- All19 model samples: actual AC46% charging, thermal0/nominal, pressure1, swap1990.5MiB, known inference matcher0 (limited matcher scope; not all GUI GPU use).
- Fresh→strict-load→capture→release free73→48→34→64%; load active15133597696B; overall MLX peak15529189989B.
- Whole-VM fresh→release deltas: Pageouts+141 (at load), Swapouts+0, Swapins+36, Compressions+1330692, Decompressions+913800. Original counters retained in resources.json.
- Strict load/eval2.767045s; real prompt19/decode2 capture2.261450s; archived first3 IDs matched79852/18887/220.

## Math, activation, and memory
- Sequential first0→middle32→last62 passed strict gates; first group PASS at07:30:00.617623UTC. Each92 BF16 checks, total276, finite/shape/dtype/raw-byte exact, zero differing bytes.
- Real norm-after decode0/1 activations for all3 groups captured from ordinary baseline GDN decode. Capture spy removed before packing/qualification.
- Real2 + zero + seed20261015 random amplitudes1/16,1,16 +8 prepared timing inputs compared against original frozen Linear modules/arrays.
- All3 activation sessions prove exactly1 true QMM delegate: K5120,sumN16480, rows10240/6144/48/48; U32 weights16480×640/U8 scales16480×320; NVFP4 group_size16,bits4,transposeTrue.
- All4 qkv/z/b/a segments evaluated. Original fallback B1/2/4×T1/19/33 and unsupported-shape rejection completed; no all-fallback qualification.
- Selected original refs142387200B (~135.79MiB) transiently retained then released; no clear_cache. Release active15133923336B vs ready15133966344B, delta−43008B≤8MiB gate PASS.
- This observes selected3 only; physical buffer sharing/output-split zero-copy/full48 allocation not certified.

## Frozen performance gate
- Spy removed before timing; same8 distinct prepared inputs,3 warm passes,16 repeats/block,5 alternating AB/BA pairs; combined3 groups/12 output segments evaluated per call.
- Concat/eval/input preparation/first-call/capture excluded; all12 outside-block output hashes matched within each pair.
- Frozen driver lines389–405: before resource probe→first timed block→12 output hashes→second timed block→12 hashes→pairJSON write→after resource probe.
- No resource probe between blocks; blocks retain elapsed durations without individual start/end timestamps. UTC samples mark probe start, not block boundaries; within-block thermal/VM attribution unavailable.
- All10 timing samples AC46%/thermal0/pressure1/swap1990.5MiB; per-pair Pageouts/Swapouts/Swapins all0; Compressions[0,0,1674,0,0], Decompressions[1474,4476,119,620,123].
- Exact paired UTC/VM/block order in derived `gdn-projection-micro-build1-ac-attempt2-timing-resource-alignment.json` SHA `ef6c9bf56ce34602fa7346396ea086c16f2f89d1374bc702ba7a579a8132b71b`; raw files unchanged.

| Pair | Order | A original ns/call | B packed ns/call | A/B speedup | Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | AB | 893958.3125 | 697828.1250 | 1.28106× | +21.940% |
| 2 | BA | 540479.1250 | 969276.0625 | 0.55761× | -79.336% |
| 3 | AB | 932315.1250 | 510427.0625 | 1.82654× | +45.252% |
| 4 | BA | 476862.0000 | 936739.5625 | 0.50907× | -96.438% |
| 5 | AB | 914846.3750 | 539984.3750 | 1.69421× | +40.975% |

- A median893.958µs; B median697.828µs; median reduction21.9395%≥5%; wins3/5<4. Result STOP-MICRO-GATE; eligible_for_followup_qualification=false.
- Observed order association: AB3/3 packed wins, BA0/2 packed wins. Cause unproven; no rerun/gate relaxation/source change.
- Full48 memory/route/logits/IDs/caches/lifecycle and E2E not run. No model-level speedup or full48 correctness claim.

## Immutable raw evidence
- Raw directory `gdn-projection-micro-build1-ac-attempt2` contains12 files; no failure.json because this is a completed gate stop.
- External SHA/bytes inventory `gdn-projection-micro-build1-ac-attempt2-raw-sha256.json` SHA `e9cfdc0ac91ce1d317b9ac36f02ab424c57dd14cfa8b448ad5161a4b6a5500d8`; includes exact VM total deltas.
- result.json SHA `2bdf28bd0c49186c2641a6aefb096e0928df8c3cefe3793f6a191ae9a5feb0ab`.
- byte-checks.json SHA `426ad4decbf6195e46dcbdc32fde57cb929e4c0ace1b59e4042bf82574c17026`.
- activation-sessions.json SHA `5dd8e4b8313093cf8d021f1f0683c999f96b0a628b22108064211100154727af`.
- timing-pairs.json SHA `04084deb8f54a8e77cf3c82c015a82def99697a2fd84c04d401956a3be645e6c`.
- resources.json SHA `68a592cdf26b89f4ef6132fc5e381a68179d402e3937055c80b4b426bb6246b2`.
- cleanup.json SHA `55e30bccbb8a92e32c84f222b400354dfec946dd090c4e1145ad309efe368a48`.
- GPU ownership released; stop here. Root must decide any future work; no automatic third attempt.
