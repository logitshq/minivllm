# V89 independent CPU review

Declared physical64 / 32-token synchronous scope: **no new blocker found**. Actual saved full PASS remains valid; this is not an async/boundary/service/performance qualification.

- Exact worker d1669d1b…c789c, library ae5cfa2d…256774; inverse patch restores all frozen40 source bytes except the intended64/256 capacity guard and its message. All27 candidate44 Python files unchanged; all71 pins match.
- Seal wrapper rejects negative/overflow/non-int values before uint32 narrowing and rejects capacities other than64/256. Native retains actual capacity64. Current positions19..50 and lengths20..51 satisfy its mathematical bounds.
- CPU canonical helper tests preserve signed-zero bits in valid rows, pad each head independently to full256, and reject even negative-zero bits in the physical tail. No MLX/NumPy/native library was loaded by the reviewer.
- Rechecked all93×132 =12,276 canonical metadata/hash records against **both** immutable V38 and V44 golds; all32 IDs/case match. Actual input/output KV shapes are[1,4,64,256], native reports capacity64 for all3requests, and canonical256 exists only as CPU bytes.
- 2,976 paired KV tail records cover both A and B source checks (5,952 individual sides); 93-step physical C-memcmp counter=14,753,056,860B. Three fresh resets/all131 uploads, changed-cache evidence, no per-step uploads, 48S/83staged links and two-ICB parity all match.
- Saved resource samples=33; raw fields/deltas match, AC/thermal0/pressure1 pass; swap-used5006.88MB / Swapouts17460761 remain fixed. Raw matcher had no model match; derived records explicitly supplement loaded ownedPID96714, kept distinct from raw matcher evidence. Owner96705/child96714 rc0/wait/reap and external release/absence pass; these are saved snapshots, not continuous GPU observation.

**Inherited out-of-scope issue:** native replay.mm:183 computes int32 `pos+1` before an explicit upper bound on pos. A standalone CPU UBSan extraction confirms overflow for pos=INT32_MAX. Frozen40 has the same expression; current bounded89 inputs cannot trigger it. For a future expanded raw-seed/service API, reject pos>=capacity before the addition in a new version; do not alter frozen89 or its PASS.

Evidence limit: per-step KV raw arrays are not persisted. The independent saved audit verifies all gold hashes, physical-shape/strict/tail records and native compare-byte counters, rather than claiming a new raw re-read or GPU replay. Source map, 25501 assertions and CPU-only UB reproduction are attached.
