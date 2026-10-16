Build2 delta: accept only exact observed source-store-only RW output triples documented by source-write-proof.json. Reflection remains RW. Unknown/otherRW still STOP; full-allocation alias proofs unchanged. Frozen build1 failed/noformalqualification remains bound. Protocol/header/math unchanged.

# V72 native ICB stationary 8-bank micro — CPU design
Includes frozen native40build2 replay.mm without edits; all existing APIs remain available and unchanged unless new v72 APIs are explicitly used.
One fresh native capture:16 evaluated outputs in forced A0..A7,B0..B7 order. A=fixed-shape compiled native QMM(R4), B=frozen69 R8/SG2;16coldcompiles outside capture/timing.
Eight independent actual MLPgate weights/scales layers0,8,16,24,32,40,48,56 (50,135,040B each;401,080,320B total); each uses its own previously saved actual X.
Native new build_ranges rejects non16dispatch capture, feedback/pingpong/legacyICB enabled, missing reflection/PSO descriptors, or active unknown/readwrite bindings.
Each dispatch must have exactly one write-only full original outputallocation and only read-only pinned W/scales/X plus immutable setBytes. No output allocation may overlap ANY input or another output allocation.
Require output mapping and expected function/geometry arm order; parameter pin role resolves original actual buffer.contents bounds. No weight repack/concat.
One16command ICB, two static ranges[0,8)/[8,16), exact captured geometry and original resolved functions; descriptor changes only supportsICB=YES when required.
Independent QMV capture-envelope barriers are reported; new stateless proof permits no internal dispatch barriers. Explicit Buffers barrier follows each execute range of8 before those8 output slots are reused.
Each selected arm owns8 mapped outputslots (16original owners acrossarms). No outputslot remap alias feature.
loop_range(arm,groups) = one FFI/one concurrent compute encoder/one CB containing groups execute(range8)+bufferbarrier; one commit/wait. No perdispatch Python, eval, outputcopy or hash.
New restore_inputs() copies native40 arena snapshots once OUTSIDE block clock; loop has no feedback/state mutation. W/scales/X can all be pinned readonly/zero-copy and report exact resolved bytes.
New copy_output reads only selected-arm registered arena output after successful completed loop; original MLXinputs/outputs retained and checked readonly outside clocks.
Front-end qualification:3vectors(real/zero/fixedrandom)×3comparisons(nativeeager vscompiledA/B, A vsB)×8banks=72strict; raw replay repeat counts1,2,3,4,5,7,8,16 perarm×8outputs=128nativechecks.
Performance predeclared prime128groups, timed1024groups(8192dispatches);6balancedAB/BA+4AA/BB. One loop ffi wall includes encode/commitwait; GPUstart/end envelope reported separately, no fullmodel/HTTP claim.
Gain gate5%,5/6wins, eachcontrol5% and mediancontrol2%; strictAC/thermal0/pressure1/swapused+Swapouts stable. Old V69 STOP result stays unchanged and never becomes adopted.
Only root may execute GPU; author CPU compiles/inert only. Unknown capture layout/dependency/geometry STOP before enabling, never silently falls back.
