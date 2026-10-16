V89 physicalCAP64 synchronous native numerical qualification; no performance run.

Frozen candidate44 package27py and allmodelmath/gates/conv+activatednormreuse remain unchanged. Allocation-only class metadata AttnCache.step and DecodeBatch.step become64 after synchronized model load beforeANYtoken, finally restore256. The prefill and DecodeBatch produce real[1,4,64,256] BF16 K/V;31steps consume positions19..49(base/reset) or20..50(changed20prompt). No physical256 KV tensor is created by this worker's prefill/decode path; canonical256 consists solely of CPUbytes afterreadback.

A uses accepted25 original_decode AST/class temporary bind with finallyrestore, exactL20..50/21..51 and physical64 caches. B captures frozen44 graph at fixedL64 into2ICBs,48S pingpong/83stagedlinks, GPU argmax+feedback; no CPU token upload perstep. Oneweightsload,1349identity,48captureengagement hooks removed before replay,131seed and132capturedowners readonly. All3requests31steps required.

Eachstep first checks132 physical A/B outputs: same shape/dtype,finite,strictnative Cmemcmp. For32KV tensors, require allphysical tailbytes[validL:64] exactlypositivezero, then CPU perhead append192zero rows to canonical[1,4,256,256]. Require all132canonical output metadata+SHA exactly equal saved actualV38 ANDV44oldA, with A/B canonicalKV fullbytes compared. No prefix-only compare/tolerance/numerical drift exception. First mismatch STOP/rawfailure保存. Expected12276outputchecks+2976KVtailchecks. Canonical allocations/readbacks are numerical instrumentation, never performance.

Native89 FROMfrozen40build2: onlycapacityguard admits64or256 +truthfulerrortext, inversepatch proves allother source bytes identical. F.capacity actual64 controls allscalarbounds. Unknowncapacity/uint32 overflow rejected. Original40library/evidence remain immutable. Library compiled CPU-only/manual retain-release; never loaded byauthor. Sourceproof.json +capacity-guard.diff retain exactchange.

Root soleGPU/resource/timeout/PIDowner CLI:
```sh
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
DYLD_INSERT_LIBRARIES=/private/tmp/minivllm-engine-optimization-20261014/v89-native-capacity64-build1/libv89replay.dylib \
/Users/ryanli/code/minivllm/.venv/bin/python -B \
/private/tmp/minivllm-engine-optimization-20261014/v89-cap64-native-fullqual-build1/fullqual.py \
--tokens 32 --out /private/tmp/minivllm-engine-optimization-20261014/ROOT-CHOOSE-NEW-V89-OUTPUT
```
OnlyCPU tests/AST/inert run byauthor. Native synchronous protocol only; async extra step,max_tokens>32,cache64 boundarycross/service/HTTP/performance/adoption remain outside.
