Proposal only; no new shader authored and no GPU execution. Await root topic agreement.

Observed: all48 GDN layers have four independent QMV commands sharing the exact same actual x buffer+offset,K5120; Nqkv10240,Nz6144,Nb48,Na48. V63 qmv-bindings.csv provides each ordinal, original parameter buffer IDs and payload byte sizes.
The four roles total192 dispatches; one multi-output dispatch/layer removes144 dispatches. R4 unchanged:64 threads/tg,2 SIMD groups,4 results/SIMD,8 output rows/tg. Total2060 row tiles equals1280+768+6+6.

Old V16 is already a 4→1 dispatch experiment, not an untouched opportunity:
- candidate-v16…/model/projection_packing.py:67–71 calls one QMM then splits output.
- :86–99 concatenates U32 weight/U8 scales, materializes slabs/views and rebinds original Linear arrays.
- actual gdn-projection-micro-build1-ac-attempt2/result.json SHA2bdf28bd…:276 strict checks,3 selected layers;21.9395% median reduction but3/5 wins<4,STOP-MICRO-GATE; no full48/model/E2E.
- Concat preparation was excluded from its timing, so avoiding concat alone does not establish a faster steady-state kernel.

New mechanism: original four weight/scales pointers retained as explicit inputs and four separate output arrays; no weight concat, rebind, repack, persistent extra slabs, output split or changed parameter identity.
One uniform threadgroup-Y branch picks qkv/z/b/a and subtracts that section's tile offset. Every selected branch invokes the unchanged fp_qmv_fast_impl<T,16,4,false,4> with its own original pointers,constant N and local tileY; x shared,B1 only.
Precise math source: installed mlx/include/mlx/backend/metal/kernels/fp_quantized.h:333–404; K5120%512=0, each N%8=0. Preserve float x_thread16,10 ascending K blocks,original qdot(:81–101),float per-row accumulation,simd_sum and final BF16 cast. No folded scale/new reduction/math tolerance.
Runtime guard MLX0.32.3/M4Max,normalB1 BF16x[1,5120],exact U32[N,640]/U8[N,320],NVFP4 group16/bits4/no bias. Pass original arrays explicitly each call; unknown/B2/B4/prefill/spec falls back unchanged.
Candidate44 integration point: model/gdn.py:202–213 currently calls qkv,z,b,a separately; only replace those projections, retain conv/gates/norm/state logic and defaultOFF metadata.

Qualification before any speed claim: strict bytes for all4 outputs+shape/dtype/finite and readonly input/weight/scales,first/middle/last real weights,real saved norm-after inputs if available plus synthetic banks. Any mismatch STOP, no tolerance widening.
Micro must compare original4 QMV vs new1 with all4 outputs consumed,predeclared balanced blocks/controls and unchanged≥5%local reduction gate; no automatic repeated tuning after STOP. Local PASS alone does not establish model or HTTP gain.
Risk: qkv now shares a producer completion with z,b,a; original graph could overlap convolution with z projection. Fusion may lose scheduling overlap even with fewer dispatches. V63 serialized samples cannot quantify it.
Model qualification still needs complete greedy/cache/readonly tests and matched actual service performance before adoption. MLP accounts for the largest sampled QMV shapes; GDN fusion is a smaller bounded dispatch-overhead experiment, not evidence of≥2ms production savings.
