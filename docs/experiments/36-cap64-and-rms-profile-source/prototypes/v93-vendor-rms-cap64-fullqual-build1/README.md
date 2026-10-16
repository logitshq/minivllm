V93 independent new vendor-plain-RMS profile numerical qualification, physicalCAP64/native89. NOT an old38/44 profile PASS and NOT full oMLX semantics.

Frozen92 candidate27py differs from44 ONLY plain RMSNorm leaf profile/defaultOFF/cachedgamma. GatedRMSNorm, current44 GDN conv/gates/activatednormreuse, allweights/layout/math elsewhere remain exact. A uses independent checkpoint raw BF16 gamma NN.RMSNorm factory/context; B uses separately offset-derived BF16 cachedgamma and actual MXfast.rms_norm. ALL161 pairs compare raw bytes, dtype/shape, distinct objects and disjoint address ranges beforeANYtoken. Model1349parameter identities unchanged; derived161 cachedgamma arrays are separate readonlynative pins (1510total). Reference161 rawgamma owners remain independent and readonly. Gamma payload1,337,344B eacharm, no fullshard/model duplicate.

Eachbase/reset/changed request has TWO independent prefills: A rawgamma context includes make_cache/forward/rebuild/eval/synchronize/Shared views; B normal preparedgamma path. Strict131seed +lastlogits bytes/finite/dtype/shape +firsttoken(oldID). Retain both sets, native ONLY uploads B own131seed; A decode ONLY continues A ownstate.393seed+3prefill-logit checks; first failure saves raw. Untimed ordinary B32 warm outside93step numericspan;3requests31decodedsteps each afterward.

EveryAdecode construction/eval/sync lies inside reference_context, which finally restores plainnorm class. No legacy GDN rebind/flag-off; A/B both current44 GDN. Bcapture observes161 distinct plainnorm calls with actualvendor branch eligibility and48 distinct convnorm calls, removes bothhooks finally before native replay. All161 cachedgamma actualnative original full-range readonly bindings required. Physical132 outputshape/dtype/finite and everybyte strict A/B via native Cmemcmp eachstep; original131captureseed+132capturedoutputs plus separateA/Brequestseed and bothprefilllogits readonly.322 gamma arrays reread Cmemcmp perrequest/final.

Old38/44 archived outputrecords/96IDs are immutable and cross-equal. All96 newA/Bgreedy IDs HARD equal oldIDs. Oldcanonical132 hashes are diagnostic equal/different counts ONLY, never a gate/oldprofilePASS. For32KV states eachstep require physical64unusedtailzero then CPU appendzero tocanonical256; no256GPU state or tolerance. CanonicalA/Bbytes strict; fields reportoldhashdiffs honestly. Required93steps12276strictoutputs2976KVtailchecks;2ICBbanks[48,45],48S/83staged;3reset/GPUargmaxfeedback,1349sameparameterowners.

Root soleGPU/resource/timeout/PIDowner CLI:
```sh
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
DYLD_INSERT_LIBRARIES=/private/tmp/minivllm-engine-optimization-20261014/v89-native-capacity64-build1/libv89replay.dylib \
/Users/ryanli/code/minivllm/.venv/bin/python -B \
/private/tmp/minivllm-engine-optimization-20261014/v93-vendor-rms-cap64-fullqual-build1/fullqual.py \
--tokens 32 --out /private/tmp/minivllm-engine-optimization-20261014/ROOT-CHOOSE-NEW-V93-OUTPUT
```
AuthorCPU source/AST/inert checks only, no MLX/NumPy/dylib/model/GPU execution. Exact89native source/lib unchanged. Any newA/Bbit mismatch or oldIDdifference STOP; no threshold rescue/performance/HTTP/adoption claim.
