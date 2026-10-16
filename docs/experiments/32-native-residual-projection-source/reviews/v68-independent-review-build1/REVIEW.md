# V68 independent CPU source review
PASS_CPU_SOURCE_REVIEW_NO_CONCRETE_BLOCKER; actual model/capture qualification remains for root. No MLX import/GPU/candidate edits.
Frozen fullqual8e24ff75…b8dde5 + pins07c2d3e2…ab1c6: all107 pins independently rehashed.
Candidate27→28py: only old model/gdn.py changed; all old methods except init/dispatch independently AST-identical; _decode equals accepted25 directly.
Frozen V66build2 helper/header/body/layout/license identities match copied candidate; source proof includes no weight concat/repack/copy.
fullqual.py:229–240 sets all48 four flags OFF, binds both GDN decode entries to original AST, finally restores flags/methods.
A core forward/output creation/eval/synchronize are inside this context (:462–465); original prefill/ordinarywarm also select OFF.
Extra layout decode (:306–319) retains each of48 actual original norm X; dispatch observer restores in finally before preparation/capture.
432 actual X/W/scales descriptors are observed with ensure_row_contiguous=False (:322–351); singleton-row strides are handled explicitly.
prepare receives observed W/scales strides and actual X unit-K stride, asserts no active allocation increase and all8 operands exact original objects/layer.
Subsequent X layout uses unchanged RMSNorm→norm_metal row-major output contract; arbitrary externalX is explicitly outside qualification.
Candidate identity guard (:261–284) rejects changed W/scales, drops stale helper before any cache mutation, and falls back to current original operators.
All1349 parameter identities are unchanged afterprepare and end;384 target ranges (2,278,195,200B) are stream-hashed beforeprepare/afterprepare/end.
B capture requires48 distinct GDN calls and48 distinct helpers each returning4 BF16 outputs with exact shapes (:377–381).
qkv/z/b/a all feed unchanged conv/gates/outputnorm suffix (:286–302); both capture wrappers restore original function objects in finally.
Native PSO support is observed;384 target full original-buffer bounds/zero-copy/readonly roles must resolve to same model owners (:413–423).
13pointer ABI is frozen source evidence; individual per-argument reflection is correctly NOT_OBSERVED in native40 report. No missing-field blocker invented.
Each request uploads131 seed arrays once; B consumes own GPUargmax feedback; only observed4B prediction copied each step, never uploaded as future token.
A full132 outputs each step must match savedactualV38 AND V44 A hashes before strict Cmemcmp to nativeB (:471–485); counts enforce93×132.
Includes all128 full-capacity caches, shape/dtype/finite, retained A-input readonly and originalcaptureseed/output/requestseed readonly every step.
48 S eligibility/83stagedlinks/2ICB bank completion checked from actual reports; no assumed successful new PSO/capture or performance gain.
Qualification asserts changed cache seed, three32ID sequences, oldgold, reset and original W readOnly; no source false-pass path found.
