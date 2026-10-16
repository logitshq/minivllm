# V65 independent CPU review — FINAL frozen source
No concrete source blocker found for root's pinned M4/27B full qualification. No MLX import, GPU, model load or candidate edit.
Reviewed frozen worker e1f4d07d…117d1a and independently rehashed all98 pins; manifest27→29 has only qwen3_5.py changed plus two new files.
Candidate qwen3_5.py:52,59–73 is defaultOFF/strictbool; removing only entry guard reproduces original model body AST independently.
residual_carry.py:79–90 keeps first inputnorm; then64 mixer-add/postnorm,63 MLP-add/next-inputnorm,1 final MLP-add/modelnorm, original head.
residual_norm_metal.py:21–24,42–48 rounds typed BF16 Add before FP32 squares/normalization and writes the same rounded residual.
Helper bytes equal old pure helper b53e4b88…21413c; reversing only Add loads/residual store reproduces old norm SOURCE literally.
Reduction tree, precise rsqrt, centered W(1+float(weight)) rounding and fp-contract(off) are retained;330 synthetic API cases are limited prior evidence.
Metadata eligibility (residual_carry.py:26–73) checks B1/exact runtime classes/cache/int32/no capture/record and129 centered5120 BF16/FP32 norms.
Unknown metadata falls back before mixer/cache mutation; embedding mismatch also falls back before mutation (:98–102); errors after graph starts propagate.
This guard is not a complete validator of arbitrarily mutated projection/model internals; pinned source+loaded model are qualification scope. No mid-graph fallback claim.
Worker :216–246 observes actual loaded129 norm dtype/shape/centered/finite eps/parameter identity before ANY token, writes observations before asserting.
Historical V33/V63 metadata is not substituted for these live observations; actual V65 observations remain for root execution.
Worker :271–288 sets carryFalse and binds both GDN entries to oldaccepted25 AST; finally restores flag and methods.
Core A model forward, all132 output construction, mx.eval and synchronize stay inside this baseline context (:456–459).
Prefill/ordinarywarm also select carryOFF (:334–351); initial1349 parameter identities are rechecked after opt-in and at end (:248,533).
B capture checks _eligible before mutation, then actual128 residual calls/weight order and48 distinct normreuse GDN calls (:368–386).
Both hooks restore in finally (:298–313); only capture is counted; native replay performs the frozen captured graph, not128 Python hooks each step.
A_calls=0 is explicitly flag/source inference, not a measured A hook count; saved V38 AND V44 oldA hashes independently gate every A output (:465–480).
B reset uploads all131 seed inputs once/request (:438–441); perstep only native replay and observed4B prediction; no CPU/saved future token upload (:463–490).
93×132 shape/dtype/finite/strict native Cmemcmp gates,128 complete caches, originalseed/output/requestseed readonly, reset+changedprompt are preserved.
Native48 S eligibility/83 staged links/2 ICB banks and completed counts are asserted from actual capture/report, not assumed new PSO support.
New residual PSO/capture, BF16 model equality and live norm engagement remain UNQUALIFIED until root actual run; no speed/ownership/adoption claim.
