# Carried residual + RMSNorm: CPU source feasibility
FEASIBLE for a new default-OFF normal-B1 decode graph; no engine authored, no MLX imported or GPU used.
Old block order is qwen3_5.py:40–45; model final norm/head order is :73–86; old Block AST and all prefill/capture paths stay intact.
API residual_norm(left,delta,weight,eps) returns (BF16 residual, BF16 normalized); exact [1,5120] operands and [5120] BF16/FP32 weight only (:83–118).
Typed T Add rounds to BF16 before FP32 norm (:21–22,42–44); reduction, precise rsqrt, centered W cast and final T cast remain identical.
CPU literal reconstruction (remove only typed-Add loads/residual store) equals current norm_metal.SOURCE byte-for-byte; fp-contract(off) preserved.
Order: r=embedding; h=layer0.inputnorm(r); mix=original attention/GDN(h); (mid,post)=fuse(r,mix,postnorm); m=original MLP(post).
For layers0–62: (r,h)=fuse(mid,m,next.inputnorm); layer63: (r,h)=fuse(mid,m,model.norm); original tied/head branch then consumes h.
This forms64 attention-add/postnorm +63 MLP-add/next-inputnorm +1 final-add/modelnorm =128 pairs; first inputnorm remains normal.
No carry crosses tokens/requests; original attention/GDN/MLP, gates/conv/normreuse, cache states and speculative storage semantics remain unchanged.
Upfront eligibility must validate normal B1 context, no record/verify/spec/capture/grouped-prefill, all129 centered hidden5120 norms, supported dtype/eps.
Fallback is selected before any layer executes; never rerun the old graph after partial cache mutation. FP16 weights require original-path fallback.
Historical V33 actual receipt.parameter_ranges records all129 hiddennorm weights BF16[5120],10240B; this is saved Shared.meta, not current dtype observation.
V63 actual PSO template T/W/TG/ZERO_CENTERED/GATED maps second bfloat16_t to W and1024/1/0;129 weight roles each resolve10240B readonly ranges.
V65 afterload-before-ANY-token must independently dump/assert all129 actual dtypes/shapes/centered/eps; no separate model probe is needed.
Existing330 raw API checks passed (BF16 and FP32 W, nonfinite payloads included); physical nonalias and model equivalence were not attested.
Old micro build1/2 remain FAIL_STOP: first A/A ratios1.05153063/1.05710117; zero candidate pairs, so neither speedup nor slowdown was established.
New graph target removes128 standalone Add dispatches; actual ICB capture must measure the dispatch change. No speed estimate or adoption claim.
The helper recomputes Add for norm reduction/output; hidden-vector traffic is not clearly reduced. Benefit hypothesis is fewer dispatches/materializations.
Fresh eps scalar per call/row-contiguous preparation may add allocation or copies; current helper cache captures no weights, and physical layout is unproven.
Next qualification: one-weight original-A/new-B, all93×132 strict outputs plus readonly, reset/changed prompt, actual fusion engagement and parameter identity.
EVIDENCE.json pins source, historical metadata, actual PSO roles, reconstruction proof and unchanged negative results.
