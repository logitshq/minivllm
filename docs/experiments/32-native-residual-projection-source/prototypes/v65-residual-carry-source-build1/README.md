# V65 carried residual B1 candidate (CPU ready only)

Candidate: `../candidate-v65-residual-carry-build1-root`; manifest: `candidate-manifest.json`.
Enable with `model.set_residual_carry(True)`; default OFF; non-bool raises TypeError. Disable with False.
The V44 conv/normreuse and gates flags retain their own behavior and defaults.

Normal B1 direct decode only: exact ForwardContext/model/block/cache classes, no record/capture/speculative/verify flags, no capture features, last_only=False, input int32[1], supported cache metadata, all129 centered BF16/FP32 hidden5120 norm weights and finite Python-float eps. Unknown metadata falls back before embedding/mixer; embedding actual BF16[1,5120] is checked before first cache mutation. No mid-graph fallback.

First inputnorm is normal;64 attention-add/postnorm +63 MLP-add/next-inputnorm +1 final-add/modelnorm use the byte-identical330checked helper. MLP/attention/GDN/head operations retain their original order and flags. Carry exists only inside one model forward; no array parameters or request state are added. All old prefill/grouped/B2/B4/spec/featurecapture/last_only/lm_head_fn paths use the preserved original graph.

Six CPU tests passed: source/AST fallback identity, exact kernel+inert compilation, defaultOFF strictbool metadata, upfront unsupported metadata, embedding mismatch before cache mutation,128-fusion symbolic order for both head branches. This is source evidence, not actual engagement or numerical qualification. Run only CPU tests with `python3 -B check_cpu.py`.

Root fullqual must use carryOFF for A even while B is ON. Before ANY token dump/assert all129 actual norm dtype/shape/zero_centered/eps; compare every93x132 output to original accepted25/V38 actual gold, readonly/parameter identities/base-reset-changed prompt; prove128actual calls for eligible B and0for A, then remove instrumentation before timing. New residual PSO/native capture resources must actually qualify. No old residual micro retry or speed claim. All earlier control-drift STOP evidence remains unchanged.

The copied helper intentionally retains its original descriptive docstring; actual330raw API evidence is pinned by the source audit. Physical nonalias/no-copy is not inferred from wrapper output declarations.
