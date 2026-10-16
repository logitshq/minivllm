# V40 fullmodel numerical qualification (CPU-ready only)
Candidate root: candidate-v40-normreuse-cache-build1-root, accepted25 + one lazy helper; 24 parent files byte identical and original _decode AST retained.
Default OFF; B opts 48 GDN objects into normalB1-only raw-convolution-u normreuse/held32. Conv fusion is absent; helper alone performs SiLU. Same parameter identities and gates ON are checked.
A binds the exact old accepted25 _decode AST temporarily to both candidate dispatch entries, then restores; every actual A output also matches saved V38 SHA/shape/dtype/bytes for same case/step.
One model load; base/reset/changed-prefill requests, 93 steps, all132 full outputs exact + all original/request seed/capture outputs readonly. B token feedback is GPU-only.
Native40 build2 is bound, including actual tiny2 PASS and local15bank30strict PASS; requires48eligible S pairs,83stagedlinks,2ICBs and completed bank counts48/45.
CPU: python3 fullqual.py --inert. Root-only child under owned resource/timeout/cleanup wrapper: venv/bin/python fullqual.py --out NEWDIR --tokens32.
Require DYLD_INSERT_LIBRARIES=/private/tmp/minivllm-engine-optimization-20261014/v40-native-icb-build2/libv40replay.dylib and standard50/50/20/0. No agent GPU execution or performance/adoption claim.
