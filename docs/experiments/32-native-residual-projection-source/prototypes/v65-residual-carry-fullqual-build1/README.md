# V65 CPU-ready full-model numerical harness

One loaded V65 candidate (29 Python sources): accepted V44 parent27, only Qwen entry/flag changed plus exact residual kernel and carry helper. Candidate default remains OFF. Runtime B enables gates48, conv/activated normreuse48 and carried residual. A explicitly disables carry for every prefill/warm/reference call, temporarily binds original accepted25 GDN decode to both candidate entries, and restores all flags/functions in finally. A outputs must match saved actual V38 and actual V44 same-case/step gold before B comparison.

Before ANY token, save live129 hidden-norm dtype/shape/eps/zeroCentered/parameter identity metadata and assert fusion eligibility;1349 array parameters and identities remain unchanged after setters. Three requests base/reset/changed-prefill,31 steps each,132 strict shape/dtype/raw outputs each (12,276 checks);131 inputs uploaded only once/request; native40build2,48S ping-pong/83staged links/two ICB banks/readonly/resource guards preserved. Changed prompt consumes its own native prediction, never archived continuation as input.

Only B capture installs the pure residual_norm hook:128 calls in exact norm-weight order, alongside48 distinct conv/normreuse GDN calls. Both hooks are removed immediately after capture and before any replay. A zero fusion calls are inferred from actual carryOFF plus pinned entryguard AST; no A counter is installed. This distinction is recorded in receipts. Actual fusion/native eligibility is a required gate, not assumed from CPU source.

Root-only CLI, with 50/50/20/0 and DYLD injection:
```
DYLD_INSERT_LIBRARIES=/private/tmp/minivllm-engine-optimization-20261014/v40-native-icb-build2/libv40replay.dylib MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v65-residual-carry-fullqual-build1/fullqual.py --out /private/tmp/NEW-V65-OUTPUT --tokens 32
```
Root external owner must enforce timeout/owned-process cleanup and external release proof; inner Resources guard samples before/after load, before/after request, every4decode, and finally. This harness is numerical only, not E2E/performance/adoption. Old residual micro STOP evidence is preserved, unchanged and not rerun.

CPU checks: `python3 -B check_cpu.py`, then `python3 -B fullqual.py --inert`. Neither imports MLX or uses GPU. Candidate six CPU source/symbolic tests are frozen separately in v65-residual-carry-source-build1.
