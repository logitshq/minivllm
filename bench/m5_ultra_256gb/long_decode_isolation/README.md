# E024: separate attention length from graph compilation

Numerical diagnostic only, B1/used257, same fixed Qwen3.8-27B NVFP4 model and
original 256-token D prefix. Original E024 candidate/qualifier stay frozen.

| Arm | Attention slice | Execution |
| --- | ---: | --- |
| A | 257 | Eager |
| B | 1536 | Eager |
| C | 257 | Original explicit-state forward compiled with model inputs |
| D | 1536 | Original E024 compiled candidate |

The two independent original loads and prefills must agree first. Four arms
then share one unchanged parameter model and receive separate evaluated buffers
from the verified canonical state. Original-dtype logits, all128 state arrays,
131-array input snapshots before/after, parameter byte hashes and actual graph
call witnesses are retained. Parameter Python handle IDs are recorded but do not
replace the byte invariant. Source/model files are verified before and after.

The same original configuration disables gates/norm/fuse/core compilation. Only
C/D explicitly compile their new closure. No undocumented no-fuse API or changed
SDPA environment is used. A–B isolates bound, A–C and B–D isolate compilation at
a fixed bound, and A–D checks reproduction of the rejected combined candidate.
Finite bit differences are diagnostic results and do not skip the remaining
arms. Execution/resource/nonfinite/input mutation failures still stop immediately.
PASS means all four finite arms and invariants completed; it never means that
all outputs are equal. No full D512/HTTP, lifecycle or performance qualification.

The [source-backed design](../../reviews/2026-10-17-e024-dispatch-isolation.md)
explains why length is a plausible cause. Actual runtime SDPA kernel tracing is
not performed here; matching source dispatch is a prediction, not a trace.

`cold_owner.py` owns one actual `e024-isolation` child and retains the original
E024 environment preparation before adapting the child command. Controller v6
uses the preceding independently audited v5 restoration as its immutable service
identity. Work limit1830s, child1800s, bounded cleanup/recovery; pressure/power/
thermal/swap constraints are unchanged. Existing user cold-restart authorization
applies. The service's warm KV and exact hidden runtime state are not restored.

Root read the complete numerical implementation and independent byte auditor,
reviewed all lifecycle differences, and passed focused CPU tests. CPU checks
cannot predict the device outcome. See prereg-v1 for the exact frozen plans,
source hashes and read-only service preparation.
