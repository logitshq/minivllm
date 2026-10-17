# E027 native-route bridge v1 — registered before execution

The existing native M5 Linear.__call__ invokes one mx.quantized_matmul over all
T rows. MLX0.32.3 source selects fp_qmv_wide for these K5120/T2,4,5 cases, while
E027-v1 compared independent M1 calls. The original native multirow kernel
already reuses weights. Therefore the prior M1-preserving PASS neither proves
a drop-in replacement nor establishes less weight traffic than native.

Keep the same33 cases and frozen candidate, inputs, runtime, resource policy,
120s child,1GiB MLX/2GiB physical observation bounds and68phase layout. Add one
native multirow output to each case after the original-expression control and
before shared-W. Compare it against the independently computed M1 raw output,
with finite original BF16 byte equality, including signed zero. Expected full
coverage becomes99 comparisons. Stop on the first mismatch; save native M1,
control and native multirow raw bytes even if shared is not reached. Any
mismatch rejects a direct native-multirow→shared substitution for this attempt.
No tolerance, new M1 baseline or altered numerical contract can rescue it.

Separate native_bridge_screen.py/native_bridge_owner.py preserve v1 files and
receipts. A new plan/pins/jobs and prereg commit precede one fresh device run
under the existing authorized pause/resume lease. This is a numerical route
bridge, no timing/performance admission. Whole-model propagated state and
uninstrumented throughput–complete-request latency remain later gates.
