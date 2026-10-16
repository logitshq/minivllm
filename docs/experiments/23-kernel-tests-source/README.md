# Kernel experiments and GPU diagnostics

All runtime features are experimental; no new default, official comparison, or Pareto point is adopted. Prototype directories are immutable copies of ready source. Original absolute paths in manifests identify actual execution, not portable commands.

V26 rowblock and V28 gate/up epilogue pass strict local numerical checks but both micro attempts stop on controls; their descriptive gains are respectively +0.124908% and -0.768400%. V28 build1 was never executed; build2 increased prime before any execution of that candidate and still stopped.

V27 local SIMD numeric passes bounded checks; micro stops on controls. Fullmodel qualification fails on step1 logits: maxabs0.125 exceeds the predeclared1/16. Do not increase tolerance or relabel as full32PASS. The candidate-v27-delta replaces the three indicated files over experiment22's candidate snapshot; its source manifest binds all28 Python files plus Metal.

Profile builds1/2/3 are separate diagnostic protocols at OPS50/1/0. Only build1 uses production controls; build3 has1830 single-dispatch command buffers but GPU buffer envelopes overlap and include overhead. Envelope sums are not exclusive shader times or production proportions. Sources are included; native binaries are excluded and require rebuilding for reproduction.

evidence.tar.gz preserves text-only original resources, sources, raw results, all pairs and external PID release checks. Numeric binary blobs are retained only locally; hashes in receipts do not imply an independent recapture.
