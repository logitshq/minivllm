# E007: standalone large-batch GDN fusion

This is a rejected experimental prototype, never imported by the runtime.
It retains separate projection GEMMs and reuses the already qualified small-
batch Metal sources with independent batch offsets for B9..64. The process-
local class patch is only suitable for this single-threaded test process.

`screen.py` records a plan before work, performs synthetic and full-model exact
comparisons, uses ABBA timings, and checks a staggered scheduler trace. No new
dependency or default runtime flag was introduced. Run via the existing owned
resource guard; see the exact command in `docs/optimization-guide/experiments.md`.

**Timing field clarification:** `cold_ms` in this v1 script/result means the
first timed call **after correctness evaluation**. It is not cold compilation
or cold model load. The medians use the subsequent declared warmups and ABBA
samples. This naming issue does not change the measured values or reject rule.
Full-forward timings include Python dispatch and synchronization but exclude
HTTP, scheduling, model loading, and prefill. Never add them to an HTTP curve.

The raw source pins and first attempt remain unchanged. Any revised experiment
must use a new output directory and independently freeze its source.
