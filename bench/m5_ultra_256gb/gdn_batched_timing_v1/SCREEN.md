# E048 local batched GDN timing screen

Prepared source only; device NOT RUN. `screen.py` is default OFF and imports no
backend before explicit execution. It reads the exact original E046 child plan,
actual independent PASS audit, complete receipt and captures table through fixed
SHA bindings. It uses first captures at steps1/129 for B2/B8 early/late and actual
layers0/32/62: twelve local cases. These are real native batched activations.

One original Qwen3.8-27B NVFP4 model is loaded through the frozen engine, with
compile/fuse OFF, gates ON, norm reuse OFF and original MLX environment. Model and
runtime verification, load and the three layers' parameter checks are outside
measurement. Both native and candidate use the same actual model parameters;
all14 parameters per selected layer must match E046 original-byte receipts.

The baseline saves the original instance's bound `_decode` before the original
candidate `install` scope. Both routes therefore keep the five original B2/B8
Linear calls. No per-row M1 substitution, packed projections or parameter copies.
Each route owns independent evaluated seed storage; every call resets new state
wrappers/descriptors before the clock. Original native and candidate replace
conv/S outputs, so the initial seed payload remains immutable; exact before/after
comparison enforces this. The timed interval contains fresh forward construction,
complete output/conv/S evaluation and final synchronize. Reset/evaluation of seed
views, disk reads, hashes, saves, counters and summary arithmetic are outside it.

First both arms must reproduce all original output/conv/S bytes, shapes, dtypes
and finite flags; both raw outputs are saved before comparison. Then5 warmups per
arm and5 groups×10 paired samples use AB/BA order. All1200 measured durations and
120 warmup durations remain in case `timings.jsonl`; no outlier removal or best
sample selection. Summaries retain per-group medians and native/candidate ratios.
After timing, input preservation, dispatch counts and original parameters are
rechecked. Native/candidate seed storage stays separate but the full model is
shared, and ABBA is a warm-resident local screen, not a cold-memory bandwidth test.

`--run --plan ORIGINAL_E046_CHILD_PLAN --plan-sha256
7df6cff99425698a1fb134df4f25dc8de4e717482f73039e56ec5a0a00930003 --out NEW`
executes only under the external owner. The900s child deadline includes loading,
hashing and cleanup; it is not a measured duration. Root owns the exact pause,
resource, terminal-handshake, reap and restoration adapter separately. Existing
source/plan paths are never modified by this screen.

Successful complete receipt: schema `GDN-batched-local-timing-result-v1`,
status `PASS`, real pid/ppid, case_count=numerical_case_count=12,
timing_count=1200, candidate_calls=672, candidate_fallback_calls=0,
instance_hooks_restored/source_model_pins_rechecked/device_cleanup_completed/
local_timing_screen_complete all true. Partial failures retain actual outputs,
timings and complete.json with status FAIL. Whole-model/HTTP/performance/frontier
qualification flags remain false. A local timing win only motivates full-model
candidate qualification and measured HTTP comparison.

`test_screen_cpu.py` contains six small injected CPU contracts: inert default,
fixed first-capture selection, metadata rejection, exact balanced schedule,
fresh state and complete timed eval ordering, and native/candidate dispatch.
Actual root CPU run ecde54 passed all six checks; evidence is
`.artifacts/m5-ultra-256gb/gdn-batched-timing-screen-cpu-v1`. The independently
reviewed owner also passed thirteen checks (8ae18d), retained separately at
`.artifacts/gdn-batched-timing-owner-cpu-v1`. These CPU results do not qualify device execution.
