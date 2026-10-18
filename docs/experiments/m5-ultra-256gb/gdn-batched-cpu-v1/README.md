# GDN batched prework: actual 12-test CPU evidence

The first full `gdn_prework_batched.test_candidate_cpu` run passed **12/12**:
zero failures/errors/skips/expected failures/unexpected successes, no forbidden
operations or backend modules, and **14 unchanged source snapshots**. No source
fix or retry was needed. Full stdout/stderr, per-test/subtest results, the actual
guard runner, source copies, prerequisite snapshots and original inventory are
retained. The suite took 0.006 s (guard procedure 0.01524 s); neither is a device
performance result.

The guard denied all socket creation, real subprocesses, signal delivery/timers,
backend imports and model payload opens. It admitted execution only after the
prior formal cycle was closed, independently restored and archived. That prior
formal audit's FAIL remains unchanged. **This archive grants no numerical,
full-model, HTTP or performance qualification.** The candidate/README were not
changed after their source-prepared freeze; the later CPU result is recorded
separately in [RESULTS.md](../../../../bench/m5_ultra_256gb/gdn_prework_batched/RESULTS.md).

- 33 regular members; 346,663 raw bytes;
  53,456 compressed bytes.
- Archive SHA256: `bee93b3578848f15cce88cedf41c7ff362e2aa8cfb2a94eefacab7960a440dd5`.
- CPU result SHA256: `3d4451c6ab7030e9996edb1a8741eb645b45354a61d6ff83c1858af487ed2aa2`.
- Guard runner SHA256: `b5c7600f515182c07edfcf967c3233aa4aec7d35783f8eb1b020506463904bf3`.

Every tar member was safely extracted and checked for exact size/SHA256;
all original bytes and tree membership were rechecked. No tests were rerun by
archiving. [Manifest](manifest.json) and [verification](archive-verification.json)
retain these bindings. `python -B verify.py` rechecks the complete archive and
sidecars independently using the standard library, with no service/device calls.
