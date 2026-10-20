**E057 Review**

**Measurements**
- `screen.py` pins `AUDITOR_SHA = 3ce4…`, but the actual E057 auditor SHA is `09e9…`; `audit.py:23` correctly pins the current screen SHA `2922…`. Source inspection therefore identifies a likely pre-timing blocker, not a benchmark result.
- `screen.py:59` validates the canary auditor pin, and `screen.py:60–70` correctly binds `e056_audit` to `AUDITOR_SHA`. Updating the pin would conflate the two auditors’ provenance unless the E056 audit’s own pin remains the canary audit.
- The `rowwise_mlp_shared_analysis_v1/audit.py` file SHA `3ce4…` matches the canary auditor pin. It does not match the E057 `rowwise_mlp_shared_cost_analysis_v1/audit.py` SHA `09e9…`.

**Correctness**
- The current wrapper does not modify frozen sources; it creates a private `FunctionType` scope and a shadowed parameter-hash helper: `screen.py:95–126`.
- The candidate receives the live shared projector on both the head and MLP paths: `screen.py:112–121`.
- `CountedMLP` checks the projectors in layer order across gate and up (`screen.py:76–92`), which supports the claim that all 128 projection calls are fed in qualification. This still does not inspect execution at runtime, and the observer is only used during qualification (`screen.py:114`) and not during timed collection.
- The six qualification runs, 16 raw comparisons, and pre-timing parameter check are required before timing (`screen.py:103–110`, `screen.py:144–151`). The exact full-history oracle checks are therefore positioned correctly in the pipeline.
- Source intent is to preserve rowwise SiLU/down/residual (`candidate.py:125–145`). This is structural evidence; it does not prove exactness independently of the oracle checks or prove the script actually runs after the pin fix.
- A candidate output can pass all 128 observer checks and still differ downstream. Final correctness depends on the full oracle comparisons actually passing, not on the MLP-call count.

**C1 Impact**
- E057’s intended scope is local verifier and commit cost only (`README` scope and `screen.py:43–45`). It explicitly excludes drafting, acceptance, HTTP, compiled execution, full-model execution, and Pareto claims.
- The design is likely to reduce local verifier cost in C1 only if the shared gate/up-plus-head path avoids enough redundant projection/dispatch/lowering work. This is a hypothesis: no E057 timing evidence was read or produced.
- E054’s `1.430x T4 / 1.550x T8` result is prior evidence for a head-sharing/native-MLP verifier. E057’s MLP-sharing extension is not established by that result. E056 proves only a 12-pair projection canary; it does not prove all-layer propagation.
- The missing historical head-only paired arm (`screen.py:44`) prevents E057 from quantifying its incremental benefit even locally.
- No evidence supports a claim of universal fastest, or of end-to-end superiority over oMLX/Lithos.

**Concerns**
- Fix the auditor SHA and verify the E056 audit pin still resolves to the canary auditor. Do not simply replace every occurrence of `3ce4…`.
- The `adaptive_device` wrapper may fail before training if the changed source does not load; resolve the pin issue first.
- The MLP observer is a Python producer receipt, not a GPU dispatch/kernel-count measurement (`screen.py:92`).
- The C1 comparison should use the same workload, seed, tokens, cache state, scheduler, and exact output/state equivalence.

**Next Experiment**
Add a paired A/B arm for C1 decode at T4 and T8: original verifier/commit versus E057 shared gate/up-plus-head verifier/commit, with identical workload/seed/cache/scheduler. Measure per-step verifier wall/device time, draft time, accepted tokens, commit/rollback, and HTTP latency/throughput. This should falsify the hypothesis that local MLP-sharing lowers full C1/per-step cost, rather than only reducing isolated kernel time.
