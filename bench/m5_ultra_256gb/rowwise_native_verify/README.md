# Layer-major native B1 verifier — source-only hypothesis

`candidate.verify(..., enabled=False)` returns `None` without touching inputs.
The opt-in function is a dependency-injected research seam, not a runtime entry
or an installed speculative decoder. It has not been executed or qualified.
An external child must bind the actual frozen model/source/runtime and compare
it against independent original sequential C1 decode.

The current prefill verifier changes the native target's matrix row count,
convolution accumulation, GDN update and attention path. The separate exact
verifier bridge will measure those differences. This alternative instead
orders the original B1 work by layer, then token. Each row retains the native
one-row projection, normalization, GDN recurrence and attention calls. Absolute
positions/lenses advance per token; the supplied canonical state is untouched.
Every retained prefix has a separate KV wrapper before the next prefix writes;
whole convolution/recurrent arrays are carried forward without slicing away
padding or rejected KV tails. All 64 layer states are retained for each prefix.

Potential saving: weights for a layer may be reused by adjacent one-row calls,
and a future compiled block may amortize host submission. This does not prove
physical cache reuse or a speedup: MLX schedules the lazy graph, weights may not
fit cache, and this prototype adds concatenations and retains up to eight full
state versions. No launch-count, bandwidth or memory-peak result is asserted.
It currently has no compiled wrapper and no performance qualification.

The supplied token block is fixed teacher input. Predictions never feed the
candidate. Supported shape is one sequence, T2/4/8, complete native Qwen64-layer
B1 state and positions ending within1536; unsupported inputs return `None`.
No B2 multi-sequence equivalence is claimed: E027 showed native multirow and
independent M1 can differ, and is preserved as a negative result. This prototype
intentionally targets the original **single-sequence C1** arithmetic contract.
It does not change the original prefill verifier or relabel that path as exact.

Before any integration, compare every T-row full-vocabulary logit and every
complete state at every prefix against independent original sequential decode,
including windows crossing1023/1024. Check source/model identity, all parameters,
input preservation, gates ON/norm-reuse OFF, fresh contexts and no new eval.
Then compile only as a separate experiment and repeat the same checks. Device
compiler fusion and attention dispatch can still change arithmetic; source
ordering is not a numerical proof. A useful verifier must beat its equivalent
C1 forward cost before adding drafter, acceptance or publication overhead.

Acceptance, EOS retirement, draft features, rollback/commit, scheduler/ahead,
HTTP and performance remain unimplemented and unqualified. Retained prefixes
are only outputs here; no accepted prefix is published into a request cache.
