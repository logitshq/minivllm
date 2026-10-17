# E024 serving coverage and a separate ahead invalidation diagnostic

This is a CPU design note, not a registered or executed experiment. Keep the
frozen 37-request E023 workload unchanged. Its candidate ahead reuse, survivor
refill and partial retirement remain mandatory; discarded prior ahead is an
observed count, not an assumed event. `discarded_ahead_observed=false` means the
same-schedule result does not cover that invalidation path. The reference's
intentional per-step ahead drop is not candidate invalidation evidence.

The new serving qualifier has exactly five plan fields:

```json
{
  "schema": "E024-route-serving-plan-v1",
  "qualifier_sha256": "SHA of serving_qualification.py after freeze",
  "qualification": {"path": "absolute original E023 plan", "sha256": "21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb"},
  "sources": {"absolute source paths": "frozen SHA values"},
  "deadline_s": 1800
}
```

`validate` checks its own SHA, the exact source map, and both this deadline and
the inherited numeric deadline as integer 1800. This satisfies the existing
`cold_cycle/controller.py:run_plan` qualifier/deadline contract; the existing
single direct child owner retains its 1830-second work budget and bounded reap.
The parent must register the real source hashes and restored predecessor.
There is no new lifecycle implementation and no execution permission implied
by this note.

## Small independent diagnostic candidate

Prefer one controlled sampling-key change to inventing a request cancellation
API. `model_runner.py:ModelRunner.run` includes `top_k` in the native ahead key;
`sampler.py:sample` returns argmax before top-k filtering when all temperatures
are zero. Changing `top_k` from 0 to 1 therefore forces the real key-mismatch
branch without changing greedy sampling mathematics. `SamplingParams` is
mutable, but the engine exposes no live parameter-update API: this is an
explicit internal diagnostic action, not a supported HTTP feature.

Use a separate finite plan with B1/B8 and current used-length 1023/1024, so the
two route-preserving graph families are exercised. Inputs extended from fixed
D prompt IDs are diagnostic contexts, not complete real D512 generation.
Retain reference eager/gates OFF and candidate gates ON/norm OFF, E028 OFF, the
same original model/source pins, normal EOS, and the existing full-state
Comparator and Observer. Do not alter the 37-request plan or its results.

1. Admit equal requests independently through each engine's normal API. Each
   arm owns its own `SamplingParams`. Run normal steps until a completed step
   leaves a real carried future for the intended batch. Record that native
   ahead key and compare the full committed and future state across arms.
2. Between completed steps, change only each participating request's `top_k`
   from 0 to 1 in both arms. Record before/after parameter values, request IDs,
   positions and token inputs. Require the pending native ahead key to differ
   from the next scheduled key solely in this field. Do not clear candidate
   `_ahead`, alter queues or caches, or inject recorded output tokens.
3. Execute both real schedulers. Reuse the reference's existing eager/no-reuse
   step and candidate's normal runner. Compare exact schedule, full-vocabulary
   logits, all 128 state arrays, committed/ahead isolation, pre/post retirement
   state and raw delivered tuples. Require candidate consumed-prior to be null,
   discarded-prior to identify the actual old future, and an actual current
   decode call plus compiled-call witness for the intended graph family.
4. Leave top-k at 1 for the next normal step and require consumption of the new
   future, then drain normally. Restore the diagnostic sampling objects after
   draining, release both engines and restore the adapter on every exit. If
   normal EOS prevents any required event, report incomplete coverage; do not
   suppress EOS or count a nonexistent path. Preserve first failure and all
   step receipts. This is numerical coverage only, not latency measurement.

The original `qualify_native.py:Observer.run` already checks both the semantic
decode key and the full native sampling key before declaring reuse;
`Observer.finish` distinguishes consumed and discarded prior futures. A small
separate loop can reuse these functions and `pair_calls`, without replacing
their comparisons or constructing expected state from the candidate.

## Cancellation boundary

`LLMEngine` exposes add/step/generate, not coherent per-request cancellation.
The frozen HTTP `BatchWorker.cancel` skips a pending job, but an admitted job's
cancellation permanently fails that benchmark server. Neither behavior proves
that decoding survivors correctly resumes after active cancellation. Calling
`runner.drop_many` alone would leave scheduler/request ownership inconsistent.
Any future cancellation qualification must first define a real coordinated
engine operation, then apply it to both arms at the same completed-step
boundary and verify surviving state and retirement. It is excluded here.
