# Long decode host work: source review and one-pass candidate

Source-only review during formal measurement. No backend imports, tests, hashes,
device profiling or performance measurement were run for this change. Existing
E024 and all active formal sources remain untouched. This is a measurable CPU
overhead hypothesis, not a finding that this work is the main bottleneck.

## What executes on each call

The actual native plan selects `.venv` MLX 0.32.3. Its installed
`mlx/nn/layers/base.py:12,99–115,280–283,325–363` shows that Module is a dict,
array assignments write its slots, and `parameters()` constructs a filtered tree.
The candidate does not call that Python `parameters()` method on each step.

Frozen `route_decode_graph/candidate.py:63–98` checks context/token metadata,
64 layer identities and every layer's two state arrays. Lines 105–107 scan GDN
flags; line 119 builds the explicit state list; lines 120–126 validate the whole
returned structure then assign states. `_m5_ultra_decode_identity` in the native
GDN source is `lru_cache(maxsize=1)`: it is not an uncached device-info probe on
each step. No guard reads tensor values or calls eval.

The wheel does not include the Python binding's C++ implementation. The following
claims use official **matching-tag source**, not a new binary provenance audit:

- [v0.32.3 transforms.cpp:429–550](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/transforms.cpp#L429-L550):
  each compiled call flattens explicit arguments and the current captured model,
  including cache hits. During tracing, captured slots temporarily receive tracers
  and are restored afterwards. This is not a per-call Python model forward.
- [trees.cpp:146–166,234–246](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/trees.cpp#L146-L246):
  captured flatten recursively visits list/tuple/dict contents and retains arrays;
  it also visits non-array leaves, which are ignored in non-strict mode. A Module
  is handled as a dictionary, not through its `parameters()` method.
- [compile.cpp:326–360,985–1046,1069–1115](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/compile.cpp#L326-L360):
  cache lookup compares input ndim/shape/dtype. A hit still replaces trace roots
  and builds a new lazy graph. The Python tracing function runs on a miss, not
  every hit. These operations have no measured contribution here.

For the fixed untied-head model, source-derived array-leaf count is
`5 + 16*18 + 48*22 = 1349`: embedding/head/final norm, attention blocks and GDN
blocks respectively. With 128 state arrays plus tokens/positions/lens, an eligible
long callable receives **1480 arrays**. This is static structure accounting, not
a runtime count or timing sample, and excludes any future added array members.

## The isolated first candidate

`long_decode_onepass/candidate.py` combines layer identity checking, all original
cache guards, GDN flag collection and explicit state collection in one loop.
It retains every dynamic condition and the same bound selection. Public
`build(model, enabled=False)` returns the model untouched; ON verifies the frozen
route source and subclasses its unchanged constructor/runtime loader. There is
no runner installation or lifecycle framework.

The forward trace body, `inputs=model`, KV handle views, complete returned-tree
check before mutation, and state assignments stay unchanged. Parameter captures
are not replaced by a cached parameter list. This only removes repeated Python
collection passes; it does not remove the compile binding's model traversal.
The added tuples/lists and branch ordering may offset the savings. Model structure
must remain stable within a call, as it already must under the original contract.

The 11 CPU test methods are source only, unexecuted: original-vs-new accepted and
rejected domains, late layer replacement/reordering, dynamic flags, live-model
capture, protected KV handles/all-state updates, invalid return atomicity and
compiler failure propagation. The fake compiler cannot qualify actual MLX
parameter replacement. The trace-body AST test protects the arithmetic boundary.

## Why prebound weights or an epoch is not the same small change

A copied flat array list is not a live view of the model's slots. Filling that
list with tracers leaves the model pointing at original arrays; tracing can reject
uncaptured inputs. Removing captured weights altogether instead risks stale
constants after replacement. A correct flat-slot design would need explicit
tracer binding/restoration plus current-slot reads, or a new managed mutation API.

Installed Module has no model-wide mutation epoch. Attribute assignment,
`Module.update`, direct dictionary writes and nested list edits are different
paths. An application counter alone cannot detect arbitrary replacement. Updating
only `__setattr__` is insufficient; wrapper identities can also change during
normal compilation. The narrow one-pass candidate therefore adds no epoch and
retains existing parameter semantics. A future immutable-weight session must
explicitly constrain writes, refresh at a drained/no-ahead boundary, and get its
own qualification; it cannot inherit the original replacement proof by wording.

Original E024 serving's **42 direct calls** remain evidence for its frozen source:
6 baseline, 18 changed and 18 restored, at used1023/1024 and B1/B8/B1, for
`norm.weight`, first GDN `in_proj_a.weight` and `.scales`. They are not 42 distinct
parameters and not qualification of this new candidate. Run that unchanged gate
again on any future candidate, then the original37/full-state/ahead and actual
HTTP gates before formal timing.

## Minimal measurement after authorization

Current `c1_pipeline_profile/instrumentation.py:83–88,130–139` records nested
`compiled_callable` and `decode_graph_submission`. Their interval difference
bounds surrounding guard/key/state-pack/writeback work **plus observer overhead**.
The callable interval includes binding flatten, cache lookup, lazy graph creation
and occasional tracing; it cannot isolate flatten cost. Invocation count is not
an exact MLX specialization counter. Separate first calls and route/shape changes.

First profile the unchanged route with the existing OFF/runner/full diagnostic.
If the outer remainder warrants work, use a new explicit diagnostic adapter to
compare old/new one-pass on the same request history, flags and original gate.
The frozen profiler intentionally verifies the old class source, so an unknown
new subclass must not be silently admitted or impersonate that source. If only
the callable interval is material, this candidate is not a solution to that
interval: investigate a separate binding experiment. No per-step synchronize,
additional tensor reads, or sum of overlapping intervals as exclusive GPU time.

## Explicit profile wiring proposal, not implemented or executed

Use a new diagnostic entry with `--long-guard original|onepass` (default
`original`) and the existing `--profile off|runner|full`. Require route graph ON;
hold GDN prework, short/fuse/gates, norm, transport, request history and warm count
identical between the two arms. Both arms use the same new entry and drain
controls. This is not a change to any formal server or qualification receipt.

The frozen profiler's `install()` must remain unchanged. Its `_verify_class()`
correctly rejects this new subclass: `OnePassDecode.__init__` is inherited from
the original class, whereas `_prepare` and `__call__` come from the new factory
file. Renaming the class, substituting a source digest, or skipping that check
would conceal the actual callable. A new narrow installer should instead:

1. Verify the pinned profiler, runner, cache, short decoder, route integration,
   original candidate and onepass source. Require the original fresh, empty,
   target-only runner and unused graph dictionaries. Reuse the existing checks
   for these original objects before changing the decoder. Verify the runner's
   actual run function and MLX backend identity; reject unknown run/decode wraps.
2. Let the unchanged combined installation create the original dispatch. In the
   onepass arm only, build the new decoder through the verified `build()` factory
   and retain that exact returned class/object identity. Verify its exact original
   base and inherited initializer against the original file, and its `_prepare`
   and `__call__` globals/code filenames against the new file. Check the same
   model, enabled state, bounds, runtime receipt and empty callable dictionary.
   Save the original `dispatch.long`, then replace only this instance's `long`.
3. For **both** arms, call the same frozen low-level
   `Session(runner, fn, recorder, [('long', selected), ('short', short)], sources)`.
   Its private mx/sample proxies, `_Functions` and `_Invocation` produce the same
   `runner_step`, `decode_graph_submission`, `compiled_callable`, callable-family,
   sample and existing-eval spans. Use the identical `long` graph label and key
   semantics. Do not wrap `_prepare` separately or call eligibility a second time:
   that would add observation work only to one arm. In profile OFF, keep the same
   decoder selection/control path but install no Session or recorder.
4. After original DONE flushes, explicit drain, joined handlers and stopped
   worker, restore Session first. Clear only the new decoder's compiled cache,
   restore the original `dispatch.long`, then restore the unchanged combined
   installation. This extra scope is needed because combined cleanup owns the
   original dispatch; restoring it first would leave the new graph references
   outside its normal cleanup. Construction failures roll back in the same order;
   foreign wrappers or active threads remain a failed closure, not a successful
   profile. Drained still does not mean discarded ahead GPU work is synchronized.

For the HTTP entry, reuse the frozen `Drain`, handler/submit wrappers,
`worker_loop` and `finalize` bodies. The existing entry has no injection parameter
for decoder selection. A small new orchestration entry can copy its startup/main
body with only argument selection, installer call, source/identity metadata and
the extra cleanup scope changed; an AST comparison should limit those changes.
Do not copy the HTTP inference loop or lifecycle into a new framework, patch
shared module globals, or misreport the old file as the new executable. Record
the actual entry, adapter, selected class and every dependency in the new identity
and profile source map. The numerical/performance flags remain false.

Before any device run, focused CPU tests should cover both arms' identical span
names/order on a fake backend, the selected compiled callable really being
wrapped, unknown subclass/wrapper rejection, profile OFF, construction rollback,
and Session→decoder-swap→combined cleanup with no retained model reference.
Then run the new candidate's exact parameter/state/HTTP gates. Profile spans
alone cannot qualify it; no such tests or implementation were run in this review.
