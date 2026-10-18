# Exact greedy verifier: a separate algorithm direction

This is a source review and proposed diagnostic, not an implementation, device
result or extension of E027 qualification. Existing target-only and speculative
sources remain frozen. The combined GDN/graph HTTP and timing sequence continues
first; this work must not contend with its device execution.

The desired large improvement is fewer target-weight passes per **committed**
token. A better drafter alone cannot establish this: its verification and rollback
must preserve the intended target generation, and their combined cost must be
less than ordinary autoregression for the same number of committed tokens.

## What the current code actually does

`SpecRunner.step` snapshots staging caches, sends `[anchor, drafts...]` through
`ForwardContext('prefill', caches)`, then rolls back to the accepted prefix. That
path uses a multirow projection, prefill attention and chunked GDN; target-only
C1 decode uses its own matrix-vector route, per-position attention masks and the
recurrent GDN decode path. Their mathematical equivalence is not an original-byte
proof. The E027 native bridge independently observed five differing BF16 values
for one multirow projection; it did not execute a complete speculative verifier.
It is insufficient evidence to claim either a final-token divergence or exact
whole-model equivalence.

The DFlash2 source's unconditional statement that greedy output is identical
therefore remains an unqualified claim for this fixed NVFP4 M5 execution path.
Do not alter the active source to fix that prose while source-pinned experiments
are running; record the qualification limit here first.

## First useful device diagnostic

Use a fixed actual committed C1 state and a short fixed teacher-forced block.
Keep each row/token and absolute position explicit. Independently clone the same
state, then compare the existing native multirow verifier against repeated
original C1 decode calls consuming exactly the same block. This is an oracle
bridge, not a performance baseline replacement. Preserve full original-dtype
logits and recurrent/conv/KV state; stop at the first difference and retain both
sides. If necessary repeat only the failing block with layer observations to
locate the first distinct operation. Normal EOS handling, accepted-prefix
rollback and subsequent continuation require separate end-to-end checks.

No ReaDy reader, random draft weights, extra corpus or acceptance claim is needed
to diagnose this boundary. Existing correct request-local proposals can supply a
later draft; the target verifier must first have an explicit numerical contract.

## Candidate if the bridge fails

A new default-OFF verifier could batch only independent projection rows while
preserving each ordinary C1 reduction. E027's shared-weight M1 kernel is relevant
as a *local component hypothesis*, not as a native multirow replacement: its
complete operator geometries and real activations still need qualification.
At each layer, project the already available token activations together, then
retain the original per-token recurrence and attention operations in position
order. Attention must use the same valid-history extent, mask and arithmetic
route; stacking queries into a new attention kernel is another unqualified
change. Pointwise/RMS operations also need their actual shape-route checked.

Intermediate per-position states can initially be retained explicitly. For this
model, recurrent state alone is144MiB per sequence, so five retained boundaries
are720MiB before conv/KV and live temporaries. These are dimension-derived byte
counts, not measured total memory or evidence of a faster implementation. On a
256GiB machine, establishing correct accepted-prefix selection is preferable to
premature state compression. A later storage optimization remains independent.

This algorithm would be compared to ordinary C1 greedy continuation, including
all outputs, states, rejected suffixes and EOS/cap. It must not repair E027's
failed drop-in result, silently redefine that earlier baseline or promote the
old passing shapes. Both old results stay as recorded.

## Decision from measured cost

For average accepted draft count a, a necessary local condition is
`t_draft + t_verify + t_state < (a + 1) * t_AR`. This omits queueing and prefill,
so passing it still does not imply an HTTP Pareto improvement. Measure the actual
sum, preserve acceptance distribution and normal-EOS output counts, and only
then compare complete-request latency and committed-token throughput. This is
the reason to establish the verifier before integrating ReaDy's trained reader.

Sources inspected locally: `tutorial_code/miniengine/engine/spec_runner.py`,
`model/attention.py`, `model/gdn.py`, and `spec/dflash2.py`; earlier evidence:
`docs/experiments/m5-ultra-256gb/E027-native-bridge-v1/` and the E027 tutorial.
No backend/model was loaded for this review.
