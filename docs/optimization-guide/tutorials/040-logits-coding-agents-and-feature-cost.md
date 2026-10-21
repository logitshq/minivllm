# 040 — let model agents write code, then make their claims survive execution

E058 turns the three Logits profiles into real coding workers. DeepSeek wrote
the token-major feature reference and candidate interface, Kimi the independent
raw-byte auditor, and Muse the rotated timing collector. Root owned integration
and one serial M5 GPU queue. This division preserves independent checks without
letting concurrent GPU jobs contaminate timing.

The optimization is small: reuse E057's shared MLP gate/up and vocabulary
projections, then retain complete decoder outputs at layers5,19,33,47,61 during
that same forward. The drafter wants `[T,25600]`: concatenate five `[T,5120]`
arrays on the hidden axis. The target's final normalized state is not a substitute
for these intermediate features.

The first DeepSeek implementation used the wrong concatenation axis. Its test
repair loop then became entangled in custom fake heads and internal call counts.
Root stopped the loop, retained its failures, and tested the corrected module
against the repository's existing scalar model and independent recorded native
traversal. Compare values, positions, every cache prefix and continuation;
counting concatenation calls is a brittle proxy for the desired behavior.

The actual device gate uses one fixed Qwen3.8-27B NVFP4 seed at used length1023.
T4/T8 and every k=1..T produce12 selected-prefix pairs and12 native continuation
pairs. Each contains full logits, the five-layer features and128 cache arrays,
including padded KV storage. The reference first matches the original model's
actual capture forward. Source/weight checks, parameter hashes and canonical
state checks bracket measurement. These checks cover this seed and these widths,
not arbitrary prompts or a real drafter's rejection distribution.

CPU fixtures missed an actual serialization detail: MLX writes
`__metadata__: null`. Kimi's first auditor rejected that as malformed tensor
metadata. Its revised parser handles only the reserved metadata key separately,
while preserving exact tensor names, dtype/shape, byte lengths, finite bits,
SHA256 and full byte equality. Never "repair" the raw output to make an audit
pass. Preserve the failed auditor and name the new version explicitly.

Muse's collector initially failed to visit all three rotations and swallowed
logging exceptions. After repair, five groups × two repeats × three arms × two
widths yield60 timed calls after12 warmups. The native baseline uses original
model capture on supplied teacher tokens. Every timed arm materializes its full
logits and selected state; feature arms materialize captured features too. This
is a different scope from E057's device-dependent autoregressive baseline.

Postprocessing also needs review. Muse's first analysis script called itself
recursively while trying to capture its own exit code. Root found no remaining
matching processes, preserved that script, and ran a corrected single-pass
version under an external30-second timeout. A child cannot certify its own
launch by launching itself again. The numerical reducer and raw evidence remain
separate from process provenance.

The result is a usable local feature path, not a serving victory. The next
experiment must run a real drafter, verify, select the matching feature/cache
prefix, and install it while retaining the correction/bonus as the pending next
anchor. For round i, let C_i be measured draft + verify + selection/commit +
scheduling seconds, and n_i be the number of outputs actually emitted after
EOS/output-cap truncation. Then:

`speculative seconds/output = sum(C_i) / sum(n_i)`

`speedup = native seconds/output / speculative seconds/output`

Before truncation, accepting a_i drafts emits a_i+1 outputs. Do not add a
hypothetical extra forward for the correction if it remains the pending anchor;
measure the executed work. DeepSeek's initial follow-up added a unitless1 to
time and omitted the baseline denominator; that feedback was rejected and
corrected too. Using multiple models creates more hypotheses and code, not an
exemption from units, byte checks or full-loop measurement.

See [the result, complete failures and figure](../../experiments/m5-ultra-256gb/E058-logits-agent-feature-run-v1/README.md)
and [the isolated source](../../../bench/m5_ultra_256gb/e058_agent_run_v1/README.md).

Observed E058 result: T4 native93.00079ms versus shared+features59.75785ms; T8 native185.64746ms versus110.11562ms. The median of within-group ratios was1.56014×/1.68432×. All3120 tensor pairs passed the v2 byte audit. These are verifier/capture costs only.
