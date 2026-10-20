# 039 — make multi-model research falsifiable, then close the verifier interface

All three Logits models completed actual Codex repository review. Tool execution
and HTTP success did not imply correct analysis: DeepSeek proposed a false hash
fix, Kimi misstated reducer math, and Muse chose the wrong feature capture point.
Keep the original reports, validate claims against source and experiments, and
record retractions. Kimi also demonstrated an HTTP200/empty-answer failure when
reasoning consumed the whole token budget. Explicit reasoning settings and
nonempty content/finish-reason checks belong in the harness.

See the [complete failure record and worker results](../../experiments/m5-ultra-256gb/logits-workers-2026-10-20/README.md).
Future bounded reviews should receive an exact file packet plus current result
summaries. Frozen preparation READMEs describe historical state. A prompt-only
file-count limit does not enforce data scope; exact packets can.

E057 has correct local logits and selected cache prefixes, but DFlash needs
five intermediate target-layer outputs to propose the next block. The source
contract is `x = layer(...); capture(x)`, after the complete MLP residual.
Capturing normalized MLP inputs or final model-normalized values changes the
drafter input even if the final target logits match.

The new [E058 prototype](../../../bench/m5_ultra_256gb/rowwise_mlp_shared_features/README.md)
adds an optional fixed five-layer capture, reusing the same forward. It also
separates greedy acceptance from state installation: n accepted drafts produce
n+1 outputs including correction/bonus, while the retained input prefix contains
the old anchor plus n drafts. The newly emitted correction remains pending.
Rejecting the first draft still produces one token; accepting all drafts produces
T. Keep features and cache prefixes aligned at every rejection position.

This is a CPU-tested source checkpoint, not a GPU or engine result. Next prove
real feature bytes, every retained state and continuation under actual draft
tokens. Then measure complete round cost divided by actual outputs, and streaming
latency. Do not infer a new Pareto point from the existing local verifier ratios.


Root-review erratum: the follow-up prompt incorrectly asserted that E055's
OFF spread uses max/min-1. Source inspection confirms range/median, as Kimi's
first review stated. The two formulas are not equivalent. The incorrect prompt,
model replies and a numerical erratum are all retained; reviewers must check
their own corrections against source too. E058 CPU counts: 7/7 initially, 8/8
after adding shared-head coverage; no device run.
