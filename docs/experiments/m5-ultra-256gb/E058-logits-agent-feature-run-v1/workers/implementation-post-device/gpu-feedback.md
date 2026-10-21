# GPU feedback

Acknowledged: the reference had to concatenate the five captured post-layer states on the hidden axis (bf16 `(T,25600)`), not stack/concatenate them on the token axis. It also had to keep the native layer call and full KV/conv cache snapshot per prefix, plus untouched canonical context.

My CPU scaffold was unreliable because it paraphrased the native traversal with custom fakes and brittle concat/call-count assertions instead of using the proven fake models; it could pass on the wrong axis layout while diverging on real allk semantics. Root's tests based on proven fakes are the better gate.

**Next experiment:** serving-level speculative decode versus plain decode on the same M5 Ultra, same prompts, greedy, fixed max tokens, measuring wall time and actual emitted tokens.

Per round: `cost = draft + (1 if verify) + correction + bonus + rejection_restart`, where `draft` is the measured actual draft cost (not a projected ratio), `verify` is the measured verifier cost for the selected prefixes, and correction/bonus/restart each count one native token step. Report `speedup = (Σ tokens_emitted / Σ round_cost)` relative to plain decode; keep the measured accept distribution. The 1.56/1.68 ratios are verifier-cost only and do not predict this.
