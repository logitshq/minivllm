**Retractions of prior claims**

I retract: (1) recommending E056/E057 as completed adoptable wins - `docs/experiments/m5-ultra-256gb/E057-c1-mlp-shared-full-cost-v1/README.md:12-13` states `All T tokens are assumed useful for these local costs. Draft generation, rejection probability, rollback scheduling and HTTP costs are excluded` and `This is a prerequisite for a speculative serving experiment, not a new Pareto point or a win over oMLX/Lithos`; (2) calling cross-schedule token ID differences a bug - `docs/experiments/m5-ultra-256gb/E055-c8-batched-formal-v1/README.md:14` states `The same-schedule prerequisite covers the E053 captured history, not every formal schedule. Cross-schedule output IDs differ. No arithmetic error is inferred from that fact alone`; (3) `saved_bytes * bandwidth > time` - dimensionally wrong. Correct is `time = traffic_bytes / bandwidth_bytes_per_s`.

**Verified defects vs evidence**

* Interface gap: `minivllm/engine/spec_runner.py:46` expects `logits, new_feats = self.model(..., capture=self.capture)` where `self.capture = tuple(drafter.cfg.target_layer_ids)` `spec_runner.py:24`, and feeds drafter via `history` `spec_runner.py:30-40` and returns `produced, new_feats[:n+1]` `spec_runner.py:58`. `bench/m5_ultra_256gb/rowwise_mlp_shared/candidate.py:38` defines `verify(model,tokens,context,runtime,*,enabled=False,head_project=None,mlp_input_project=None)` with no `capture` param and returns `Output(logits,prefixes)` `candidate.py:32-35,140,145` - it does not supply target-layer hidden features. Without extension the shared branch cannot feed the drafter's `history`.
* Cache ownership mismatch: candidate builds independent per-row `prefixes = [list(context.caches) for _ in range(T)]` `candidate.py:101` and `BatchKV/BatchState` copies `candidate.py:111-113`. SpecRunner uses `c.snapshot()` / `c.rollback(n+1)` `spec_runner.py:44-45,54-55` in place. Wiring requires one ownership model, not both.
* Prior break-even formula inverted dimensions. Must compare time vs time.

**Scope limitations (not defects)**

E057 is local verifier cost only: 400 timed calls (200 device-baseline plotted) `E057:10-11`, pooled p50 T4 58.94ms vs 85.79ms, T8 107.88ms vs 168.91ms, but `All T tokens are assumed useful` `E057:12`. E055 is C8 batched GDN only, `Three repeats at C8 are not a concurrency sweep` and `+2.11% throughput ... not met ... Keep as diagnostic gain, not adopted` `E055:11-12`. `DefaultOFF and modular: shared MLP, shared head and C8 batchedGDN remain independent` `E057:14` - C8 can keep native decode. `SpecRunner solo path: batch >=2 speculative not worth it` `spec_runner.py:2-3`. Candidate admission is strict fixed shapes `candidate.py:49-63,77-86` - must fall back to native on unsupported shapes.

**Hypothesis**

If acceptance rate `E[n]` is measured, the shared verifier's per-verify time saving may be offset by draft cost and rejected tail wastage. No claim until actual `n` distribution is observed.

**One next C1 speculative-round experiment**

Reuse `minivllm/engine/spec_runner.py:29-58` accept path unchanged, only replace its verify backend for C1:

1. Extend `candidate.verify` to accept `capture` and emit `new_feats` in the same order as `drafter.cfg.target_layer_ids`, from the same forward that produces `logits`. Capture correctness: capture must be the post-attention-layernorm `h` `candidate.py:119-121` or final `normalized` `candidate.py:135` at those target layers for positions `anchor ... anchor+T-1`, sliced to `n+1` accepted positions. Verify by asserting `new_feats[:n+1]` equals a reference `model(...,capture)` on `anchor + accepted drafts` prefix case for six selected-prefix/continuation cases as in E057 audit `E057:3`.

2. Anchor: `req.all_token_ids[-1]` at `start = req.num_computed_tokens` `spec_runner.py:34-35`. Drafts: `drafter.draft(feats, anchor, start, ... bs)` `spec_runner.py:42` where `bs = self.block or (5 if st.q>=0.45 else 3)` `spec_runner.py:41` and `block=[anchor]+drafts` `spec_runner.py:43`.

3. Accepted-prefix cache: retain only `anchor + drafts[:n]`. Implement via SpecRunner's `snapshot/rollback(n+1)` `spec_runner.py:45,55` reusing cache; if using candidate's `prefixes`, select `prefixes[n]` as committed state and discard others. Update `st.q` `spec_runner.py:57` and `history` with `new_feats[:n+1]` `spec_runner.py:58`.

4. Correction/bonus token: `post[n]` where `post=argmax(logits)` `spec_runner.py:47-49`, `produced = drafts[:n] + [post[n]]` `spec_runner.py:53`. `next anchor = bonus` per `spec_runner.py:7` comment.

5. Edge cases: full-rejection `n=0` => `produced=[post[0]]` single correction, rollback to `anchor` only, history gets 1 feat; all-accepted `n=len(drafts)` => `produced=drafts+[post[n]]` `bs` tokens, history gets `T` feats, bonus may be EOS.

6. Break-even: per round measure `T_round = T_draft + T_verify_shared + T_commit` including host-sync, and `k = n+1` actual emitted tokens. Break-even iff `E[T_round] < E[k] * T_native_decode_per_token` where `T_native_decode_per_token` is solo native decode median. Separately, memory-bandwidth saving time is `saved_traffic_bytes / bandwidth_bytes_per_s = time`, not `*`. No synthetic T counting; rejected tail contributes 0 to `k`. Preserve native fallback when admission guards `candidate.py:49-63` fail.
