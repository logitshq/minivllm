# Greedy verification width: source review and Apple GPU priorities

Scope: read-only review of DeepInfer `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` and Minivllm's rowwise shared-feature benchmark. No GPU run was made here. The official measurements below are theirs; Minivllm numbers and gains are unknown.

## Official evidence

- [QA §Speculative decoding, lines 239–251](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L239-L251): fixed-5 wastes verification; DSpark predicts per-token acceptance with a Confidence Head, drafts 5, and uses a profiled throughput curve with expected acceptance to choose the verified prefix. Official EP64/DP64 measurements: verified length 5.00 → 3.2–4.4, accepted length 3.20 → 3.29 (text says 3.29 vs 3.20), and estimated throughput +4–7% at ~60 sequences/rank. Table spans −1.5% at 16 to +11.0% at 96, so the gain is load-dependent, not a universal speedup.
- [QA §Speculative decoding, lines 262–264](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L262-L264): asynchronous schedule fixes the total budget first; the returned confidence then drives cumulative-product/top-k selects and remap on GPU. Acceptance information is stale during lookahead.
- [QA §Asynchronous scheduling, lines 266–275](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L266-L275): overlap forward/sample with next schedule/post-process; avoid a host sync between steps.
- [`mtp_varlen.hpp`, lines 19–41](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp#L19-L41): sampler cost is a median of 9 samples, enabled after 5, keyed by sequence count with a fallback.
- [`mtp_varlen.hpp`, lines 44–129 and 226](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp#L44-L62): forward slope is fitted from `(forward_ms - baseline_ms)/graph_size`; 5000-step window, at least 10 samples per graph, median of medians, nonnegative slope.
- [`mtp_varlen.hpp`, lines 150–207](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp#L150-L207): empty curve means verify all drafts. Acceptance rates come from the MTP confidence head. For lookahead, N−2 rates set total budget and N−1 rates allocate query lengths; the head must not let confidence of later drafts gate an earlier draft. Counters distinguish paired post-process from one-ahead scheduling.
- [`mtp_varlen.hpp`, lines 225–265](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp#L225-L265): expected tokens start at one anchor, add one per verified draft, and maximize `expected/time`. Time is `baseline + slope*graph_size + sampler`, but the baseline is chosen at the first curve breakpoint at or above the current query count. All requests receive an anchor and only the highest-ranked drafts consume the chosen budget. `update_runtime` looks up the curve by graph size without fully guarding an empty/missed curve.

## Minivllm behavior and missing inputs

- `rowwise_mlp_shared_features/candidate.py` verifies T = 2/4/8 inputs and returns a private cache prefix per input plus five fixed-layer features. It does not mutate canonical state.
- `rowwise_mlp_shared_features/round.py::select_greedy` compares drafts with the posterior, accepts the longest exact-match causal prefix, emits the correction/bonus but does not compute it, and returns prefix/features only through `count`. Width is fixed to 2/4/8; there is no dynamic selector.
- Missing trained confidence head: no per-draft acceptance probability can be claimed. Use observed match history only as a transient cost model, never as a trained predictor.
- Causal censoring: if draft position `j` mismatches, later matches carry no acceptance signal. Online stats need `(attempted_j, accepted_j)` with `accepted_j` defined as “all positions ≤ j matched and input had depth > j”; only `j < accepted+1` is observed for that round. Never backfill later positions as failures.
- DFlash noncausal block: keep width 8. Verification may use target prefix 4 or 8, but acceptance remains the causal greedy prefix; a later noncausal match cannot rescue an earlier mismatch.
- Full round costs: include embedding, all layers, attention/GDN state, MLP shared projections, norm/head, argmax/copy, private-cache/feature commit, and any round-to-round sync. A kernel-only time or last-graph baseline is insufficient.

## Prioritized Apple GPU optimizations (first principles, unmeasured)

1. **Cost-aware greedy width (prefix 4 vs 8).** Define candidates `{4,8}` as causal verification depths over an 8-token DFlash block. For each request, simulate selected depth `d` in per-position order; for observed or censored positions use a conservative anchor/prior, and estimate accepted-plus-bonus token count. Full-round cost `C_d(n_anchor, n_extra, context)` should be measured as wall time for the same round (including features/commit); never charge partial block rows. Select `d` only when `token_estimate_d / C_d` improves over both `d=4` and `d=8` with an explicit tie/failure fallback. On unified memory, larger unified batches raise row work and memory traffic, so 8 is not free.
2. **Measured, smooth cost model.** Fit nonnegative intercept and per-row/per-context slopes from actual round timings, using a small rolling median and cold-start bounds; evaluate each discrete 4/8 candidate at its actual row count instead of charging a whole block or the first breakpoint. Include commit/feature traffic and synchronization. Keep the selector OFF until measured.
3. **Fuse/cut round-boundary traffic.** Avoid a host argmax→Python scan→relaunch path; keep greedy reductions on GPU and commit only the causal prefix's KV/GDN changes and features. Features already carry 25600 BF16 columns per row (5 layers × 5120), so accepting fewer rows can reduce store traffic; whether that matters is a profile question. Async lookahead must never commit unaccepted rows.

## Distinction and limits

- Official: variable-length acceptance/throughput measurements above, from their hardware/benchmark.
- Ours: source-derived failure modes and design hypotheses. No Minivllm throughput, latency, memory, or acceptance claim; needs root GPU experiments with full-round timing and causal-censored logs.
