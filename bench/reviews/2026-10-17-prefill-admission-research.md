# Bounded new-prefill admission: a hypothesis after the HTTP baseline

2026-10-17. Primary-source reading and local source inspection only. No model,
service, runtime hash, production edit or performance experiment was performed.
Checked `docs/optimization-guide/papers.md` and the existing Apple/GDN research
notes first; the two papers below were not already recorded there.

## New primary sources

[Stable Token-level Scheduling for Serverless LLM Inference, v1](https://arxiv.org/html/2610.17689v1),
October 13, 2026; read §§III–VI. TWP combines feedback-driven admission with
chunked prefill to limit interference with active decoding. Its reported setup
is V100 32GB, Llama-3.2-3B FP16, and heterogeneous prompts. The token-load policy
depends on remaining/expected output lengths; normal EOS does not reveal those
lengths in advance. Its chunk and controller constants are not our parameters.
The evaluation names token throughput among its metrics but reports requests/s
in its throughput discussion. We import neither its speed ratios nor a claim
that its KV-centric cost model describes our GDN-heavy model.

[Online Batching with Lookahead: Adversarial Limits and Stochastic Gains, v1](https://arxiv.org/html/2610.16013v1),
October 12, 2026; read the introduction's scope, §2 model, and §§4–5. The analysis
assumes fixed batch service time and bounded capacity. Its scope explicitly
excludes autoregressive generation, variable output lengths and sequence-dependent
cache constraints. Prediction quality and arrival assumptions matter. This does
not justify a predictive waiting timer for our closed-loop D workload or establish
an Apple/GDN speedup. We do not transplant its bounds into our Pareto claims.

## Our narrower inference, not a paper reproduction

Candidate: **`max_new_prefills_per_decode_step=1`, default OFF**. At a scheduler
step that already contains decode work, admit at most one new waiting request
in FIFO order. Keep active-row order, existing prefill continuation, the original
total token budget, sampling, EOS/cap handling and model expressions unchanged.
When there is no decode work, retain the original admission rule. Do not predict
remaining output length, reorder waiting jobs or add a sleep/polling loop.

The frozen scheduler consumes the decode budget first and then admits waiting
requests until the remaining token budget is exhausted. Our D prompts contain
886–978 tokens; a 2048-token step can admit two full prompts alongside decoding.
The runner constructs both prefill graphs before its common evaluation boundary.
Restricting new admissions could reduce simultaneously live prefill intermediates
and the interruption experienced by existing decoders. This is a source-derived
hypothesis, not an explanation proven for E023's earlier resource abort. See
[the local memory inventory](e023-cache-memory-budget.md).

There is a real tradeoff: the waiting request starts later, and smaller batches
may lose throughput. C1 should retain its original route because the next request
normally arrives after the preceding request completes; verify that rather than
assuming it. ITL is diagnostic, not the adoption objective. No speedup is projected.

## Necessary qualification and measurement order

1. Finish the current native **actual HTTP baseline** first. This proposal does
   not change any running or frozen E028/E023 plan.
2. CPU-check OFF against the original schedule, and ON for FIFO order, the
   admission limit, eventual progress and exact EOS/cap retirement. Test partial
   prefill continuation and simultaneous retirement/refill explicitly.
3. Register a new numerical schedule domain. Both real engines independently
   execute the pinned ON policy: eager reference with gates/grouping OFF versus
   the current compiled/fused candidate. Retain the original 37-request workload,
   full-vocabulary logits and all 64-layer state comparisons, including staging,
   ahead, retirement, finite values and original-dtype bit equality. No recorded
   token or fabricated scheduler output may drive an engine.
4. Record the policy's own C1/C8 HTTP arrivals and raw SSE, then perform the same
   full-state replay. The frozen `http_join.py` assumes the original admission
   rule, so a separately pinned narrow policy-aware checker is required; the
   original qualification receipt cannot cover the new policy implicitly.
5. Only after those gates, measure three fresh paired OFF/ON processes under the
   original pressure-1, model, D-task, normal-EOS, output-cap and allocator
   contracts. Report actual committed tokens per group window against complete
   POST→DONE p50 latency, with p95, output lengths and every process retained.
   Keep this policy separate from E026/E028 when attributing effects.

Keeping weights and scalar expressions unchanged is insufficient for numerical
equivalence: different batch histories can select different projection/reduction
routes and alter rounding, subsequent greedy choices and recurrent state. The
same-schedule eager comparison is therefore mandatory. It does not by itself
prove OFF and ON produce identical token streams; any such additional claim
requires a separate per-request full raw-ID equality check. No quality average
or performance gain may substitute for a failed declared correctness gate.
