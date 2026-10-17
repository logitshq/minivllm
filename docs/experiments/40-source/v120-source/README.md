# V120 short C1 prework — actual117 bound, GPU UNRUN

One thin factory and loopback SSE adapter use frozen candidate116, with no target/draft math changes. A is the original ordinary ModelRunner with its existing ahead path. B explicitly loads DFlash2 with `bounded_load=True`, installs `ModelRunner(...,spec_block=5)`, and enables qualified transaction+broadcastM1. Neither arm uses the numerical observer/reference/tensor-hash callbacks. Candidate-intrinsic transaction bookkeeping stays intact.

The actual V117 receipt and external release are bound in `qualification-binding.json`; the original HOLD example is retained separately. Root-only inference checks that binding and qualified candidate bytes before importing MLX. Source pins and 7 stdlib-only behavior/AST checks pass. `client.py` request/group functions have exact V105 AST. No author model/GPU/HTTP/server run occurred.

Start with root's two-fresh ordinary→speculative pilot, each3warm+3measure, one fixed20s unloaded/reaped gap:12requests/384tokens total,6measured/192tokens. Keep negative results. If promising, do a separate3-pair balanced confirmation; do not pool the selected pilot into it. No fabricated CI/significance. Root owns all PID/port cleanup and resource guards; supplied root owner was source-read for first-run blockers, not executed by this author.

Primary axes are full client POST→SSE `[DONE]` p50 latency and actual total-output throughput. Request clocks include queue, prefill, draft, verify, commit and stream delivery; they exclude server loading, warmups, resource samples, cooling and post-DONE stream teardown. TTFT is separate. Existing first-text interactivity is supplementary because speculative bursts alter first-chunk size. Qualification timings never enter these metrics.

Root CLI (all authors idle before performance):

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v23-root-control-build1/v120_paired_c1_owner_build1.py --out ABSOLUTE_NEW_OUTPUT
```

The owner launches `server.py --execute-gpu --mode ordinary|speculative --qualification-binding BOUND_JSON --port OWNED_PORT --owner-pid DIRECT_PARENT_PID --run-dir NEW_SERVER_DIR` with50/50/20/0 and no DYLD injection, waits for `/v1/models`, then calls `client.measure_arm`. Client standalone CLI:

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v120-spec-throughput-latency-prework-build1/client.py --execute-http --mode ordinary --qualification-binding /private/tmp/minivllm-engine-optimization-20261014/v120-spec-throughput-latency-prework-build1/qualification-binding.json --base-url http://127.0.0.1:OWNED_PORT --server-pid OWNED_PID --out ABSOLUTE_NEW_CLIENT_OUTPUT --power-domain 'AC Power'
```

This first pilot is AC/thermal0/pressure1 with original paging guards. Stop on any guard/output/source/owned-cleanup failure; retain partial rows, no silent retry or pooling. Server/client do not implement global process cleanup; root owner does exact owned cleanup and port-absence proof. Ordinary/speculative local raw32 must match V117's native fixture. External official APIs may only expose usage/visible text and have different numeric-profile continuations; never label those raw-ID equality.

`historical-reference-points.json` preserves dated native15/official-none compatible configuration medians and official none-vs-MTP results, without copying argv or credentials. They locate the next useful test; they are not contemporaneous paired evidence. A new C1/Pareto claim requires fresh compatible controls. Slower MTP does not replace the faster none baseline.

512/4K→128 are **not implemented or qualified here**. VerifyTransaction retains a fixed existing physical KV allocation; raising max_len alone fails at the first boundary. A future probe must reserve at least640/4224 slots before any token, hold the same target/draft math, pin actual tokenized fixtures, and pass new own-prefill/same-frontier128-cache+lastlogits strict checks, tailzero/readonly/reset/EOS/budget gates before performance. NormalEOS means128 is a cap until actual delivery proves otherwise. No long-context or full-frontier result follows from117.
