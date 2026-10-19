# Expert coactivation: a route-footprint screen for fixed GLM

Read 2026-10-19: [Nishu et al., arXiv:2609.22471v1](https://arxiv.org/html/2609.22471v1), submitted 2026-09-18; §§2–5, §6 assumptions, Appendix B. Source review only; no model import, GPU, service action, runtime patch or new harness.

## Paper mechanism and limits

The paper targets bandwidth-bound verification with every expert resident. A wider speculative window can touch more distinct experts and increase weight traffic. Its intervention changes training: global rather than microbatch load balancing; shared experts; adjacent-token KL with the earlier router output stop-gradient; and autoregressive expert selection. For each next token, retain its highest-scoring `k-l` experts from the previous selection, then choose `l` others. Distinct experts per `gamma+1` window are bounded by `k+gamma*l`. Experiments use `l=1`. [§§3–4, §6](https://arxiv.org/html/2609.22471v1#S4).

The authors report 21% greater throughput at comparable task accuracy for their trained 16-expert, top-4 MoE: 1,032M total / 353M active parameters. Models train from scratch on 6T tokens; the draft is 3% of target size. Appendix B names H100 training hardware; the reviewed text does not identify the inference device. Comparable accuracy is not identical target outputs. Their result is not a GLM/M5 measurement, and assumes memory residency already works. [§§5.1–5.5 and Appendix B](https://arxiv.org/html/2609.22471v1#S5).

## Our applicability and evidence

Changing GLM's expert choices to enforce the paper's retention rule would change the fixed target computation. It is outside our exact engine-optimization contract. We can instead measure natural overlap and retain the original choices, quantized weights, activation, routing weights and reduction order.

Actual derived `config.json:233–256` specifies 288 routed experts, top-8, one shared expert, sigmoid scores, normalized selected weights and scale 2.5. Installed `mlx_vlm/models/glm5_next/language.py:48–112` implements selection with correction bias, then uses the original sigmoid scores for weighted combination. `mlx_vlm/models/switch_layers.py:175–252` already has block verification fast paths, a tokenwise fallback and sorting in the general path. We cannot infer that expert reuse is missing merely from an expert-union count.

The latest completed full loader diagnostic remains [E039 FAIL / RESTORED](../../docs/experiments/m5-ultra-256gb/E039-stacked-glm-load-v1/README.md): zero completed requests. [E041](../../docs/experiments/m5-ultra-256gb/E041-glm-filelike-io-smoke-v1/README.md) qualifies only tiny CPU file-like IO bytes/lifetime. Neither provides a GLM route trace, accepted-token rate or throughput result. The memory-residency gate is therefore still first.

## One falsifiable next experiment — natural route-footprint screen

**Proposed, not executed.** After the unchanged loader/resource/recovery gates pass, collect the existing four diagnostic requests' ordinary greedy routes in a separate, untimed, opt-in trace. At the existing `MoE.__call__` gate return (`switch_layers.py:265`), retain each position's selected IDs and routing-weight bytes without changing them; transfer the small trace to CPU after evaluation. Confirm observer-on/off original outputs and caches match before interpreting it. No router retraining, forced overlap, speculative acceptance approximation or offload.

CPU analysis uses consecutive windows `W=1,2,4,8`, separately within each request and layer. Obtain each expert's packed weight + scale + bias byte count `b(layer,expert)` from E036 metadata, respecting mixed quantization. Compute:

- `B_repeat(W)`: sum of selected expert bytes separately for each token;
- `B_union(W)`: sum once for each distinct `(layer, expert)` in that window;
- `H(W) = 1 - B_union(W)/B_repeat(W)`: ideal repeated-byte headroom, with the shared expert reported separately.

Predeclare the screening hypothesis: at W4 or W8, median `H >= 20%` in **each** request. Retain every window, layers and failures; requests with insufficient tokens make the screen incomplete. If the threshold fails, deprioritize a new cross-token routed-expert reuse kernel. Passing only motivates an exact kernel qualification: `B_union` is not measured DRAM traffic, and existing kernels/cache may already capture the headroom. Preserve each token's expert-slot order and all rounding/reduction points in any later implementation.

This trace concerns accepted ordinary tokens; rejected speculative candidates may activate different experts. It cannot choose an optimal draft window or predict SD speed. That would subsequently require actual verifier routes, draft/verify time and accepted tokens per cycle under the same exactness checks. While GLM load remains blocked, this note and existing byte metadata are the preparation; inventing a synthetic routing trace would not answer the hypothesis.

## Local source identities

Read-only SHA256 of the installed files, 2026-10-19:

- `mlx_vlm/models/glm5_next/language.py`: `1567e5dae06cd77032995af45a6c0bf5dea66679d0a3da7eba0ab5f7454c0ce4`
- `mlx_vlm/models/switch_layers.py`: `ace00152d1d004e52292d8265af640d8dd69ffa67b8906886b53a99eb21951de`

Both are under `.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/`.
The existing [BaseRT/NAX review](m5-nax-prefill-opportunities.md) already covers the separate matrix-dispatch direction; it was not re-read as new primary research here.
