# Ordered replay: no-go for a Metal prototype as the next optimization

**Decision:** retain the exact CPU reference, but do not prioritize a Metal
implementation now. It proves a storage representation can preserve the toy's
rounding sequence; it does not remove the next token's need for the full logical
state. The current code repeats substantial arithmetic. No profile currently
establishes that saved state writes would repay it.

## Dependency and operation budget

Let P=K·V and L be the pending log length, cycling 0…M−1 without resets.
`toy.py:123–149` must first reconstruct every old state element for `kv`, finish
that reduction, compute delta, then reconstruct/update state for the output.
The merge pass reconstructs it a third time. All K·V elements of the dense base
are read on **every** update; this representation does not skip the dense read.
Each `cell` replay entry performs three separately rounded multiplies and one
add (`toy.py:107–116`). Averaged over a complete merge cycle, excluding diagnostic
state snapshots and lifecycle operations:

| M | Literal toy base-word reads / P | Literal replayed updates / cell | Literal FP32 scalar ops / P¹ | Optimistic fused ops / P² |
|---|---:|---:|---:|---:|
| 1 | 3 | 2 | 14 | 7 |
| 2 | 2.5 | 3 | 18 | 8.5 |
| 4 | 2.25 | 5 | 26 | 11.5 |
| 8 | 2.125 | 9 | 42 | 17.5 |

¹ Literal dense oracle: 10P+3V operations; deferred: (4M+10)P+3V. These count
scalar arithmetic/rounding nodes, **not instructions, cycles or Python costs**.

² A hypothetical better kernel reads/replays old state once, retains it across
the `kv → delta` dependency, then updates, accumulates output, and writes the
base only on a merge. With exact rounded scaled-key/query common subexpressions,
dense costs 7P+2K+3V; deferred costs [7+1.5(M−1)]P+2K+3V, plus scale work for
old entries unless already cached. Thus M=8 still adds roughly **2.5× arithmetic**
relative to the similarly optimized dense recurrence. This is a lower-cost
construction to assess, not implemented code or a native equivalence proof.

`gdn_norm_reuse_metal.py:90–120` has K=128 and SPLIT=4. Retaining an old-state tile
would require at least 32 FP32 state values per thread across the delta reduction,
before other live values. Register pressure/spills may defeat the intended
traffic saving. The existing native kernel reads S in both `kv` and update loops;
its output uses the freshly computed scalar. Eliminating a second native read
must also be evaluated in a dense kernel, rather than credited only to logging.

## Traffic ceiling, not a speed prediction

The native B1 state is [48,128,128] FP32: 3 MiB per GDN layer. The 48 GDN layers
therefore contain 144 MiB of recurrent state. In the direct per-value-head toy layout (without sharing key factors across
heads), a steady M=8 log has at most seven entries of 2+128+128=258 words
per value head. The analytical persistent writes
per head/token are P/M+(1−1/M)·258 = **2,273.75 words**, versus 16,384 dense words:
86.12% fewer state/log writes, about **124.0 MiB/token** saved across 48 layers.
No hardware traffic was measured; caches and reuse can change external traffic.

Even assuming one base read in both optimized kernels and ignoring all log reads,
state read+write volume improves from 2P to P+2,273.75 words: only **1.756×** at
M=8, not an 8× total-traffic reduction. Reading each distinct pending log factor
once adds an average 903 words/head/token and lowers that volume ratio to about
**1.675×**. Repeated loads, spills, metadata, output and synchronization worsen it.
Full-model projections, attention, normalization, scheduling and HTTP stay outside
this saving. Under the optimistic 1.756× *state-component speed* assumption,
that component must already consume at least **21.1%** of elapsed decode time
for a 1.10× total speedup; this assumption itself is not established.

## Exactness constraint and next gate

Neither `kv` nor output may be replaced by algebraically distributed dot products
through the log: FP32 is not distributive. Decays and deltas cannot be combined.
Each reconstructed element retains the per-step multiply/add dependence; the
CPU counterexample demonstrates why. Native Metal additionally requires its
actual exp, FMA/contraction, split-reduction and normalization contract. The toy's
separate scalar rounding does not establish that contract.

A small Metal experiment becomes justified only after an existing-path profile
shows a material state-write bottleneck and a register/traffic budget predicts
an advantage over an equivalently tiled dense kernel. It would then need native
original-dtype state/output exact gates before performance measurement. Until
then: **no-go for implementation priority; keep as a bounded hypothesis**.

Source basis: `toy.py` SHA `15f4bc23c0d413da8adeab857d8ed4c757c8c877aedcf728f402ffc241fd60c1`;
native norm-reuse source SHA `b42aaceb140c53d87b9d5c5ebc6392a1e1ca0bcf7b55266afc6f68b22e0defd1`;
48-layer scope checked in `gdn_prework_unbound/serving.py:167`. No device operation,
performance test, frozen-source edit or additional CPU test run produced this note.
