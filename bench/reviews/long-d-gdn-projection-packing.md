# Long D512: separate GDN projection packing from the short decode graph

2026-10-17. Source and retained small-receipt review only; no tests, runtime/model import, GPU, service action or implementation. The current unbound-prework canary must finish before adding another candidate.

**B1 output-row packing remains a testable hypothesis; existing failures do not rule it out. They do rule out assuming arbitrary-batch exactness or projecting short-path speedups onto long D512.** The simplest next candidate would combine only the four GDN input projections, with its own default-OFF per-instance switch.

## Actual native boundary

- [`compiled_decode.py:22–38`](../../tutorial_code/miniengine/engine/compiled_decode.py) checks used length≤256 before both graph reuse and packed/fused execution. The current batch domain is1–8. D prompts of886–978 tokens never enter it.
- [`packed_decode.py:14–41`](../../tutorial_code/miniengine/engine/packed_decode.py) does not inspect sequence position. It concatenates output rows and splits them back; K remains5120. Thus256 is a joint route qualification boundary, not a matrix-packing mathematical requirement.
- The existing class is **not** a neutral packing-only adapter: `PackedProjection.__call__` calls custom `qmv`; `.linear` changes output projections to that helper too. `PackedDecode` also packs every MLP and attention group. [`gdn.py:_decode_fused`](../../tutorial_code/miniengine/model/gdn.py:253) additionally enables fused GDN preparation and redirects `out_proj`. Passing `_decode_projections` across the whole model can redirect the vocabulary head as well. Reusing that switch wholesale would conflate several changes.
- [`layers.py:80`](../../tutorial_code/miniengine/model/layers.py:80) ordinary `Linear` calls native `mx.quantized_matmul`. [`nvfp4_qmv.py`](../../tutorial_code/miniengine/kernels/nvfp4_qmv.py) uses custom R1 for supported B1, with FP contraction disabled. That source warns that contraction changed a vocabulary-head BF16 result. Keep this separate from packing-only qualification.

## What previous experiments actually establish

| Evidence | Observed result | Relevance to long B1 packing |
| --- | --- | --- |
| [M4 v20/v21](../../docs/experiments/18-projection-packing-evaluation.md) | GDN4→1 had120 byte records PASS, but +0.147238% measured improvement and4/6 winning pairs: performance STOP. MLP2→1 was −0.122966%, also STOP. Old AC +21.9% was order-confounded and remains rejected. | No exactness veto for B1; no demonstrated large benefit on M4. Different hardware and synthetic timing inputs do not settle M5 D512. |
| [M5 short fusion](../../docs/experiments/m5-ultra-256gb/decode-fusion-v1/m5-ultra-fusion-report.txt) |214 complete forward pairs plus kernel tests passed; bundled projection/GDN changes improved the registered short19→32 C1 throughput8.14%. | Combined short-path result, not packing-only or long-state evidence. |
| [M5 larger-batch exploration](../m5_ultra_256gb/decode_batch/README.md) | Original retained `pareto-batch-probe-v1/full.json` ends B16/step0 with `exact:false`; its child log lists logits and nearly all state signatures different. A changed candidate preserved mixer projections, completed12 pairs, but B32 regressed and was not adopted. | Do not widen to arbitrary B. Current retained source `nax-prefill-source-review/quantized.cpp:89–118,1846–1897` selects QMM at M≥12 on g17d/K5120; packed N then can alter reduction geometry. This is a batch-route issue, not evidence that used>256 alone changes a B1 projection. |
| [E027 native bridge](../../docs/experiments/m5-ultra-256gb/E027-native-bridge-v1/README.md) | Case19, synthetic N10240/T2: five of20,480 BF16 elements differ between native multirow and independent M1; first−56.75 versus−57.0. Shared candidate was not reached. | Forbids treating independent M1 as a drop-in oracle for B>1. It did not test B1 output-row concatenation. |

For g17d B1/K5120, the retained native dispatcher takes `qmv_fast` for original N10240/6144/48/48 and packed N16480: each N is divisible by8. The M5-Max-only R2 condition ends in architecture `s`, whereas this device is `d`. This source comparison removes one known dispatch change; it is not a device bit-equality proof for all long activations.

## Smallest next experiment, after the prework canary

Use one temporary `_decode` adapter per real GDN layer. Derive just `[qkv|z|b|a]` U32 weights and U8 scales from current original parameters; call **native** `mx.quantized_matmul` once and split at10240/16384/16432. Keep original ordinary convolution, gates, recurrent kernel, norm and `out_proj` unchanged. If evaluating packing on top of an independently qualified prework candidate, make that a separately named paired comparison with prework enabled in both arms. Do not silently enable packing via `_decode_fused`.

Begin with B1, target-only ordinary decode, gatesON/normOFF. Unsupported batch/context/capture/spec routes return to original code. Parameter handle replacement must invalidate the derived group; changed-value weight and scale replacement/restoration must be checked. Do not rebind original model parameters to slab views as the older M4 experiment did. Do not cache activation outputs across the four original projection calls.

The structural saving is four matrix operations→one per GDN layer: at48 layers,144 fewer such operations per B1 forward. This is not144 guaranteed command-buffer reductions. Total weight bytes read remain unchanged, and split/dispatch/guard costs remain. Keeping original parameters adds47,462,400 derived bytes per GDN group,2,278,195,200 bytes across48 groups (about2.122GiB), excluding temporary packing/allocator costs. The previous all-model pack was8.714GiB. Neither number is measured physical residency or a throughput prediction.

Reuse the existing real-activation canary's two D fixtures, early/middle/late GDN layers and three-step chains. Compare all four projected outputs before state use, then complete layer output/BF16 conv/FP32 S against ordinary native decode, with candidate state continued from its own prior result. Require exact dtype/shape/finite/raw bits and actual dispatch evidence. Keep normal EOS; missing late coverage stays incomplete. Add changed-value weight/scales plus restoration, and B2 fallback checks. No random-only or forced-M1 oracle.

Only after these checks should the existing37-request same-schedule harness and actual C1/C8 HTTP capture/replay be reused. Any subsequent timing must include the complete forward or HTTP request, both arms' guards and state handling, fresh derived-pack cost separately, and all negative pairs. A local matrix speedup cannot establish a new Pareto point. Prework and ownership changes remain the higher-priority independent candidates until this packing hypothesis shows a material complete-forward effect.
