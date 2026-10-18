# GDN prework toward C8: reuse the existing B2–8 kernels first

2026-10-17. Source-only design; no imports, tests, device work or candidate
implementation. **No throughput or latency improvement has been measured for
this extension.** HTTP C8 does not imply that every model call has eight rows.

The smallest next hypothesis is a separate default-OFF long-decode adapter
that keeps all five original native `Linear` calls and admits B2–8 to the
already-existing batched convolution/prework/state kernels. Preserve the
frozen B1 candidate and its evidence. Do not start by packing projections,
looping over independent M1 projections, widening gates compilation, or
combining E024/E028.

## Concrete source boundary

| Source | Finding and implication |
| --- | --- |
| [B1 adapter:65–105](../m5_ultra_256gb/gdn_prework_unbound/candidate.py#L65) | Eligibility hardcodes B1 in x/conv/S; two reshapes also hardcode 1. The five projections remain ordinary native calls and parameters are read on every invocation. A new adapter needs batch-shaped checks and reshapes, not new packed weights. |
| [Existing fused kernel:35–62](../../tutorial_code/miniengine/kernels/gdn_decode_fused.py#L35) | `fused` already accepts B1–8. B>1 uses prework grid `(8192,B,1)`, batch pointer offsets, and a separate state grid `(128,192,B)`. B1 retains its original two pipelines. No used-length/position argument enters this operator. |
| [Original decode:208–227](../../tutorial_code/miniengine/model/gdn.py#L208) | The ordinary reference creates `[B,4,10240]` convolution input, reduces it, activates/splits/repeats heads, then runs the original state kernel. It replaces conv/S before norm/out projection. The candidate must preserve that commit order and never retry the old path after state advancement. |
| [Gates:87–112](../../tutorial_code/miniengine/model/gdn.py#L87) | `_decode_gates` is compiled only for B1. For B2–8 it returns the original `_gates`; retain that fallback even with gates configuration ON. Broadening the prework guard must not accidentally broaden gate compilation. |
| [Original context gate:76–85](../../tutorial_code/miniengine/model/qwen3_5.py#L76) and [GDN dispatch:63–73](../../tutorial_code/miniengine/model/gdn.py#L63) | `_norm_reuse_normal` is proof of ordinary decode context, not permission to turn norm-reuse configuration ON. Keep exact context/cache types, norm OFF, target-only operation and unsupported-context fallback. |
| [Short graph:22–40](../../tutorial_code/miniengine/engine/compiled_decode.py#L22) | Its B1–8 path is bounded to used≤256. Existing short fusion does not establish long D512 state coverage. Long GDN state has fixed shape, but its values depend on the full history. |

The first-principles saving is removal of intermediate convolution/activation/
head-repeat work and repeated Q/K norm work. The new prework calculates norms
once per key head and shares them among three value heads; it does not reduce
the five projections' weight traffic or the recurrent S read/write volume.
At larger B the remaining projection/state traffic may dominate, so less
source-level work is not a predicted end-to-end gain.

## Arithmetic and state risks that still need device evidence

The [prepared convolution:9–22](../../tutorial_code/miniengine/kernels/gdn_decode_fused.py#L9)
explicitly rounds BF16 products and each addition. It is intended to match the
native sum, not mathematically reassociate it. The batched native reduction
and kernel compiler still require exact comparison on real activations.
The [activation and norm prework:10–78](../../tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py#L10)
retains typed BF16 sigmoid/multiply and serial d=0..127 norms. The
[state body:80–120](../../tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py#L80)
keeps SPLIT=4/VALUE_TILE=32 and segment-reduction order from the
[original kernel:15–59](../../tutorial_code/miniengine/kernels/gdn_metal.py#L15).
Batch offsets `(b*16+h/3)` for shared Q/K/scales versus `(b*48+h)` for V/S,
BF16 boundaries, compile options, and noncontiguous source handling remain
qualification risks; source similarity is not an arithmetic proof.

[DecodeBatch.rebuild:116–147](../../tutorial_code/miniengine/cache.py#L116)
reorders, removes and appends conv/S rows from different histories. Test that
state association explicitly. Each candidate chain must consume its own prior
S/conv; copying reference state back after every step would hide divergence.
No position bound is needed by this kernel's addressing, but no untested
history range, speculative state, grouped prefill or B>8 is thereby qualified.

## Minimum staged validation

1. First-stop real-activation canary: original native **batched** forward at
   B2 and B8, early and late natural D512 history, GDN ordinals 0/24/47,
   three consecutive steps. Retain actual row IDs/lengths, full reference
   logits/greedy causality, original x and independent initial states. Compare
   complete layer output BF16, conv BF16 and S FP32, finite and unsigned bytes;
   save both sides of the first failure. A local chain still uses reference x
   and is not candidate full-model qualification. Early EOS means incomplete.
2. Before claiming the whole B1–8 domain, add B3–7, differing row histories,
   row permutation/removal/refill, B1→B8→B1, strided inputs, and default-OFF/
   B9/context fallback. Preserve original inputs/parameters and verify changed
   parameter values/restoration read current arrays; no derived parameter cache.
   Pin the real batched prework/state sources and original state-kernel source.
   The old [synthetic kernel check:19–44](../m5_ultra_256gb/decode_batch/qualify_kernel.py#L19)
   converts arrays to F32 before hashing, so do not reuse that as the new
   original-dtype evidence format.
3. Then reuse the unchanged 37-request full-logit/all-cache/ahead comparison
   and actual C1/C8 capture→reap→independent replay. New dispatch expectations
   must reflect B2–8 candidate calls instead of the frozen B>1 fallback rule;
   do not relabel old receipts. Only subsequent observer-OFF, fresh-process
   comparisons can establish throughput versus full POST→DONE p50/p95.

The [E004 record:3–20](../m5_ultra_256gb/decode_batch/README.md#L3) retains
short B1–8 success **and** larger-batch packing failure: B16 arithmetic failed,
and the mixer-preserving variant regressed at B32. The
[packing review:16–35](long-d-gdn-projection-packing.md#L16) also preserves
M4 GDN4→1's +0.147238%/4-of-6 performance STOP and the rejected confounded
claim. [E027:9–29](../../docs/experiments/m5-ultra-256gb/E027-native-bridge-v1/README.md#L9)
found five BF16 differences for native multirow versus per-row M1 at T2;
forced M1 is not a valid replacement oracle. These findings motivate keeping
native batched projections intact; they neither prove nor disprove a long-B8
prework gain. Prefer this small domain extension before speculative packing.
