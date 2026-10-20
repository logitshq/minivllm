# C1 shared backbone projections: the next bounded experiment

2026-10-19, source/metadata review only. No backend import, device execution,
model payload read, service action or implementation. E054 timing is still an
independent decision gate; this note assumes no verifier speedup.

**Start with the 64 MLP gate/up pairs.** They already match the frozen shared
K5120 kernel and account for **44.5263%** of logical projection weight/scale
bytes. Keep each mixer and every row norm, SiLU, multiply, down projection and
residual addition in their original order and shapes. Gather only independent
normalized rows for the two gate/up projections. This changes a substantial
weight-reading opportunity with one optional verifier seam, without writing a
new engine, changing the recurrence, or generalizing the kernel.

## Actual model and compatible shapes

The reviewed execution package is the E042/E054 frozen native tree, **not merely
whatever happens to be current in `tutorial_code/`**:

`F=/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine`.

The actual Qwen3.8-27B snapshot is
`/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`.
Its `config.json` SHA is
`7380cfc6eb77891381ee24e944e6bd0caed7a957707f2512ddd6effc82c7e706`:
64 layers, 48 GDN/16 attention, hidden5120, MLP17408, head dimension256,
24 attention heads/4 KV heads, GDN16 key heads/48 value heads with dimension128,
untied vocabulary248320, NVFP4 group16/4bits.

`nvfp4_multirow_shared/candidate.py::supported` (62–75) requires BF16
`X[T,5120]`, U32 `W[N,640]`, U8 `scales[N,320]`, T2/4/8, N divisible by4,
N*K<2^32, and the fixed MLX0.32.3/M5 Ultra device. All eligible N below satisfy
those bounds. Admission is a shape fact, **not numerical qualification** of these
new live activations or a speed measurement.

| Projections | Count | Logical N×K; stored W / scales | Current kernel |
|---|---:|---|---|
| `mlp.gate_proj`, `mlp.up_proj` | 64 each | 17408×5120; `[17408,640]` / `[17408,320]` | Eligible; first target |
| `linear_attn.in_proj_qkv` | 48 | 10240×5120; `[10240,640]` / `[10240,320]` | Eligible |
| `linear_attn.in_proj_z` | 48 | 6144×5120; `[6144,640]` / `[6144,320]` | Eligible |
| `linear_attn.in_proj_b`, `in_proj_a` | 48 each | 48×5120; `[48,640]` / `[48,320]` | Eligible, very small N; launch/barrier cost may dominate |
| `self_attn.q_proj` (query+gate) | 16 | 12288×5120; `[12288,640]` / `[12288,320]` | Eligible |
| `self_attn.k_proj`, `v_proj` | 16 each | 1024×5120; `[1024,640]` / `[1024,320]` | Eligible |
| `linear_attn.out_proj`, `self_attn.o_proj` | 48 + 16 | 5120×6144; `[5120,768]` / `[5120,384]` | **Unsupported K6144** |
| `mlp.down_proj` | 64 | 5120×17408; `[5120,2176]` / `[5120,1088]` | **Unsupported K17408** |
| `lm_head` | 1 | 248320×5120; `[248320,640]` / `[248320,320]` | Existing E042 head path |

Shapes and byte counts were cross-checked against the already retained E042
`rowwise-head-shared-v1-child/parameters-before.json`, SHA
`55065230f58359d08e6e6830a0c62dede9c66be0aa8365868231befc31bc51da`.
That is a producer parameter metadata receipt; no weight payload was reread.
`F/model/layers.py::Linear.__call__` (79–86) passes original packed weight/scales
to `mx.quantized_matmul` with no bias or extra postscale in this model path.

## The minimal seam: share MLP inputs, keep sequential mixers

The existing `rowwise_native_verify/candidate.py::verify` (78–88) loops layers,
then T rows, calling the complete `DecoderLayer` for each row. At a layer's entry,
all T input hidden rows already exist. A row's MLP does not update that layer's
KV/GDN state, so postponing its MLP until the other rows' mixer results exist
introduces no mathematical cross-token dependency. This is the relevant change
in scheduling; native AR still cannot know future input tokens without a drafter.

Prepare a separate optional candidate arm, such as `mlp_input_project=None`,
leaving the frozen E042/E054 sources untouched. Its per-layer implementation
should be the following small decomposition of
`F/model/qwen3_5.py::DecoderLayer.__call__` (43–49):

1. For t=0..T−1, construct the **same** prefix-owned cache wrapper and absolute
   context as current `verify` (81–84). Call the original
   `input_layernorm(x[t:t+1])`. Call the unchanged mixer at B1, passing the
   original `_norm_reuse_normal=True` for GDN, **without `_projection`**.
   Preserve `mixed[t] = x[t:t+1] + mixer_output` as a separate BF16 expression.
2. Call `post_attention_layernorm(mixed[t])` separately for each row. Concatenate
   only those original M1 results into BF16 `[T,5120]`.
3. Call the unchanged shared projector twice, once with the live gate weight and
   scales, once with live up weight and scales. Do not concatenate/pack weights
   or replace Linear modules. Both return `[T,17408]`.
4. For each row, perform exactly the original `nn.silu(gate[t:t+1]) *
   up[t:t+1]`, native `mlp.down_proj` at `[1,17408]`, and `mixed[t] + down`.
   Concatenate only completed rows for the next decoder layer.

Keep the original one-row embedding, final row norms, existing shared head,
selected-prefix ownership/commit and continuation unchanged. The new switch is
OFF by default, rejects unsupported parameters before any state update and
fails if an explicitly enabled projector returns None; there is no retry through
a mutated cache. All 64 layers with both projections enabled imply **128 shared
MLP projector invocations per verifier call**, separately from the one head call.
Guard actual no-bias NVFP4 metadata/live arrays, not just dimensions.

The two T8 projection outputs total557,056 bytes; one `[8,5120]` BF16 matrix is
81,920 bytes. These are logical local intermediates, not a peak-memory estimate.
Lazy graphs, original KV prefix copies and allocator lifetime may retain more.
Keep actual resource sampling and measure staging/concat inside timed work.
There is no reason to duplicate multiple GiB of packed weights for this change.

## Attention and GDN are separate follow-ups

**Attention:** `F/model/attention.py::_project` (30–37) already has a narrow
`_projection(x)` seam. Collect the original per-row input norms at layer entry,
share q/k/v projections across T, then supply each row's slices through a tiny
row projection facade. Its `.linear(op,x)` must call the **original `op(x)`** for
K6144 output projection. Keep B1 q/k RMSNorm, RoPE offsets, `kv.write`, the
row-specific max_len/lens mask, SDPA, sigmoid gate and output projection in the
original `_decode` (79–93). Do not substitute a T-query causal attention kernel:
that is another reduction/masking change, not projection-only reuse.

**GDN:** qkv/z/b/a depend only on the known layer-entry normalized row, so their
projections are independent; conv tail and S remain dependent across t. However,
`GatedDeltaNet.__call__` (63–73) sends *any* non-None `_projection` directly into
`_decode_fused` (252–260). That path changes conv/recurrent arithmetic through the
fused kernel. **It is not a projection-only hook into the E054 native oracle.**
A later isolated helper must accept precomputed qkv/z/b/a slices while retaining
the exact `_decode` body (208–227): row B1 shapes, original four-tap sum order,
SiLU/head splitting, `_decode_gates` compile policy, recurrent Metal call,
conv/S replacement, gated norm and native K6144 output projection. Preserve
`BatchState` and actual `Linear` module identities; wrapping b/a modules can
silently defeat `_decode_gates` type admission. Leave the tiny b/a projections
native initially if that makes the isolated change smaller. Do not mix this
work with the separately qualified batched-GDN graph experiment.

**Non-K5120 projections:** the shared kernel hardcodes both K and its tile
sizes/strides. Merely broadening `supported` would read incorrect addresses.
K6144/K17408 need distinct proven layouts and new qualification; the current
plan leaves them as native M1 operations. Do not use native M=T QMM as an
unqualified substitute for the original per-row arithmetic.

## Smallest next device canary and decision

After root owns an idle device and has E054's real cost result, reuse the existing
load/pause/resource/recovery owner and the original used1023 seed; no new
lifecycle framework. Existing E042/E054 saved tensors contain states/logits,
**not these intermediate MLP activations**. A real-activation canary must capture
the actual normalized MLP inputs during original teacher-forced execution; it
cannot relabel E040's deterministic synthetic X as a live activation.

First compare gate and up at layer0, layer32 and layer63, T4/T8: six layer/width cases,
twelve projection comparisons. For each, evaluate original native M1 calls per
row and the shared kernel from the same captured BF16 rows; save and compare
all original BF16 bytes plus finite flags and actual dispatch. Then compare the
complete MLP outputs using original rowwise SiLU/multiply/down/residual. The
capture pass can continue on native outputs, so failed comparison cannot poison
the remainder. Include concat/staging/eval/sync in paired local timing; exclude
model loading, capture and artifact writing. This small result qualifies only
these captured points.

Only if those pass, run the optional arm for all64 MLPs with the existing
T4/T8 full-logits + selected k=1,T/2,T full-state + native-continuation comparison.
Expect 128 shared MLP calls per verify, no silent fallback, unchanged weights and
canonical seeds. This is the necessary full-model numerical gate before
repeating E054's full-verifier+commit versus causal native AR(T) cost comparison.
The original default arm is a useful controlled comparison; performance
qualification additionally requires the new arm to beat native AR after draft
and acceptance overhead, not merely beat head-only verification.

A local MLP win can be canceled by repeated down-projection reads, changed lazy
scheduling, launch/staging overhead, or prefix-state materialization. T8's
512-thread shared kernel may lose occupancy. Retain slow groups and all failures.
If E054 shows a large baseline deficit, use these measurements to decide whether
44.5% logical byte coverage can plausibly close it before generalizing more code.

## Logical byte arithmetic and source identity

As in the earlier head review, D=14,412,349,440 bytes sums U32 packed weights
and U8 scales for one complete decoder+head projection sweep. It excludes full
embedding storage (only selected rows are read), norms/conv, activations, caches
and allocator/compiler effects. Values below come from the retained parameter
metadata, not measured DRAM traffic or latency:

| Scope | Logical bytes | D share |
|---|---:|---:|
| Head already targeted | 715,161,600 | 4.9621% |
| All backbone projections | 13,697,187,840 | 95.0379% |
| First target: all MLP gate+up | 6,417,285,120 | 44.5263% |
| All currently compatible K5120 backbone projections | 9,356,083,200 | 64.9171% |
| Unsupported K6144/K17408 backbone | 4,341,104,640 | 30.1207% |

Sharing only MLP gate/up ideally replaces T logical reads of those matrices by
one: the denominator reduction is `(1−1/T)*44.5263%`, or33.3947% at T4 and
38.9605% at T8. That is **not** predicted DRAM reduction or end-to-end speedup;
the original rowwise path may already reuse cached weights.

Full small-source SHA256s read for this review:

| File | SHA256 |
|---|---|
| `bench/m5_ultra_256gb/nvfp4_multirow_shared/candidate.py` | `d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e` |
| `bench/m5_ultra_256gb/rowwise_native_verify/candidate.py` | `4ce1b1720dc256b01d30a0f9b19b109bde9949050e98b50a31d308f9e71c8f15` |
| `F/model/qwen3_5.py` | `dae9694d717a0ebe80c9c54317c65f5d206e4d5695bc898fe0e93eb04a8590ce` |
| `F/model/attention.py` | `112997ec74e55ecdb88641e86f7bb29ad8c6c6a73481f1afe463a4332156d52c` |
| `F/model/gdn.py` | `c88096bf3b51a1ad7a24dd4f4e2c71bbb6b220b288b83a9242ca190d6c0a5030` |
| `F/model/layers.py` | `c2c9936df2a34877edb5e1702152583e54549f69b37d7c93e9996af1e7c20908` |
| `F/engine/compiled_decode.py` | `22111101109b9d9c69e8fd114aa114533fe708e462303d5e72bedd6d217da1bf` |

The actual used1023 oracle is outside `CompiledDecode`'s max_len≤256 admission,
so native M1 `Linear` is the baseline. The K5120 shared kernel retains the source
B1 dot reduction order, but this proposal still needs its own real activation
and end-to-end numerical qualification. The existing E040 projection and E042
head results do not qualify new backbone changes.

Implementation follow-through: the separate default-OFF
`bench/m5_ultra_256gb/rowwise_mlp_shared/candidate.py` now supplies the bounded
MLP layer-loop adapter. The first 12/12 guarded CPU attempt passed, retained in
`.artifacts/m5-ultra-256gb/rowwise-mlp-shared-cpu-v1`; this validates fake-array
dependency/order/ownership only. No device canary or speed test has run.
