# B1 projection and GDN preparation fusion

`LLMEngine(..., compile_decode=True, fuse_decode=True)` additionally concatenates
MLP gate/up, attention Q/K/V, and GDN QKV/Z/B/A output rows. It uses the same
NVFP4 dot products and the original parameter arrays remain model parameters.
The derived copies live in the runner and refresh on array replacement.
Recreate the engine after changing model topology or non-array configuration.

A single GDN preparation kernel combines convolution, BF16 SiLU, serial Q/K
norms, and convolution-state advance. It reuses the qualified state-update
kernel. BF16 products AND additions round at each step; replacing the sum by
FP32 accumulation fails exact agreement and was rejected in exploration.

Fusion is default off, limited to the pinned M5 Ultra/MLX 0.32.3 27B NVFP4/BF16
configuration, batch 1 and used length <=256. B2/B4 retain the qualified graph;
other contexts retain eager execution. It does not change prefill, speculation,
weights, prompts, or generated token counts. First use allocates derived weights
and traces the graph; steady-state results explicitly exclude those cold costs.

`qualify.py` checks 214 full forward pairs (logits and 128 cache arrays each),
including four prompt contexts, mixed histories, B1/B2/B4, boundary fallbacks,
12 packed weight/scale replacement and restoration cases, and unchanged original
parameter keys, shapes, dtypes and content hashes. It also checks live scheduler insertion, removal,
lookahead and stopping. `qualify_kernel.py` checks synthetic scale/state cases
and strided inputs directly against the original operator chain.

`frontier.py --prepare` freezes current sources after qualification. Running
without `--prepare` creates 24 sequential, fresh processes, three per condition:
C1 previous accepted graph, fusion, oMLX, Lithos; C2/C4 fusion and oMLX. Every
process has three warm and three measured groups. The baseline comes from its
frozen accepted source, and both native variants use the same 2 GiB allocator
pool and 50/50/20/0 MLX controls. Compiled gates are on; the separate V19 toggle
is off. No prompt cache or draft; greedy, normal EOS, 19 input/32 output tokens.
Lithos uses its locally adapted M5 Ultra profile; its OpenAI endpoint is C1 only.

All attempts remain recorded, including the discarded shared-function compile
probe and the FP32-convolution mismatch. Use ../lithos/guarded.py for standalone
GPU qualification; frontier owns and reaps its own servers. Never overlap with
the separate large-model benchmarking job. Record paging and swap deltas,
including any pre-existing nonzero swap; do not claim that swap was always zero.
