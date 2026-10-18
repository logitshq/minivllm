# ReaDy 8ac61b5: source-only compatibility review

Reviewed `/Users/ryanli/code/ReaDy` at commit
`8ac61b51d529fad980566cfde7a3c41415ba93a5` and the native `tutorial_code/miniengine`
sources. No backend import, checkpoint download, model execution, or new benchmark.
The root research note is already committed; this note records integration decisions.

## Concrete interface differences

1. **Square states hide an axis mismatch.** ReaDy `MODEL_SPEC.txt` and
   `ready_inference/reader.py` expect `[B,2,48,value=128,key=128]` from target layers
   30 and 62. Native `cache.py` records `S[Hv,Dk,Dv]` and its Metal recurrence
   indexes `S[dk,dv]`. A bridge must select the actual request's two committed
   states and transpose the final axes; identical dimensions do not prove
   semantic layout. Confirm those configured layers are recurrent before access.

2. **Use the committed-cache boundary.** `engine/spec_runner.py:49` calls the
   drafter before target verification; lines 51–59 snapshot, verify and rollback
   to the accepted prefix. An instance-local reader input can be set immediately
   before that existing draft call from `caches[30/62].S`, with offsets equal to
   the request's `num_computed_tokens`. Never read the provisional full verify
   result before rollback. Target-only `model_runner.py:82–102` also keeps an
   uncommitted `_ahead` cache; that is not the reader source. Current speculative
   mode disables this ahead path, so its ownership should remain unchanged.

3. **The draft input contract is not plug-compatible.** Official
   `adapter.py:36–45` reshapes the hook input into `[B,7,5120]` and injects after
   draft layer 4's `post_attention_layernorm`. Native DFlash2 passes the entire
   `[anchor, masks...]` block through all layers (`spec/dflash2.py:163–174`), with
   default block length 3 or 5 (`spec_runner.py:48`). It also has DynamicConv and
   its own selector. Matching hidden width/layer index is insufficient: pin the
   released full trained DSpark draft architecture and block semantics before
   adopting its five reader tensors. Do not attach random weights or assume that
   seven draft proposals and seven hook positions mean the same thing.

4. **Backend and numerical domain differ.** The official release is
   PyTorch/Triton/CUDA, BF16 unquantized target/draft/KV, FP32 recurrent state.
   Our qualified target is NVFP4 on MLX/Metal. The target family and reader
   dimensions align, but there is no provided MLX kernel or established reader
   acceptance gain for this quantized target. `kernels.py` retains reduction and
   matrix products outside its fused pointwise kernels, with explicit BF16
   rounding. A mathematically equivalent MLX port is not already bit-qualified.

5. **Its runtime patch changes more than draft proposals.**
   `ready_inference/patches.py` forces deterministic target verification attention
   and changes EOS-at-cap precedence in SGLang. Do not copy either global patch
   into native serving. Keep native normal-EOS/cap semantics and qualify target
   verify/rollback against the original ordinary greedy path.

6. **Released coverage is small and not our HTTP gate.** `VALIDATION.json` has
   six CPU reader cases, 24 B200 kernel cases, 12 graph replays and three C1
   natural-EOS generations (43/74/55 tokens, cap256). Adapter capacity is B<=4;
   that is not evidence for our C8 scheduling or D512/normal-EOS workload. The
   reported paper gain excludes prefill and cannot be inserted into the M5
   POST→DONE chart.

## Smallest useful next gate

First measure existing native speculative draft/verify/rollback cost and accepted
tokens per proposal on the exact target; no reader port is needed for that
diagnosis. If worthwhile, make a default-OFF per-drafter adapter (no class-global
monkeypatch) around the committed-cache boundary above. Start with a fixed C1
block whose full trained draft contract is verified. Check a deterministic
non-symmetric state fixture to expose axis errors, then actual target state and
draft activations; compare complete reader outputs in original dtype against a
pinned reference, and check target logits/cache/IDs through accepted-prefix
rollback and EOS-at-cap. Only afterward test actual C1 HTTP; C8 needs a separate
batch/request-index gate and is outside the released implementation's B<=4 bound.

Two selected FP32 states total 6 MiB per request; a separate materialized gather
reads and writes at least 12 MiB per proposal, before reader operations. The
official preallocated B4 state buffer is 24 MiB. These are byte counts, not time
predictions: a Metal implementation may avoid an extra copy through controlled
views, but must preserve request ownership and committed-state lifetime. Reader
preparation should be reused within one proposal; the benefit must exceed its
normalization/projection and state-access costs.

Primary source links: [official implementation at the reviewed commit](https://github.com/StartLuxLabs/ReaDy/tree/8ac61b51d529fad980566cfde7a3c41415ba93a5),
[model interface](https://github.com/StartLuxLabs/ReaDy/blob/8ac61b51d529fad980566cfde7a3c41415ba93a5/MODEL_SPEC.txt),
[adapter](https://github.com/StartLuxLabs/ReaDy/blob/8ac61b51d529fad980566cfde7a3c41415ba93a5/ready_inference/adapter.py),
[validation scope](https://github.com/StartLuxLabs/ReaDy/blob/8ac61b51d529fad980566cfde7a3c41415ba93a5/VALIDATION.json).
