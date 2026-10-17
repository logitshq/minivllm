# E026: ownership design, recent GDN research, and adoption boundaries

2026-10-17. Read-only source/paper review; no new speed claim. The candidate remains default OFF. Correctness gates admission; the optimization objectives remain complete-request latency and whole-machine output throughput.

## Inspect the competitors' state ownership

The local oMLX `omlx/patches/mlx_lm_mtp/qwen35_model.py:265` explicitly applies `mx.contiguous` to the final convolution rows in the no-lengths branch of `_process_chunk`. Its other branch uses `take_along_axis` for sequence-specific lengths. This is a related GDN/MTP source path, not proof that every previously timed oMLX request selected it. The checkout HEAD is `328edc756010693e61f8bd4e47f00996e0758176`; exact inspected file bytes are pinned below.

Lithos `monolith/nn/gdn.py:54–58` declares fixed-shape convolution and recurrent states, with two checkpoint slots by default. It describes alternating read/write slots plus accepted-prefix rewriting. This is source evidence for explicitly sized persistent state, not a measured memory comparison or justification for importing its whole compiler. HEAD is `223658090ae549e59220bdec459957cbdad7926a`; exact file bytes are pinned below.

Our inference: separate the small persistent state from the temporary prefill computation. E026 implements only that ownership operation, with the original GDN arithmetic intact. A lazy copy still needs evaluation; per-layer immediate evaluation and adding eligible tails to the existing end-of-prefill evaluation are different placements. The latter remains an unimplemented hypothesis and must earn its own numerical and performance result.

## July 2026 GDN optimization case study

[Technical Report: AI-Assisted Gated DeltaNet Optimization on NVIDIA Blackwell](https://arxiv.org/html/2607.16831v1), published July 18. Sections 3–6 distinguish short decode dispatch/runtime costs from structural chunk/WY prefill reformulation. They also document a benchmark-baseline change reversing an apparent advantage over FlashInfer. This is NVIDIA B200 evidence, not Apple performance. Our core already has chunked prefill; the transferable question is where its existing matrix work, dispatch, and state movement spend time. We do not assume algebraic reassociation is bit-identical or port CUDA launch choices to Metal without measurement.

## September 2026 quantization and fused-scale correctness

[Why Gated DeltaNet Survives 4-Bit Quantization](https://arxiv.org/html/2609.04098v1), published September 3. Section 6 reports a mismatch when separately calibrated NVFP4 modules are served as one fused GEMM with a shared global scale. A plausible output or improved perplexity alone failed to expose broken gate scaling. The reported checkpoint and W4A4 recipe differ from our frozen MLX checkpoint. We adopt the audit lesson: verify each original projection and its scales before promoting packed/fused paths. We do not change this experiment's checkpoint, activation precision, cache precision, or numerical acceptance rule.

## Scope of the next experiment

A three-arm real-weight single-layer diagnostic is being prepared: fixed 14 tensors, synthetic finite BF16 inputs, original recurrence and projection code, exact output/state comparison. It requires its own bounded resource contract. It cannot establish full-model correctness, HTTP performance, the complete Pareto frontier, or the cause of E023's resource abort.

## Exact local source identities

```json
{
  "/Users/ryanli/code/omlx/omlx/patches/mlx_lm_mtp/qwen35_model.py": "506b499df8b74ffc717fc598c607977ea6924b052ab23678ca8b36104faba4e5",
  "/Users/ryanli/code/lithos-metal/monolith/nn/gdn.py": "1fbfadb029908c0f7ff526b3f521d27be001982a3f77480007d8bc8d30ff3875"
}
```
