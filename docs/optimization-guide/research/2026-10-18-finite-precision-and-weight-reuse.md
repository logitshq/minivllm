# Exact verification and useful weight reuse

Read 2026-10-18 while the observer-OFF combined timing cycle runs. Research and
source preparation only; no new device result or imported speedup.

[Evaluating Losslessness in Speculative Decoding Under Finite-Precision Inference,
v2, 2026-09-27](https://arxiv.org/html/2609.15504v2), sections 3.5 and 4.2, reports
43–45% exact BF16 trajectory agreement for two Orthrus/Qwen3-1.7B checkpoints on
1,190 prompts, versus full observed agreement in FP32. It does not localize the
first differing operation. This is evidence that a theoretical lossless argument
and similar task scores do not establish exact finite-precision output equality.
It is not evidence about our NVFP4 model or a reason to change its precision. Our
rowwise experiment therefore keeps the original C1 arithmetic and checks each
prefix's full logits and all128 cache arrays before any acceptance experiment.

[Lithos's official 2026-10-08 design article](https://www.lithosai.com/newsroom/lithos-metal)
describes layer-level mixer fusion, separate MLP kernels and pre-encoded indirect
command replay. It retains recurrent-state column slices across multiple tokens;
DSpark then amortizes target weight reads over accepted proposals. This suggests
two distinct opportunities: fewer intermediate writes inside a mixer, and fewer
weight reads per committed token. Its headline rate is on M5 Max with DSpark and
must not replace our independently measured M5 Ultra HTTP points. The article's
own distinction between projected target throughput and full generation is also
material when evaluating a proposed verifier.

[MacPaw's ANE study](https://research.macpaw.com/projects/neural-engine), accessed
2026-10-18, studies LFM2.5-2.6B on M2/M3/M4 Pro, comparing energy and latency within
an 8-bit quality class. It reports lower effective ANE weight-read bandwidth than
the GPU and uses multi-token verification to reuse weights. Its 98.4% top-1
agreement criterion is not our exact native-state contract. The readable page
has no publication date; search recency is not a verified publication date.
An ANE port is therefore not an immediate substitute for the current exact Metal
optimization, nor evidence of a gain on this M5 Ultra or either target model.

## Local experiment decision

For an independently timed speculative cycle, record actual committed count A
and total cycle time C, including draft, verification, publication/rollback and
host work. Across cycles use sum(C)/sum(A), not mean(C/A) or mean(A) alone. Compare
this with the same-workload ordinary decoder, then require a separate full-request
HTTP improvement. Prefill, EOS and rejected proposals remain in the accounting.
These are our measurement decisions, not reported paper results.

The new default-OFF layer-major rowwise prototype preserves each original C1
operation while placing same-weight operations close together. Whether MLX
actually reuses cache lines or reduces time is unmeasured. Concatenation and full
prefix-state retention can instead make it slower. First establish exact fixed
teacher-forced outputs, then measure complete forward cost against T ordinary
steps; stop this candidate if it cannot amortize that cost. A future trained
drafter cannot repair an incorrect or uneconomical verifier.
