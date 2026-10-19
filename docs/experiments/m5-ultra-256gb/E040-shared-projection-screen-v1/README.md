# E040: shared-weight local projection screen

**Independent numerical/execution audit PASS. Head T4/T8 improve against both local baselines;
other cases remain mixed.** All six saved cases are finite and three-way BF16 byte-identical:
18 full output matrices, 3,720,192 elements per route, 11,160,576 pairwise word comparisons.
All 2,400 raw timing samples and the full baseline/group/pair/AB–BA order are independently
reconstructed. Seven of twelve median ratios exceed 1. This is a local projection screen,
not an engine Pareto point or full-model qualification.

![All five groups and their median ratios](local-projection-speed.png)

| Projection | T | Native M1 / candidate | Custom B1 / candidate |
| --- | ---: | ---: | ---: |
| head | 2 | 0.991231× | 0.928371× |
| head | 4 | 1.169216× | 1.095522× |
| head | 8 | 1.258057× | 1.136031× |
| up | 2 | 0.910122× | 0.985041× |
| up | 4 | 1.002304× | 1.022069× |
| up | 8 | 0.995709× | 1.073009× |

Ratio means **baseline latency / candidate latency**; above 1 favors the candidate. For each
baseline/case, divide its 20-sample median by the candidate's 20-sample median within each of
five sequential groups, then take the median of those five ratios. This is not the ratio of
pooled medians. Native-vs-candidate groups precede custom-vs-candidate groups. The plot retains
all 60 group ratios and marks twelve medians; these are observed within-process variation,
not independent-run confidence intervals. No group was dropped.

Head T4/T8 have all five group ratios above 1 against both baselines. Relative to native M1,
their median ratios are 1.169216× and 1.258057×, corresponding to approximately
14.47% and 20.51% lower candidate latency. Head T2 remains below parity on both median comparisons.
The smaller up projection is mixed: T4 native is only 1.002304× with groups on both sides of
parity; T8 native is 0.995709× even though custom B1 is 1.073009×. These measurements support
further head T4/T8 qualification, not enabling the kernel across every projection or T.

The candidate shares one 5,760-byte packed-weight/scale tile between token SIMD groups while
retaining the B1 per-row arithmetic order. Unlike the rejected register-reuse implementation
in [E038](../E038-real-projection-screen-v1/README.md), it shows local head wins. E038 and E040
are separate runs, not paired samples; this archive makes no causal hardware-counter claim
about occupancy, bandwidth, registers or the reason for their difference.

Weights are the pinned Qwen3.8-27B NVFP4 vocabulary head (K=5120, N=248320) and layer-0 up
projection (K=5120, N=17408). Inputs are deterministic synthetic BF16 values, not captured
hidden states. Independent native M1 calls, custom B1 calls and the shared candidate produce
saved full original-dtype outputs before timing. Timed outputs are evaluated and released,
not saved per sample. The frozen source and execution support five route warmups, fresh
operation construction, eval/synchronization, baseline row slicing/concat and candidate
staging/shared-copy cost inclusion. The offline reader does not rerun any MLX computation.
Only contiguous real-weight input layouts are covered here.

The child and owner completed with exit 0. Original oMLX PID 52818 retains its start identity,
with observed T→R resume and before/after healthy original-model receipts. Four indexed
resource snapshots (one before pause, three during the child) plus an after-resume snapshot
all report pressure level 1, unchanged swap used 2278.44 MiB and unchanged Swapouts 807478.
These observations do not establish continuous isolation or exclude unobserved GUI/GPU activity.
The recorded runtime is MLX 0.32.3, Apple M5 Ultra (`applegpu_g17d`). Reported MLX allocator peak
is 727,269,376 B; this is a producer allocator counter, not independently measured physical RSS.
The one exact saved stderr message is the known `mx.metal.device_info` deprecation warning;
it is retained and is not a numerical failure.

## Reproduction and evidence

`evidence.tar.xz` contains 78 explicit original members: raw inputs/outputs and timings,
producer summaries, weight references/headers, execution/lease/health/resource records,
frozen sources, seven mechanism CPU tests' evidence, and the one CPU entry-test evidence.
Every member was reread and SHA-verified against its original and `manifest.json`:
23,208,015 uncompressed bytes, 3,110,068 archive bytes. Archive SHA256:
`57800d10708e247bba3dac8d6f4635644fede945b22275f8e2a2a3f6d106fcb7`.

The two real `selected-weights.safetensors` payloads (765,296,968 B combined) are
excluded. `weights.json` retains original full shard SHA, header SHA, exact ranges, slice SHA
and selected-file SHA. Header/range/manifest consistency is independently checked; original
payload SHA assertions are producer evidence and were not rehashed by this offline reader.
No real model weights are included in this documentation archive.

The first E040 offline audit and first render both completed successfully. No E040 failed
audit or render attempt was overwritten. The reader retains E038's explicit handling of the
observed safetensors `__metadata__: null` field, and checks this run's actual T→R receipt.
`references.json` distinguishes coordinator-reported tool/session IDs from saved execution
records. All actual command stdout/stderr/return codes are retained.

Extract the archive into a fresh folder and run
`python3 -I -B audit.py --evidence EXTRACTED_FOLDER --out FRESH_RESULT.json`.
`render.py` reads `independent-audit.json`; `render-command.json` records the installed plotting
interpreter. Render in a fresh copied folder because existing image outputs are refused.
`render-receipt.json` binds source/data/figure SHA, and `visual-review.json` records actual PNG
inspection. `document-integrity.json` binds all surrounding files except itself.

No GPU, service action, benchmark rerun or commit was performed by this offline audit/archive.
