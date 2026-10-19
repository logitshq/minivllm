# E043: native-M8 shared-weight screen — rejected for speed

**Numerical/execution audit PASS; candidate rejected for speed on both measured projections.**
The independent reader checks four saved BF16 output matrices across two T8 cases, totaling
2,125,824 pairwise original-word comparisons: finite and byte-identical. All400 raw timing
samples and the ten groups' exact AB/BA order and medians are independently reconstructed.
All ten group speed ratios are below1.

| Projection | Native M8 latency / candidate latency | Candidate latency increase |
| --- | ---: | ---: |
| head | 0.545830011× | 83.21% |
| up | 0.693705675× | 44.15% |

These are medians of five group ratios, not the ratio of pooled medians. Each group contains
20 AB/BA pairs. The probe constructs a fresh operation for each sample and includes eval,
synchronization and contiguous staging; five warmups per route precede timing. These source
properties are supported by the frozen execution, not independently rerun by this reader.
Every raw timing is retained; each timed output tensor was evaluated/released rather than
saved. Full numerical outputs were saved before timing for each case.

## This experiment has a different arithmetic oracle from E040

The baseline is **one native `quantized_matmul` on X[8,5120]**, never eight independent M1
calls. This candidate preserves the native wide path's kl16/four-dot/scale/shuffle arithmetic
hypothesis. Four output columns share11,520B packed weight/scale storage across eight token
SIMD groups. It aims to avoid the native path's two four-input-row weight tiles, while paying
shared storage,512 threads, one barrier and repeated per-token weight decode. The observed
byte equality qualifies only these actual saved inputs; no hardware counters identify the
cause of the slowdown. Source attribution and pinned native headers are retained.

[E040](../E040-shared-projection-screen-v1/README.md) compared a B1-arithmetic candidate with
separate M1/B1 calls; its1.258× head result and this0.546× result use different baselines and
must not be presented as paired regression measurements. E043 supplies no new engine point.
Native M8 local output equality also does not transfer E042's forward-prefix qualification
to this new kernel. The candidate remains default OFF and is not installed in production.

Both screens use actual pinned Qwen3.8-27B NVFP4 head K5120/N248320 and layer0 up K5120/N17408
weights, with deterministic synthetic BF16 inputs rather than captured hidden states. Repeated
same-weight warm calls may have different cache residency from complete model traversal.
These losses reject adoption based on this screen; they do not establish cache capacity,
DRAM traffic or prove that every streaming whole-model reuse design must lose.

## Execution and retention

The device and parent returned0; the original oMLX PID52818 identity was retained, paused T,
resumed S and healthy before/after. There are five child physical observations, one physical
admission observation, six indexed resource snapshots and one separate after-resume snapshot.
Observed pressure is1 and swap/Swapouts do not increase. The reader reparses saved resource
counters and checks saved parsed physical observations against the registered gates; no live
resource remeasurement or hard allocation guarantee is implied. The exact known104-byte
`mx.metal.device_info` deprecation warning is retained, not treated as a numerical failure.
MLX allocator peak is reported as723,206,144B, separate from physical process footprint.

The plan was committed in `cc3b2bb` before actual execution. Plan SHA256:
`9f0dbea058ee647c0ccf16eb1f9e3badf86a06686e9385ca4ae81d81130ac2ed`.
Candidate SHA256: `63055795d00405e7aa7fb0960ed597fc95da6fdeefd3c28187957ce2387f4a50`.
The8-test CPU mechanism pass and later11-test mechanism/probe pass are both retained with
original source snapshots, commands and results. Historical source READMEs/reference flags
remain preparation snapshots; the completed device evidence is reported here.

`evidence.tar.xz` contains 92 explicit members: raw synthetic input/output tensors,
all timing samples, source and weight identities, complete actual execution/lease/resource
records, native header attribution, CPU attempts and preparation. Every archived member was
reread and matched to its original and manifest: 9,252,440 uncompressed bytes,
3,118,304 archive bytes. Archive SHA256:
`635604cff085caa99c8866f1c211db531c87f5d0ff2dc917015d970d39d72ecb`.

Both real selected-weight payloads (765,296,968B combined) are excluded. Their
original full-shard SHA, header SHA, exact ranges, copied-slice/readback SHA and selected-file
SHA remain in `weights.json`. The offline audit independently checks header/range/manifest
consistency, not a new full model payload hash. No model weights were packaged into Git.

The first offline audit and first archive attempt passed; no failed E043 attempt was discarded.
Extract to a fresh folder and run `python3 -I -B audit.py --evidence EXTRACTED --out FRESH.json`.
`references.json` distinguishes coordinator tool IDs from the retained actual process receipts.
`document-integrity.json` binds outer metadata except itself. No new GPU/service action, test,
benchmark rerun or commit was performed here. A combined later diagnostic chart may reuse the
retained group ratios; no standalone new figure is required for this negative screen.
