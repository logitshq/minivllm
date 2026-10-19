# Bounded numerical device smoke — prepared, not executed

Run only after root reviews and finishes the offline GLM copy. This is a
numerical smoke alongside the restored service, never a timing experiment.
Default invocation describes the contract without source reads, output writes,
timers or backend import. Only explicit standalone-child CLI use admits work:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/nvfp4_multirow_reuse/smoke.py \
  --run --out /absolute/fresh/receipt-directory
```

The parent must retain the actual child command, exit code and output, including
timeout/nonzero exits. Candidate and custom B1 source SHA256 are hard-pinned;
actual MLX 0.32.3, Apple M5 Ultra and g17d are checked before construction.
The smoke source SHA and loaded MLX core path are recorded. No source, model,
checkpoint, service, cache setting in another process or wired limit is changed.
The previously frozen candidate.py is unchanged.

There are 36 fixed cases. All (K,N)=(512,64),(5120,256),(5120,6144) run T2/4/8
with contiguous and constructed column-stride2 x/packed-weight/scales views.
The smallest shape additionally tests positive zero, original BF16 negative-zero
bits and finite stress inputs. Weights originate from deterministic BF16 values
and actual MLX NVFP4 quantization, group16/bits4 without global scale. The input
recipe is fixed in code; no observed failure may be removed or regenerated.

Dense weights are generated and quantized in at most 256-row bands. Source BF16
bands and packed weights/scales are saved once per shape, with per-file hashes;
each case references that immutable artifact and saves actual logical x plus all
three full BF16 output matrices. Layout recipe is retained separately from the
logical on-disk values. The API explicitly stages noncontiguous inputs; this is
not a proof that the compiler retains an internal stride2 view.

Each output row is compared in original BF16 bytes with separate actual native
M1 calls AND separate custom-B1 calls. Native M=T QMM is never an oracle.
Oracle rows are materialized one at a time so repeated strided-weight staging
cannot accumulate sixteen simultaneous copies; this is numerical-only work.
Candidate/native, candidate/custom and custom/native comparisons, first differing
row/column and raw words, full output hashes and finite checks are retained.
Failure stops subsequent cases and preserves inputs, outputs and partial work.
No tolerance, FP32-converted digest or argmax-only acceptance is used.

Constructed buffers remain small: packed largest weight plus scales is about
16.875 MiB; a dense band is 2.5 MiB, with no full dense matrix or full FP32 weight
temporary. Concatenation and stride2 copies are admitted within the 256 MiB
live-buffer envelope. Each materialization checks actual MLX buffer size,
active/cached memory and cumulative peak below 256 MiB. The MLX memory limit is
also set to 256 MiB, but it is an allocator guideline, not a hard process-RSS or
system-memory guarantee. Saving raw evidence, compiler metadata and host memory
are outside that MLX counter. There is no inference performance claim.

The child arms a 120-second default-action SIGALRM before backend import. This
terminates the child even if a native call does not return to Python; it does
not depend on a Python exception handler. An incomplete receipt is written
first. Such termination may leave a partial safetensors file and no final result:
only an actual zero child exit plus final PASS and all 36 case results can pass.
The parent should still impose its own bounded wait and retain the exit. No
recovery/service framework is added and no foreign PID is targeted.

The one `test_smoke_cpu.py` inert-entry/argument check passed **1/1** under the
forbidden-operations guard (tool `5d14cc`), with no failures, errors, skips or
forbidden operations and six tracked sources unchanged. Independent source review
found no blocker in the stated numerical-only scope; installed MLX API signatures
were inspected without importing the backend. The exact receipt and test-time
sources are archived at
`docs/experiments/m5-ultra-256gb/nvfp4-multirow-smoke-cpu-v1/`, with every copied
member and original file independently SHA-verified. The test-time version of
this document is retained unchanged in that archive.

**The actual GPU smoke remains unrun.** CPU entry checks do not qualify numerical
equality or the actual device memory envelope. Success of the future smoke would
qualify only these synthetic projection cases. Actual model activations/weights,
full prefix logits/states, compiled integration and performance remain unqualified.
