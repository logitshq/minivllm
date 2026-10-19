# E037: NVFP4 multirow reuse — synthetic device smoke

**36/36 synthetic cases passed the actual device run and independent offline raw-byte audit.**
All 108 saved BF16 output matrices are finite. The candidate, separate native MLX M1 calls,
and separate custom B1 calls produced identical saved bytes: 186,368 elements per route,
559,104 pairwise BF16-word comparisons, no mismatch. This is a numerical smoke result;
**no full-model qualification, performance result or Pareto point was produced.**

| K | N | T | Input kinds | Declared layouts | Cases |
| --- | --- | --- | --- | --- | ---: |
| 512 | 64 | 2, 4, 8 | finite, zero, negative zero, stress | contiguous and column-stride 2 | 24 |
| 5120 | 256 | 2, 4, 8 | finite | contiguous and column-stride 2 | 6 |
| 5120 | 6144 | 2, 4, 8 | finite | contiguous and column-stride 2 | 6 |

The independent standard-library reader checks every saved output word, not just producer
hashes. It checks complete cases, tensor keys/dtypes/shapes, byte-span coverage without gaps
or overlap, all artifact hashes, execution closure and frozen source bindings. It also checks
all 32,800,768 saved BF16 source-weight values against the fixed recipe and every saved input
against its recipe, including the negative-zero bit pattern. The packed weights/scales contain
18,450,432 bytes. These are deliberately patterned synthetic values, not model weights or a
representative activation distribution. The reader does not independently recompute NVFP4
quantization or the MLX matrix products.

The frozen smoke source issues independent M1 calls for each token on both reference routes;
it does not use native M=T as the oracle. Actual native dispatch, quantizer execution and
kernel compilation are supported by that frozen source plus the successful execution records,
not a separate hardware trace. Saved safetensors preserve logical values, not physical strides;
the audit confirms identical logical input/output values across each declared layout pair.
Physical stride construction and staging are supported by the source and execution only.

The runtime receipt reports MLX 0.32.3 on Apple M5 Ultra (`applegpu_g17d`). The reported MLX
allocator peak is **71,516,160 B (68.203125 MiB)**, below the 256 MiB smoke guard; final active,
cache and argument-buffer counts are zero. Those are producer allocator observations, not
independent process RSS or system memory measurements. The execution context records restored
oMLX alongside this numerical check; the coordinator also reported other services present.
There was no timing isolation or service-inventory audit here. Neither driver duration nor
allocator values establish engine speed or whole-model memory fit.

## Evidence and failed audit attempt

`evidence.tar.xz` preserves all **208 original members / 86,000,324 uncompressed bytes**:
180 device files (original synthetic inputs, source bands, packed weights/scales, outputs and
receipts), 6 execution files, 12 smoke CPU files and 10 mechanism CPU files. The original
`incomplete.json` intent is retained unchanged; successful driver exit 0, 36 case receipts and
final result establish completion. Existing CPU evidence is 1 smoke guard test and 5 mechanism
tests, all PASS; none was rerun for this archive. Raw device evidence was never modified or rerun.

`manifest.json` lists every explicit original member, its original path, size and SHA256.
`archive-verification.json` records a second read of every compressed member and rechecks every
original source digest. The final archive is 68,068 B; patterned synthetic values compress well.
Archive SHA256: `8240f5a157f92bac880073c632ea3edf23ccaac61cdb2e4b44da4aa46adeb9c5`.

The first offline audit failed because it incorrectly required empty stderr. The real run
emitted exactly this 104-byte, already observed warning:

> mx.metal.device_info is deprecated and will be removed in a future version. Use mx.device_info instead.

`attempts/audit-v1-failed/` retains that auditor and archive script, actual command, logs,
return code, failed result, complete raw archive and verification. The second auditor changes
only this acceptance rule to require that exact warning; arbitrary stderr remains a failure.
It reports **PASS** in `independent-audit.json`. This was an audit assumption failure, not a
failed device numerical case. Both archive manifests describe identical original members.

## Reproduce the independent check without a device

Extract `evidence.tar.xz` into a fresh directory and use a fresh result filename:

```sh
mkdir /tmp/e037-recheck
tar -xJf evidence.tar.xz -C /tmp/e037-recheck
python3 -I -B audit.py --evidence /tmp/e037-recheck --out /tmp/e037-recheck-audit.json
```

`audit.py` needs only the Python standard library. It does not import MLX, execute a kernel,
load a model or touch services. Original execution paths remain historical provenance; the
reader resolves retained files from `--evidence`. `references.json` records source pins and
execution/audit references. `document-integrity.json` covers the surrounding documentation,
audit scripts, manifests, results and both compressed archives; it excludes itself.

The next numerical gate remains real model weights and recorded activations, then complete
prefix/state/logit qualification. Only after that can isolated projection and full-verifier
cost measurements test whether this weight-reuse design improves latency or throughput.
