# V23 conv numerical runner — ready for root review, CPU preparation only
- Run `python3 -B /private/tmp/minivllm-engine-optimization-20261014/v23-conv-numeric-build1/check_cpu.py` for source/evidence checks; no MLX import.
- GPU argv (root only):
```sh
/Users/ryanli/code/minivllm/.venv/bin/python3 -B /private/tmp/minivllm-engine-optimization-20261014/v23-conv-numeric-build1/runner.py --execute-gpu --sum-y 4 --sum-evidence /private/tmp/minivllm-engine-optimization-20261014/v23-root-control-build1/native-pipeline.json --checkpoint-dir /Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6 --weight-layer 0 --state-bank /private/tmp/minivllm-engine-optimization-20261014/v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1/real-L0-D1 --chain-steps 8 --output-dir /private/tmp/minivllm-engine-optimization-20261014/v23-conv-numeric-build1/gpu-numeric-run1
```
- Estimate: cold shader compilation roughly 30–120s; actual duration unmeasured. Numerical arrays <40MiB live, allocator pool limit64MiB and MLX memory limit512MiB; process/compiler RSS is additional, not a hard512MiB process cap.
- Reference is AST-extracted from frozen accepted15 GDN `_decode` w/concat/product/sum; nn.silu unchanged. Original recurrence module also comes from accepted15, not currentW26/V19-defaultOFF.
- Candidate is the immutable archived V23 prototype. A separate probe changes only act store to conv store and tail-output identifier, sharing exact loop/header/options.
- sum_y is mandatory. This build binds root nativePSO JSON M1024/width32→Y4. Independent metadata selects the single branch before outputs; no branch scan or output-based selection.
- Every case records actual original product shape/element strides through ensure_row_contiguous=False GPU descriptor probe. A mismatch from [1,4,10240]/[40960,10240,1] stops before candidate launch.
- Internal C++ flag bits are NOT exposed by this runner; row/column flags derived from observed strides are labeled derived. This remains explicit pending evidence, not an invented native-flags observation.
- Six deterministic preconv cases: zero-tail/zero-raw, zero-tail/nonzero-raw, nonzero-tail, BF16 cancellation, per-product rounding and head boundaries.
- Only one small real checkpoint conv tensor range is read (81,920B plus JSON header), reshaped exactly like accepted15 loader; no mx.load(shard), 27B model construction, tokenizer or model run.
- Real saved g/beta/S from V18 are validated by bank metadata/hash and used for unchanged recurrence o/S bytes and an8-step independent-state/tail chain.
- Historical V18 real banks retain postconvu only; V19 has no preconv raw/tail/w banks. Synthetic preconv plus actual weights/state does NOT qualify genuine real activations.
- Optional `--real-bank DIR` requires bank.json arrays raw/old_tail/w/g/beta/S with shape,dtype,file,sha256 and little-endian blobs; u-only old banks are rejected for this role.
- Conv, act, tail, probe-tail, o and S_new compare shape/dtype then strict finite raw bytes. All inputs hashed before/after; first failure preserves receipt and stops. No tolerance or argmax substitute.
- Success status is only PASS_CONV_NUMERIC_SYNTHETIC_ONLY (or WITH_REAL_BANKS_ONLY), never model/engine qualification. Logits/IDs, final cache, lifecycle and real preconv capture remain later gates.
- Before use root retains GPU ownership/resource/ownedPID guards; runner does not spawn background workers or load serving apps. Output dir must be new; old artifacts/prototype are never rewritten.
- MLX typed helpers retain complete Apple MIT notice in LICENSE-MLX-MIT. Prior V18 activation-only PASS is source provenance, not V23 GPU numerical evidence.
