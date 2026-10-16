V66 standalone original-pointer GDN four-projection numeric prototype

No engine integration, model inference, native ICB adoption, timing or performance claim.
Old V16 concat/slab 4→1 STOP-MICRO-GATE evidence remains unchanged. Concat was outside its timing; avoiding concat alone does not prove steady-state improvement.

Source: exact pinned MLX0.32.3 fp4/fp8/helper/qdot/fp_qmv_fast_impl text, MIT license retained. R4/SG2/64threads/8rows unchanged; each TG picks one of four disjoint local-Y sections (1280/768/6/6 tiles). Four original U32 weights + four original U8 scales + one shared BF16 X, four separately allocated output arrays. No concat/slab/rebind/repack/weight copy. Declared main kernel13buffer arguments≤31; actual Metal reflection remains required before native graph adoption.

Only B1 K5120,N10240/6144/48/48, NVFP4group16bits4. Helper retains strong original handles. ensure_row_contiguous=False; numeric independently probes actual shape/strides before prepare, then verifies active allocation unchanged. Exact strides required; unknown views reject.

Numeric loads24 safetensors ranges for GDN layers0/32/62 only (142387200 bytes total). Eight deterministic synthetic BF16 vectors plus saved L0/32/62 MLP post_attention_layernorm BF16 vectors are generic tests, explicitly NOT GDN-origin or fresh captures. Every layer uses all11 vectors. Compare all4 outputs QMM↔same-R4 control↔four-pointer candidate:396 strict shape/dtype/finite/raw checks. All35 original W/scales/X retain exact metadata/address/SHA after every layer; output ranges must be distinct and not overlap inputs.

CPU: run `python -B check_cpu.py` and `python -B numeric.py --inert`; no MLX/NumPy/native/GPU import or checkpoint payload load. Mapping/metadata/source checks only, no fake numerical tests.
Root-only actual numerical CLI through existing sole-GPU timeout/resource/PID owner:
```sh
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
/Users/ryanli/code/minivllm/.venv/bin/python -B \
/private/tmp/minivllm-engine-optimization-20261014/v66-gdn-four-projection-build1/numeric.py \
--run-gpu --out /private/tmp/minivllm-engine-optimization-20261014/ROOT-CHOOSE-NEW-V66-OUTPUT
```
Expected local qualification seconds to tens of seconds; selected MLX weights136MiB plus small vectors/outputs. Allow additional runtime/compiler/RSS overhead; allocator cache cap2GiB is a cap, not resident allocation. Runner contains no child supervisor; root owns timeout/reap/release. No DYLD injection.

Any byte/finite/readonly/layout/source mismatch STOP; no tolerance widening. Numeric PASS only unlocks separately declared local micro protocol. Model/full-greedy/cache/ICB/lifecycle/service qualification still required. Combined producer may delay qkv consumption until z completes and lose original conv overlap; no≥2ms or model gain assumed.
