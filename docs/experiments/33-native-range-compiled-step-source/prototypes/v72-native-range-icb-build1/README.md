V72 is a new native ICB boundary exploration. The old V69 CPU-eval micro STOP (+1.046% descriptive) remains unchanged and is not adopted.

`replay.mm` includes frozen native40build2 verbatim; only six v72 exports are added. No fullmodel, F feedback or S ping-pong is enabled. Original frozen69 R8 helper/header/body bytes are imported by absolute path. Both arms are cold-compiled outside capture/timing; A uses native QMM/R4 and B R8/SG2 over exactly the same immutable per-bank W/scales/X. Eight original MLPgate banks (layers0,8,...56) load401,080,320B checkpoint payload, without new packing/layout or retained extra weight arrays. Their saved original per-layer BF16 X is used.

One capture forces16 dispatches A0..A7,B0..B7. Native enable requires24 pinned original W/scales/X ranges resolving48 active bindings, one exclusive writeonly allocation per dispatch, all16 outputs distinct, and whole-allocation output/read disjointness. Actual active reflection determines read/write roles; unused builtin arguments are not guessed by index. Small nonparameter readonly buffers use native40 snapshots; constants preserve original setBytes. Unknown/access/geometry/dependency conditions STOP. Original functions/full descriptors are retained; only supportsICB is enabled where required. Native report includes actual active argument/geometry records.

One16-command ICB exposes ranges[0,8),[8,16). One ffi call loops a selected8-command range in one CB. Every group ends with an explicit Buffers barrier, including the last group, before those8 output slots are reused. Arena snapshots restore once at build, not per block. Weights/scales/realX are zero-copy only after actual buffer-range resolution. One native loop has one encode/commit/wait; official GPUStartTime/EndTime seconds are optional diagnostic envelopes, not exclusive dispatch sums.

Stage1 is a separate fresh `--qualify-only` marker:3 activation vectors(real/zero/fixedrandom)×3 comparisons×8banks=72 finite BF16 strict checks. Then native group repeats1,2,3,4,5,7,8,16 perarm×8outputs=128 raw-output checks against captured-reference bytes, saved under `numeric/`. All40 original input arrays and16 captured original outputs remain pinned and checked readonly; native small readonly arena clones are also checked. No performance blocks execute in this mode.

Stage2 requires the actual successful Stage1 receipt path+exact SHA before runtime import. A new process/output marker repeats numerical preparation, then frozen prime128groups/timed1024groups=8192dispatches/block;6AB/BA+4AA/BB controls. Python wall spans oneffi per timed block, including native encode/commitwait. Per-slot copies/hashes, resource probes/reports and readonly checks are outside clocks. Gate:≥5% arm-median reduction,≥5/6wins, eachcontrol within5% and mediancontrol within2%. Resources require AC, thermal0, pressure1, unchanged swap-used/Swapouts. No model/HTTP/frontier/adoption claim follows from a local result.

Root owns GPU/process supervision and timeout/reap. Do not run the dylib or script GPU path concurrently with other GPU work. Exact controls/native preload:

```sh
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
DYLD_INSERT_LIBRARIES=/private/tmp/minivllm-engine-optimization-20261014/v72-native-range-icb-build1/libv72replay.dylib \
/Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v72-native-range-icb-build1/micro.py \
--qualify-only --out ABSOLUTE_NEW_QUAL_DIRECTORY
```

After root validates Stage1 and releases/reaps all owned processes, the same CLI without `--qualify-only` uses `--qualified-receipt ABSOLUTE_STAGE1_RECEIPT --qualified-sha ACTUAL_SHA --out ABSOLUTE_NEW_PERF_DIRECTORY`. Both marker directories must be new. CPU-only checks: `python3 -B check_cpu.py`; `python3 -B micro.py --inert`; `./check_contract` is a pure C++ CPU binary without Metal linkage. Loading/executing `libv72replay.dylib` is root-only.
