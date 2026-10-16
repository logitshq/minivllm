# V81: two-kernel GDN native range, numeric-ready only

A is the frozen V44 activated prework plus held32 state; B is the frozen V78 FP32 scaled-ready prework plus its new state PSO. V78's actual local 30 o/S +30 Q/K-ready +30 V/scales +15 read-only checks passed in the pinned receipt `f5d8fc89…ec7807a`. That receipt does not qualify native replay. V81 has not run GPU code.

The same 15 saved banks (9 historical real, 6 synthetic) supply fixed, externally materialized BF16 activation and read-only FP32 g/beta/S to both arms. S never receives output feedback. The 15 original S payloads occupy 47,185,920 decimal bytes (45 MiB); the full native footprint additionally includes held outputs, snapshots and one shared arena, and is reported from actual capture. This is a rotating input working set, rather than one hot S repeated in isolation.

The V72 dylib is not reused: its contract permits only 16 independent QMV dispatches. V81 includes frozen native40's capture/arena source verbatim and adds six independent exports. One 60-command ICB captures `A0.prework,A0.state,…A14.state,B0.prework,…B14.state`; ranges are `[0,30)` and `[30,60)`. Each bank performs exactly two dispatches. Every state has `setBarrier` before it; each repeated 30-dispatch group ends with a Buffers barrier, including the final group before output slots can be reused. Original capture must show the prework→state synchronization and exact producer/consumer allocation bindings.

The actual reflection/function/full descriptor and original geometry must match four exact PSO names and exact argument indices/names. All activated/g/beta/S inputs require actual ReadOnly reflection; S has no RW exception. Store-only output parameters may be reflection WriteOnly, or reflection ReadWrite with the narrow pinned source proof in `source-access-proof.json`. Reports keep that distinction. Unknown or different access, names, indices, aliasing, dependencies or geometry STOP. All 180 mutable registered arrays have exclusive original allocations; the only shared original allocations are the identical 60 A/B immutable inputs. Every packed arena allocation extent is disjoint. Original setBytes payloads are copied once into an immutable constant bank; shape buffers are snapshot-checked. Future actual reflection and numeric qualification remain mandatory.

`qualify.py` is inert by default and exposes only `--qualify-only` for GPU work. It verifies the public A/B paths (30 strict o/S checks), both captured two-kernel paths against public A44 (60 checks), then native groups 1/2/3/4/5/7/8/16 per arm (480 terminal finite FP32 raw-byte checks). All comparison hashes and final 60 native raw arrays are saved; the first mismatch saves both raw sides and stops. The original 75 saved/activated inputs and 180 captured original arrays remain pinned and hash-checked through 17 native read-only checkpoints. Native S/S_new must actually resolve to slots 6/9. No model or HTTP path is loaded.

Performance is disabled in this build. `protocol.json` freezes the later V72-style plan: 128 prime groups, 1024 timed groups, one FFI/CB encode/commit/wait per block, 30,720 dispatches per timed block, six AB/BA pairs plus four AA/BB controls. Required gates are ≥5% arm-median reduction and ≥5/6 wins, each control ratio within 5%, and each-arm control median within 2%. A separate actual V81 numeric PASS receipt plus its SHA is required before implementing/enabling performance. Numeric wall/GPU CB envelope diagnostics are not performance evidence.

CPU checks: `python3 -B check_cpu.py`, `./check_contract`, `python3 -B qualify.py --inert`. The contract binary has no Metal linkage. The native dylib is compiled on CPU and is never loaded by these checks; compilation does not prove Metal behavior. All upstream MLX MIT notice bytes are retained.

Only root may run a fresh numeric marker under sole GPU/process ownership:

```sh
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
DYLD_INSERT_LIBRARIES=/private/tmp/minivllm-engine-optimization-20261014/v81-gdn-two-kernel-native-range-build1/libv81replay.dylib \
/Users/ryanli/code/minivllm/.venv/bin/python -B \
/private/tmp/minivllm-engine-optimization-20261014/v81-gdn-two-kernel-native-range-build1/qualify.py \
--qualify-only --out ABSOLUTE_NEW_NUMERIC_DIRECTORY
```
