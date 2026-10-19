# Native M8 shared-weight hypothesis

Default OFF, fixed BF16 `X[8,5120]`, NVFP4 U32 `W[N,640]` and U8 scales `[N,320]`, `N % 4 == 0`; MLX 0.32.3 on Apple M5 Ultra / applegpu_g17d only. `build(mx)` is inert; explicit ON rejects unsupported inputs without fallback. This namespace has CPU mechanism evidence only until a separate actual device receipt is recorded. It changes no engine defaults.

The pinned native `fp_qmv_wide_impl` uses two four-input-row tiles for M8, four output columns and 16 K lanes. This prototype loads one four-output-column weight/scale tile into 11,520 bytes of threadgroup storage, then computes eight input rows in parallel with 512 threads. Each result retains native `g += 16`, four float4 dot products per group, scale accumulation and shuffle distances 8, 4, 2, 1. The stored half-SIMD leaders do not cross output-column boundaries. Copies have unique writers and all threads reach one barrier.

The intended logical global-load reuse is two native four-row tiles to one shared eight-row tile. This is not a measured DRAM reduction. Native reuses each decode across four input rows; this prototype repeats decode per token and pays additional threads, shared-memory accesses and synchronization. Arithmetic source preservation does not establish binary equivalence: safe custom-kernel compilation may differ from the installed native metallib. No B1 `contract(off)` policy is imposed; a true native M8 oracle is mandatory. Source and license pins are in `references.json`.

## Minimal actual screen

`real_projection_probe_v1.py` is a small adaptation of the E040 probe. It retains the fixed checkpoint, full source-shard SHA checks, header and selected-slice readback SHA, original head/up shapes, original BF16 bytes and finite checks, fresh output requirement, four-GiB allocator limit, 256-MiB cache limit and 120-second child deadline. Its only two routes are one `mx.quantized_matmul(x,w,s,**Q)` for the entire M8 tensor and the candidate. It must never substitute independent M1 calls.

Each head/up case first compares the whole `[8,N]` BF16 output; any mismatch is saved and stops timing for that case. Five warmups per route precede five groups of 20 paired AB/BA trials: 200 timed samples per case, 400 total. Samples include fresh graph construction, evaluation, synchronization and any contiguous staging. Inputs are deterministic synthetic activations, not captured hidden states.

Repeated hot local projections may have different cache residency from cycling the complete roughly 14-GB logical model projection set. Cache size, actual DRAM traffic and whole-engine gains are unknown. A hot up-projection loss alone cannot exclude a benefit with sequential full-layer weights. The screen yields neither a full-model numerical qualification nor a throughput/latency frontier point.

Default describe (no device import):

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/nvfp4_m8_shared/real_projection_probe_v1.py
```

Device child command, only under the root-owned pause/restore and physical-memory controller, with a fresh output directory:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/nvfp4_m8_shared/real_projection_probe_v1.py --run --out /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/m8-shared-real-projection-v1
```

No device run was performed by the implementation subagent. The proposed parent independently samples a four-GiB child physical ceiling; this is distinct from the child allocator setting and is not a hard allocation guarantee.

## Executed CPU attempts

- `cpu-v1`: 8/8 PASS, candidate dispatch, bounds, unique writers, barrier reachability, source arithmetic and independent shuffle-lane ancestry; 0.0793 seconds.
- `cpu-v2`: 11/11 PASS, adds an inert CLI/native-M8 route check, full saved-BF16 comparison including last-word mismatch and nonfinite rejection, and two 200-sample fake timing cases with exact AB/BA order; 0.1032 seconds.

Each attempt retains its command, runner, source snapshots, full stdout/stderr and result. Guards forbid MLX/NumPy/Torch imports, processes, network and signals. Source hashes before/after matched. These checks do not compile Metal or validate GPU numerics/speed.

Frozen candidate SHA256: `63055795d00405e7aa7fb0960ed597fc95da6fdeefd3c28187957ce2387f4a50`.
