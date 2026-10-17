# E027: M5 shared-weight M1 local numerical gate

Preregistered at `a33249d`, then executed once on Apple M5 Ultra / MLX0.32.3.
33 local cases and 66 complete BF16 comparisons passed. This is a matrix-level
numerical result, with no whole-model or throughput/latency qualification.
The current Pareto curve is unchanged.

- 27 synthetic points: three (N,K) shapes × three banks × rows2/4/5.
- Six real layer-0 projection points with synthetic activations; no actual
  model-activation or other-six-geometries claim.
- 110 original GPU shape/stride descriptors; all X/W/scales read-only checks.
- Full native/control/shared outputs preserve signed-zero bit equality.
- MLX peak30,097,544B, cache0, final active0; largest physical peak413,172,552B.
- Six external resource samples: existing pressure2 stable, swap/Swapouts
  unchanged; ≥62.522GiB real free. Numerical-only admission policy preserved.
- Owner45262 and child45274 completed; child exit0/wait/reap/strict absence.
  Same original service74384 identity restored T→S by the authorized outer lease.

`raw-evidence.tar.xz` contains owner, child, lease, inputs and root raw-audit
records: 1,173 members /271,633,022B logical payload, compressed to58,059,176B.
Identical byte payloads use tar hardlinks; every original path remains present.
Every member was extracted into memory again and its full SHA/size checked.
[Manifest](manifest.json), [root full raw comparison](root-raw-audit.json),
[owner summary](owner-summary.json). The root raw review covers319 distinct
binary files /260,130,624B and66 output comparisons /1,945,856B.

Archive SHA256: `e7a0ef8d84b3c9913ea7c40dece4ba3f409053980aa1ebe8952b74ac7eb57826`.
Plan SHA256: `840b1eb0a4e6127bf6fb5e322fdca82d34fabcd7f92712b429b25e596f0f2351`.
Original owner result deliberately retains `raw_audit:PENDING`; a later audit
is a separate receipt, never an edit to the executed record.

[Tutorial](../../../optimization-guide/tutorials/027-shared-weight-m1.md) ·
[Candidate and original preregistration](../../../../bench/m5_ultra_256gb/shared_w_m1/README.md).
The archive is source/evidence preservation, not a portable model/runtime bundle.

Independent audit: **PASS**, [full receipt](independent-audit.json), SHA256
`f0f3bb2c5f04110e4405ee510474d8acda89166078aef472bb3f32a7a6abe421`.
All576 source/runtime/bundle pins,33cases/66comparisons,110descriptors,68self
physical phases and service lifecycle checked. The audit is numerical-only.
