# E028 v2: partial numerical success, resource-aborted qualification

**RESOURCE_ABORT_INCOMPLETE_NOT_QUALIFIED.** The full 10-case / 88-comparison /
53-phase gate did not finish. The unchanged default-OFF identity-refill
candidate actually ran in this attempt, but this result grants no full-matrix,
full-model, HTTP or performance qualification.

The contiguous B2→4 case completed its eight checks. B4→6 completed only the
three initial-input / rebuild-read-only checks before the resource guard
stopped the process. Its OFF/ON branches both constructed a rebuild graph, but
no B4 output was saved or compared. Reorder, removal, no-new, length growth,
empty-old, noncontiguous and delayed-evaluation controls were not reached.
The reverse output-write checks were also not reached.

At B4→6 `after_rebuild_graph`, MLX's cumulative peak was **1,131,872,976 bytes**,
above the unchanged **1,073,741,824-byte** observation limit. Active allocation
was 105,611,296 bytes and cache was zero. Process physical peak was
1,109,951,736 bytes, below its separate 2 GiB limit. This phase identifies when
the excess was observed; it does not isolate which setup/rebuild operation
caused the historical allocation peak. These are observed budgets, not OS
allocator caps, and the independent auditor preserves the resource failure.

The independent stdlib audit reads all **16 safetensors paths / 160 tensor
records / 671,563,776 payload bytes**. It reparses original BF16/F32 words and
checks finite exponent bits, exact shape/dtype, full payload hashes, original
source patterns and exact first-scalar source writes. Both -16 source-write
checks preserve every other byte. It also reconstructs the B2 output directly
from original source bytes, including per-head KV truncation/zero padding and
old padding, without importing the candidate or measured code. All **11 saved
comparison groups / 108 tensor pairs / 441,376,768 compared payload bytes**
pass. There is no saved first numerical difference.

There are eight self-resource phases: seven PASS and the final resource FAIL.
Four owner resource samples retain pressure 2, Swapouts 316235 and swap used
445.06 MiB, with at least 50 GiB real free pages. Two owner physical probes are
valid and independently reparsed. Child **58999 exited 1**, was waited/reaped,
and has a strict-absence receipt. The outer lease waited for owner **58987**
and records the original service **74384**, identity
`Fri Oct 16 08:01:58 2026 / omlx-server`, returning **T→S**. Restoration is
verified from pinned source and saved receipts; no new live service probe is
claimed. Wall durations are not speed measurements.

`raw/` contains every original owner, service-lease, inputs and CPU-prepare
file, including partial/error snapshots, failed-phase raw physical data and
all frozen sources. Copies were checked against the originals. All 573
pre/post pins and 1,084 archived source copies were verified. The initial
independent receipt also rehashed all 573 original installed paths
(259,538,832 bytes); runtime binaries are not duplicated in the archive.
Backing-disjoint values are receipts from pinned `get_array_buffer_size`
calls. Saved tensor bytes do not independently prove original GPU backing or
strides. No failed or unexecuted case is promoted into a numerical pass.

Important identities:

- Plan: `26ebdcf2ee6fb09b0c86a15b3bb87a7d221ba67811a4ccf382ae150d3b495b42`
- Candidate: `bb9f8a461625e97c80e75140c9d9b7b8718a633595ab4f3a422267e752ded8f5`
- Screen v2: `2a2fab6b04a6e781f378ac8dc677531c445e5a8cd7bff8e0907daeaed522514c`
- Lease source: `76108bc98300183658a2ad71cb542c0dc8e061253c5120daf86074e43e0b248f`

Reproduce the offline verification:

```sh
python3 -I -B docs/experiments/m5-ultra-256gb/E028-identity-refill-v2/verify.py
```

Add `--runtime-pins` only while the original installed paths are unchanged.
`independent-audit.json` records evidence verification separately from the
unqualified experiment outcome. A small CPU receipt also checks signed zero,
largest finite values, signed infinities/NaNs and first-difference reporting;
these are auditor tests, not device tests.

`manifest.json` records every archive input's bytes and SHA-256;
`evidence.tar.gz` contains those inputs and the manifest. `tar.verify.json`
records a full member-byte verification and the tar digest. The tar and its
verification receipt are outside their own manifest.
