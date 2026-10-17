# E028 v3: complete two-layer synthetic state qualification

**PASS for the fixed synthetic state matrix only.** The default-OFF candidate
completed all ten cases and all 88 comparisons for one attention cache plus
one GDN cache. This is not a 64-layer model, forward/logit qualification, HTTP
run, throughput/latency result, or adoption decision. It contributes no point
to the performance Pareto frontier.

The independent stdlib verifier reads every saved original-dtype byte:
**108 safetensors paths, 880 tensor records, 5,312,069,632 payload bytes**.
It checks BF16/F32 finite exponent bits, dtype, shape, full tensor/file hashes,
IDs/lens and exact comparison bindings. All **88 comparison groups / 724 tensor
pairs / 4,334,575,616 compared payload bytes** pass.

The verifier also reconstructs the expected data independently of both arms:
all deterministic source patterns, the offset/strided source views via their
original parent indices, keep/reorder/remove row selection, per-head KV
truncation/zero padding and full GDN concatenation. This preserves and checks
unused old KV padding. It validates all 24 deliberate first-scalar writes
(-8 before delayed evaluation, -16 to sources, -32 to outputs), with every
other byte unchanged. The separately evaluated, unwritten native oracle's
entire saved input file matches the initial sources and its output matches
the independently reconstructed output. No measured implementation is imported
by the verifier.

The fixed cases cover B2→4, B4→6, B6→8, offset/noncontiguous source expressions,
reorder, removal, no new rows, L1536→2048 growth, empty old batch, and delayed
evaluation. ON takes identity-refill in exactly five cases; five controls use
the original fallback. OFF always uses the original fallback. The saved
`get_array_buffer_size` receipts show separate OFF/ON backing allocations;
the auditor verifies their recorded sums and source binding. Packed saved
tensors prove values, not the original physical GPU stride or allocation
layout.

All **53 self-resource phases**, **14 owner resource samples** and **11 owner
physical probes** pass their unchanged contracts. MLX peak is **614,891,549
bytes**, maximum active is **398,753,792 bytes**, and final active/cache are
zero. Physical peak is **1,623,885,336 bytes**, below 2 GiB. Pressure remains 2,
Swapouts 316235 and swap used 445.06 MiB stay unchanged, and real free pages
remain above 50 GiB. These are observed resource limits, not OS allocation
caps; pressure 2 is allowed only by this bounded numerical diagnostic scope.
The 11.764-second child duration includes setup, observers and raw snapshots
and is not a performance measurement or comparison.

Child **59771** exited zero and has wait/reap plus strict PID-absence evidence.
The lease waited for owner **59759** to exit zero and records the original
service **74384**, identity `Fri Oct 16 08:01:58 2026 / omlx-server`, restoring
T→S. Restoration is verified from pinned lease source and saved receipts;
the offline audit makes no new live-service claim.

The archive preserves every original owner, service-lease, inputs and
CPU-prepare file. All **576** pre/post pins and **1,090** archived source copies
were verified. The initial audit additionally rehashed all 576 original
runtime/source paths (259,583,583 bytes); installed runtime binaries are not
duplicated here. The candidate remains identical to v1/v2. V3 materializes
each synthetic parent during input construction to bound setup liveness;
the candidate, fixed cases, intended input bytes and limits remain unchanged.
The sibling `E028-identity-refill-v3-cross-version` audit independently compares
retained v2 inputs and rebuilt outputs and is separate from this archive.

Important identities:

- Plan: `fbcf114be8507484d30b2ceb367e5864f2a2928c475423d3a30192c23cfc72a7`
- Candidate: `bb9f8a461625e97c80e75140c9d9b7b8718a633595ab4f3a422267e752ded8f5`
- Screen v3: `4deeff89a314e50870f25c5bb4a94ac3d6ccd798edb44a982834e4e1ec9068cc`
- Lease source: `76108bc98300183658a2ad71cb542c0dc8e061253c5120daf86074e43e0b248f`

## Offline replay

Git stores the compressed evidence instead of duplicating the multi-gigabyte
`raw/` directory. On a fresh checkout, first verify and safely extract only
the raw archive members. All tar members are relative regular files: absolute
paths, `..`, symbolic links and hard links are rejected. Use Python 3.13 from
the repository environment:

```sh
.venv/bin/python -I -B - <<'PY'
import hashlib, json, tarfile
from pathlib import Path, PurePosixPath
a = Path('docs/experiments/m5-ultra-256gb/E028-identity-refill-v3')
receipt = json.loads((a/'tar.verify.json').read_text())
h = hashlib.sha256()
with (a/'evidence.tar.gz').open('rb') as f:
    for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
assert h.hexdigest() == receipt['tar_sha256']
manifest = json.loads((a/'manifest.json').read_text())['files']
assert not (a/'raw').exists(), 'Use a fresh extraction directory'
with tarfile.open(a/'evidence.tar.gz', 'r:gz') as t:
    members = t.getmembers()
    assert len({m.name for m in members}) == len(members)
    assert {m.name for m in members} == set(manifest) | {'manifest.json'}
    for m in members:
        p = PurePosixPath(m.name)
        assert m.isfile() and not p.is_absolute() and '..' not in p.parts
        stream = t.extractfile(m); h = hashlib.sha256(); size = 0
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block); size += len(block)
        expected = manifest.get(m.name)
        if expected: assert expected == {'bytes': size, 'sha256': h.hexdigest()}
        else: assert h.hexdigest() == receipt['manifest_sha256']
    t.extractall(a, members=[m for m in members if m.name.startswith('raw/')], filter='data')
PY
.venv/bin/python -I -B docs/experiments/m5-ultra-256gb/E028-identity-refill-v3/verify.py
```

Add `--runtime-pins` only if the original installed paths remain unchanged.
`independent-audit.json` records the narrow numerical qualification separately
from `full_model_qualified=false` and `performance_qualified=false`. The CPU
selftest receipt covers finite-bit edge cases, independently enumerated small
parent views and first-difference reporting; these are auditor tests.

`manifest.json` lists every archive input's size and SHA-256. `evidence.tar.gz`
contains those inputs plus the manifest; `tar.verify.json` records full member
verification and the tar digest. The tar and its receipt are outside their
own manifest. Original files and extracted raw copies remain untouched locally.
