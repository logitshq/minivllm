# E025 same-split NAX v2: partition gate rejected

The preregistered M5 Ultra screen stopped at its third comparison. **Two checks
passed; NAX partial equality failed with 11 / 233,472 differing BF16 elements.**
The independent failure audit passed and root reproduced the same audit receipt.
Audit PASS supports the recorded rejection; it does not qualify this candidate.

| Observed gdn_z T19 check | Shape | Result |
|---|---|---|
| Default OFF versus native QMM | 19 × 6144 | Exact bytes; all finite |
| Reconstructed SIMD reduction versus native QMM | 19 × 6144 | Exact bytes; all finite |
| NAX versus SIMD partials | 2 × 19 × 6144 | FAIL: 11 elements / 11 bytes differ |

Both custom kernels compiled and materialized outputs. The reference reduction
matched the actual native QMM before the NAX comparison was evaluated. The first
difference is byte 4,700, coordinate `[0, 0, 2350]`: SIMD `0x3d29`
(0.041259765625), NAX `0x3d2a` (0.04150390625). The largest absolute partial
difference is 0.125; six differences lie in partition 0 and five in partition 1.
[Root's complete raw review](root-raw-review.json) and the
[independent audit](independent-audit.json) retain every differing coordinate.
X/W/scales are complete before/after snapshots and match byte for byte; X and
all compared output elements are finite.

**NAX final reduction, enabled callable, six fallback lengths, mlp_down and out
were not run.** The planned matrix had 33 comparisons, 68 phases and six custom
kernels. The actual prefix contains three comparisons, six self-resource phases
and two kernels. A failed partial gate does not establish a difference in an
uncomputed final output or model logits. No full-model or performance
qualification resulted, and there is no new Pareto point.

The preregistration commit is `4d97a8d02b2d20022ee314ddb9406b486e3d7b62`,
created 2026-10-17 12:29:38 UTC before preflight at 12:30:12 UTC. Plan SHA256:
`3beae0672fd90d6a83a3869f66164273229751151d99868259dcb3f1c4b16a12`.
The frozen screen, candidate, generated-source identities and 570 parent
before/after pin records are preserved. The auditor independently rehashed 538
frozen source copies and 32 uncopied runtime files, and rebuilt both Metal text
receipts from pinned frozen headers. The child terminal source recheck was not
reached; parent rechecks are separate evidence.

The last self-resource phase is before the failing comparison. The recorded
MLX historical peak is 45,219,886 B; the largest physical peak observation is
335,659,752 B. Cache stayed zero. These retained observations fit the registered
1 GiB MLX / 2 GiB physical diagnostic budgets, but do not prove the peak after
the failing evaluation or a missing final release. Three external resource samples
show pressure 2 unchanged, swap use 445.06 MiB unchanged, Swapouts 316,235
unchanged and at least 67,817,275,392 B real free. This separately admitted small
numerical screen supplies no performance measurement or relaxed full-model gate.

Owner 55019 directly owned child 55031. The child exited 1, was waited/reaped and
passed strict PID absence. The outer lease restored the same original service
74384 from T to S. [Owner summary](owner-summary.json) and
[child summary](child-summary.json) retain their original FAIL state. Their raw
`PENDING` audit marker is not rewritten by later audit work.

`raw-evidence.tar.xz` contains all actual owner, child, lease, input-plan, CPU
preparation, root raw-review and independent/root-repeated audit files. It also
contains the independent auditor, its four CPU negative tests, and the archive
script. No unrun-case artifacts are invented. The package preserves sources and
recorded evidence; it is not a complete portable runtime or model bundle.

The [root-repeated audit](root-repeated-audit.json) is byte-identical to the
independent audit. [CPU negative challenges](cpu-challenges.json) include signed
zero, equal NaN bits, tensor layout corruption and invalid physical readings.
Identical payloads are stored as tar hardlinks while retaining every original
path. Every member was reread and its full size/SHA256 verified; root independently
repeated the complete archive verification.

Archive: **20,198,408 B**, **1,137 members**, **61,719,852 logical raw bytes**;
31,829,025 unique payload bytes. [Manifest](manifest.json).

Archive SHA256:
`99b091b4f6a99cb2b56c98d092531f24127134498b31d609fc48fbe919294a5a`.

The [archive helper](archive-evidence.py) rebuilds into a new `--out` directory
from retained `.artifacts` inputs and refuses to overwrite an existing archive.
Keep this candidate default OFF. The
[tutorial](../../../optimization-guide/tutorials/025-same-split-nax.md)
records the hypothesis, exact attempt, rejection and remaining correctness gates.
The [qualified throughput–latency plot](../../../optimization-guide/progress/2026-10-17-v5/README.md)
remains unchanged.
