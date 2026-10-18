# B2/B8 GDN canary preparation: 20 CPU tests passed

Both complete fake-only modules passed on their first guarded attempt on
2026-10-18. No device/model/service execution or numerical qualification occurred.

| Evidence | Tests | Elapsed | Tool output |
| --- | ---: | ---: | --- |
| `gdn-batched-canary-cpu-v1` | 12/12 | 0.031402291 s | `803c39` |
| `gdn-batched-audit-cpu-v1` | 8/8 | 0.030197709 s | `7ad3be` |

Each run has zero failures/errors/skips/blocked operations and unchanged source
hashes. The actual command was `.venv/bin/python -I -B /tmp/run-gdn-batched-{canary,audit}-cpu-v1.py`.
The archive retains both runners, complete stdout/stderr, results, artifact
manifests, and all 16/18 respective source snapshots. Guards prohibit real
subprocess/network/signals and model/backend imports. These are mechanism tests,
not a replacement for the planned real-activation screen.

The prepared screen has four reference batches (B2/B8, early/late), twelve
three-step candidate state chains and 36 actual candidate calls at GDN ordinals
0/24/47. It preserves native batched projection arithmetic and each row's actual
greedy history, stops on an early EOS instead of inventing continuation, and
saves 140 complete original-dtype tensor files in a future device run. The
independent auditor checks 108 output/state pairs, 48 continued-state pairs,
80 full-vocabulary greedy rows and 324 recorded comparison arrays. Its only raw
parser change is a 16→32 MiB single-file ceiling: B8 FP32 S alone is 25,165,824 B.
The dtype/finite/offset/SHA/byte gates remain intact. See
[CANARY.md](../../../../bench/m5_ultra_256gb/gdn_prework_batched/CANARY.md)
for scope, storage independence, parameter receipt limitations and the actual
future CLI. Full-model, HTTP, timing, and B3–7 remain unqualified.

`evidence.tar.xz` has **51 regular members**, 555,482 uncompressed bytes and
**61,680 compressed bytes**. SHA-256:
`e535a8349298b119f8e05747cc48942e9a5a3d85adaa986cc1262976e49d0640`.
The archive command and a separate verification both passed (tool `22b4a9`).
Every original member was rechecked before/after packing; `verify.py` checks the
manifest and performs bounded manual extraction without links/path traversal.

```sh
python3 docs/experiments/m5-ultra-256gb/gdn-batched-canary-prepared-cpu-v1/verify.py
```

The prior frozen B1/batched candidate and old evidence were not changed. Actual
canary registration, resource owner, device execution and independent raw audit
remain future work.
