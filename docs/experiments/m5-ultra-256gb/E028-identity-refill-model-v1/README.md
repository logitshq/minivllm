# E028 fixed full-model qualification: PASS

2026-10-17, M5 Ultra 256 GiB. The frozen Qwen3.8-27B-nvfp4 workload passed
same-schedule numerical qualification: the eager reference and the current native
stack with the default-OFF identity-refill candidate agreed on the recorded full
logits, state, raw/committed token and lifecycle checks. The independent v2 audit
passed after correcting a case-boundary bug in the original auditor. This is no
live HTTP qualification and adds no throughput–latency point.

## What actually ran

| Case | Requests | Steps | Raw = committed tokens | Eligible refills |
|---|---:|---:|---:|---:|
| D-C1 | 8 | 4,096 | 4,096 | 0 |
| D-C8 | 16 | 1,027 | 8,192 | 7 |
| Lifecycle | 13 | 40 | 123 | 5 |
| Total | 37 | 5,163 | 12,411 | 12 |

Each eligible refill compared all 64 layers / 128 arrays immediately after rebuild
and before decode: 1,536 additional array comparisons. Across the whole run the
ledger records 6,624,069 bit-equal array checks and 13,248,138 finite checks, including
10,315 full-vocabulary logits arrays. The offline auditor reconstructs scheduling,
checks coverage and validates recorded comparison results; it does not recompute
GPU arithmetic. Source/model pins were rechecked by the numerical child; the
independent owner audit rehashed runtime and source evidence, not model shards.

Normal EOS was enabled, but neither EOS ID (248044, 248046) occurred: all 37 requests
hit their caps. Actual early EOS termination is therefore **not covered**. Qualification
is limited to the fixed cases, shapes, model, source versions and same discrete
schedule. Real HTTP scheduling, HTTP token delivery and performance remain pending.

The frozen plans are [here](../../../../bench/m5_ultra_256gb/identity_refill/prereg-model-v1/README.md).
The candidate and producer were unchanged during this run. They remain default OFF.

## Original audit failure and narrow correction

The [v1 failure](independent-audit-v1.json) occurred at ledger line 5130, lifecycle
step 1. The producer reuses the C8 engine pair for lifecycle, whereas the old auditor
reset the free-slot stack to initial order. The independently reconstructed drained
C8 order was `[1, 0, 3, 2, 5, 4, 7, 6]`, so the first lifecycle slots were `[6, 7, 4]`,
not `[0, 1, 2]`. Pending retirements were already carried correctly. Only those three
slot fields differed in that first schedule; [forensics](audit-v1-forensics.json)
preserve the original failure, source locations and exact line hashes.

The separately versioned v2 copies the independently reconstructed C8 stack into
lifecycle, validating complete drain, a full unique slot permutation and pending
retirements. It never takes the expected stack from the candidate's next schedule.
C1 → C8 still starts fresh. Five numerical and six owner CPU tests passed, including
wrong-slot/state and resource/restore/source negative checks. The complete
[second independent audit](independent-audit-v2.json) then passed on the original
1,628,395,463-byte ledger. No model rerun or changed workload was used to obtain PASS;
v1 sources and failure remain intact.

## Resources and original service restoration

The user authorized cold stop and restore before execution. Stop ran at
14:19:10–14:19:16 UTC; the instrumented owner finished at 14:37:24; restoration
completed at 14:38:06. The owner took 1,087.833 seconds with two engines and correctness
observers. Its MLX peak was 74,784,272,346 bytes (69.648 GiB). Neither value measures
production latency, throughput or a single serving engine's memory.

All 212 resource samples had pressure 1; swap usage stayed at 437.06 MiB and Swapouts
at 316235. Swapins rose from 76702 to 76742, so the record does not establish an absence
of paging. Model child 68871 exited 0, was waited/reaped and confirmed absent.

The [restoration journal](restored-journal.json) is `RESTORED`. Replacement oMLX PID
74456 runs the original launchd job and loaded only `Qwen3.8-Flash-Next-oQ6e-mtp`.
Fresh log evidence confirms Lightning MTP and resident PLE; the load worker exited 0
and was reaped. The independent audit found persisted/visible restoration receipts
consistent. This meets `cold-restart-persisted-visible-v1`; it does not restore the old
PID, warm KV cache or an exact hidden runtime signature. Final HTTP status and
PID/job/listener observations are the pinned controller's receipts; the offline audit
independently checks the raw physical and log evidence.

## Complete evidence and reproduction

[summary.json](summary.json) is a compact machine-readable report. The original
[archive](raw-evidence.tar.xz) contains all 692 files from `owner/`, `journal/` and
`execution/`, including both audits, commands, stdout/stderr, CPU results, source
copies and original resource/ledger records. Its 1,633,685,854 uncompressed member
bytes compress to 2,480,860 bytes; repeated JSON comparison metadata compresses well.
Archive SHA-256: `b61917ac2bff91c97aa35460e1176382b03cbbf521f6c4428390b0a9ba8a6900`.
Every member was decompressed and rehashed against [manifest.json](manifest.json),
then every source file was rehashed again to detect changes during packaging.

From this directory, the following standard-library-only check verifies the archive
and every member without extracting files or importing MLX:

```python
import hashlib, json, tarfile
from pathlib import Path
m = json.loads(Path("manifest.json").read_text())
p = Path(m["archive"])
assert p.stat().st_size == m["archive_bytes"]
assert hashlib.sha256(p.read_bytes()).hexdigest() == m["archive_sha256"]
seen = set()
with tarfile.open(p, "r|xz") as archive:
    for member in archive:
        assert member.isfile() and member.name not in seen
        expected = m["members"][member.name]
        assert member.size == expected["bytes"]
        h = hashlib.sha256()
        with archive.extractfile(member) as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        assert h.hexdigest() == expected["sha256"]
        seen.add(member.name)
assert seen == set(m["members"])
print("PASS", len(seen), "members")
```

Original numerical/owner CLI, source hashes and absolute evidence paths are retained
in `execution/audit-v2-command.json` and the archive. Their full replay expects that
frozen local layout and runtime; this portable archive check verifies original bytes,
not another model execution. Next work is a recorded HTTP scheduling qualification
and paired C1/C8 OFF/ON timings, then same-run oMLX/Lithos concurrency sweeps before
updating the throughput versus complete-request latency frontier.
