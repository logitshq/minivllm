# oMLX outer protocol CPU evidence · 2026-10-17

Actual **57/57 PASS** in 0.876801s, with zero failures, errors, skips,
expected failures or unexpected successes: controller20 + independent audits12 +
owner13 + execution-driver12. This run occurred after E024 serving had returned
RESTORED with recovery rc0; the supplied recovery receipts are retained.

The dedicated oMLX Python ran explicit fake lifecycle/offline fixtures. The only
real child was the original, exactly allowlisted isolated Python `--help` test,
at its original call site; its command, PID, rc/stdout/stderr are retained. No
MLX/vendor model module loaded, no network/service/model action occurred, and
141 source records matched before/after and their frozen copies.
This is **CPU protocol evidence**, not actual oMLX model, full-state, output-quality,
HTTP throughput/latency or restoration qualification.

[Original result](cpu-result.json) · [Summary](summary.json) ·
[Manifest](manifest.json) · [Archive verification](archive-verification.json).
`raw-evidence.tar.gz` preserves all 149 files / 2,757,420 logical bytes,
including stdout/stderr, every frozen source, original artifact manifest,
runner and recovery inputs. Every tar member was independently read back and
its full SHA256/size matched. Compressed size: 642,490 bytes.
Archive SHA256: `7624494d65b74dc5e8a8bcd75977cf4ab038f22c9ef1d488da7d3ace3c001392`.

Offline archive check (does not rerun tests): `python3 verify.py`.
This archive is source/evidence preservation, not a portable runtime bundle.
