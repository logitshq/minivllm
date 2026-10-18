# Combined GDN + route: HTTP CPU preparation

The current suites passed **14 HTTP entry checks + 9 independent HTTP audit checks + 14 lifecycle checks + 13 cycle audit checks = 50 CPU checks**. These are fake control-flow and source-binding checks, with no model numerical, HTTP-history or performance qualification.

The first HTTP-entry attempt is preserved in full: **10 passed and 4 failed** because its guard rejected asyncio's local socketpair creation (`socket.__new__`). The separate v2 runner permitted only the internal AF_UNIX socketpairs while still forbidding actual network connections, process launches, signals and backend imports. Its full14 checks passed; production sources did not change between those attempts. This archive retains both runners, stdout/stderr, exact source snapshots and original result bytes. The other three suites passed their first attempts.

The frozen producer uses real C1×8 and C8×16 HTTP histories, explicit route sidecars, then separately owned numerical replay processes. The cycle auditor consumes two explicitly SHA-bound independent numerical audits, independently rechecks their terminal/input/source/PID bindings, and directly reuses the frozen resource and preparation algorithms. Its PASS requires the real four-phase lifecycle and independent restoration. No one-call-per-step assumption or offline tensor-arithmetic claim is introduced.

`raw.tar.xz` contains the complete five closed CPU trees and the15 explicit new HTTP/lifecycle source, baseline and driver files listed by `manifest.json`. It excludes the active owner/journal/execution, model weights, runtime binaries and the still-developing formal implementation. The baseline is retained provenance, not permission to rerun an already-started cycle. Each regular member was compared by size/SHA-256 against its original and the original inventory was rechecked.

Verify without extracting or reading current originals:

```sh
python3 archive.py --verify .
```

Only exact manifest members under `raw/` and `source/` are accepted; symlinks, absolute paths, traversal and duplicate members are rejected. The verifier never extracts the tar. Optional `--originals` rechecks the original closed trees and current source files; a later deliberate source change can invalidate that optional check without changing archived bytes. Small summary/manifest/verification files remain outside the tar.
