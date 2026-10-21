**Independent Audit Result**

- **Command executed**: exact specified invocation; exit code `1`.
- **Audit status**: `FAIL` — `ValueError: malformed tensor metadata for '__metadata__'` raised in `audit.py:173` while parsing safetensors; traceback preserved in `device-audit-stderr.txt`.
- **Pairs/arrays/bytes checked**: not reported (audit aborted before completion; stdout empty, no `device-audit.json` produced).
- **Provenance recorded**: `device-audit-provenance.json`
  - `audit.py` sha256: `82525dc74426384317cc51953c3cab750f97492e86df37fdfaecea270981620b`
  - `pairs.json` sha256: `003cb98b2bf042ce2d097f6b044b696433dc2f9e684a902847145829e86c91b8`
  - `complete.json` sha256: `86e5346a5f923065936c58e7ed356eff8aa6388c2531a591d883b3fb3b1320b8`
- **Producer self-reports (not independently verified due to audit failure)**:
  - `complete.json`: `PASS_FEATURES_LOCAL_COST`, 24 completed pairs
  - execution `final.json`: `PASS_CAPTURE_AND_RESTORE`
  - lease `resume.json`: `identity_match` true, `state_after` = `S` (not `T`)
- Actions: failure preserved; no checks weakened, no raw files edited, no GPU/network touched. Per instructions, stopping here — integrated audit did not pass, so saved evidence cannot be considered verified.
