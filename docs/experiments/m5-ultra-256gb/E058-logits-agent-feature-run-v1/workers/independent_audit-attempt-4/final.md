Done. The concrete compatibility failure is fixed in my own auditor only, and the repaired v2 audit on the immutable real evidence now passes.

**Root Cause**
- Actual header of `raw/T4-k1-selected-reference.safetensors`: 131 top-level entries = 130 tensors + `"__metadata__": null` (verified directly from the file).
- MLX `mx.save_safetensors` emits that reserved key; the frozen v1 auditor treated every top-level entry as a tensor and died with `ValueError: malformed tensor metadata for '__metadata__'`.

**Fix** (only `audit.py`, frozen v1 copy untouched)
- `parse_safetensors` (`audit.py:170`): `__metadata__` is skipped from tensor enumeration when absent, `null`, or a dict of string keys → string values; any other type or non-string value raises `ValueError`.
- Duplicate-key detection via `object_pairs_hook` still applies at all JSON depths (including a duplicated `__metadata__`).
- 130-array contract, dtype/shape, finite-byte scan, per-array byte equality (signed-zero fails), SHA256 and path-safety rules are all unchanged.

**Tests** (`test_audit.py`, repo venv `/Users/ryanli/code/minivllm/.venv/bin/python -B -m unittest test_audit -v`, exit 0)
- Accepted: absent / null / empty map / string map (parser-level, plus end-to-end audit PASS for null+empty+map candidate files).
- Rejected: metadata as bool/int/float/str/list/tuple; non-string values (`{'k': 1}`, None, True, list, dict, mixed).
- Cannot hide: extra tensor alongside `__metadata__` still surfaces; 129 tensors + `__metadata__` still fails the "expected 130" contract; duplicated `__metadata__` key rejected.
- 40 tests, all OK, output preserved in `/private/tmp/e058-test-v2.log`.

**v2 Real-File Audit** (subprocess, stdout/stderr/exit captured)
- `device-audit-v2.json`: **status PASS**, 24 pairs, 3120 arrays (24×130), 12,285,255,872 bytes byte-identical reference↔candidate, all SHA256-verified, all finite.
- `device-audit-v2-provenance.json`: actual subprocess `exit_code: 0`, captured stdout (5534 B) / stderr (0 B) file refs, SHA256 of repaired `audit.py` (`ce927305…44b`), `pairs.json` (`003cb98b…91b8`), `complete.json` (`86e5346a…1320b8`), plus verification fields read from `complete.json`.
- Old `device-audit-provenance.json` (v1 FAIL, exit 1) and its `device-audit-stderr.txt` first-failure record are intact. Raw files and frozen v1 source unmodified. No GPU/MLX/network used.
