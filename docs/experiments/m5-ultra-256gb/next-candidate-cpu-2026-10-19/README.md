# Candidate CPU checks — 2026-10-19

Run after actual E033 formal and restoration closure. All attempts are retained.

- `next-candidate-cpu-v1`: FAIL before tests; runner imported a relative-import test under a non-package name.
- `next-candidate-cpu-v2`: actual 155 tests; 154 pass, one GLM registration test errors because the correct rejection raises Refuse rather than the expected ValueError. No forbidden operation or source drift. The focused repair has its own subsequent history.
- `glm-text-facade-cpu-v1`: actual 11/11 PASS, no skips, forbidden operation or source drift.

Runners reject backend/model imports, process launch, network connect/bind and signals. Asyncio SIGINT handler mutations are explicitly suppressed. Small NumPy fake tests are allowed. This is CPU control/mechanism evidence only; no new numerical device or performance qualification.

The archive retains exact source snapshots, test output and manifests. `verification.json` records independently reread member hashes. Fixing later tests or sealing GLM sources does not rewrite these originals.
