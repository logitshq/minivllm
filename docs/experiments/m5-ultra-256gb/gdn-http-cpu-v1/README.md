# GDN HTTP: actual 16-test CPU evidence

The complete frozen `gdn_prework_http.test_http_cpu` module passed **16/16** once,
with zero failures, errors, skips, expected failures, unexpected successes or
forbidden operations. All **216 source/metadata snapshots**
matched before and after execution. The runner and queue terminal receipts,
full stdout/stderr, per-test/subtest outcomes and complete source copies are retained.
No tests were rerun by this archiver.

The queue waited for the prior E024 route formal cycle to close, independently
restore and archive. That cycle's recorded status was
`EXPERIMENT_FAILED_EVIDENCE_RETAINED`, with its formal audit `FAIL` and restoration
`PASS`; this CPU success neither changes nor repairs that historical result.
The guard blocked device/backend imports, model payload access, real subprocess,
network and signal delivery. Local asyncio socketpair/handler registration was
allowed, as recorded by the runner. These are fake CPU contracts, **not a model,
real HTTP request, numerical, full-state or performance qualification**.

- Archive: 239 members; 3,943,046 uncompressed bytes;
  577,532 compressed bytes.
- `evidence.tar.xz` SHA256: `d07bee95951d29e14d1e7b8c27b3f933be8a074d38910cc61023ba00244b9e55`.
- Frozen runner SHA256: `22e9ef7a46e1a2ea76940abc8e5152088e124d5f060f02c2ee63c1eeb51b35cc`.
- Actual result SHA256: `9e6fcada30534ea2d50451d9a1dd1f6695aecf64019971430615c264f73be800`.

[Manifest](manifest.json) binds every member to its original source path, size and
SHA256. [Verification](archive-verification.json) records safe extraction,
all-member checking and unchanged original trees. Run `python -B verify.py` from
any directory to independently recheck the compressed archive and sidecars;
it uses only the standard library and performs no service/device actions.
