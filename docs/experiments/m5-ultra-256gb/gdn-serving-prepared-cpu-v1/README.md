# GDN serving: CPU and preparation evidence

The original guard attempt stopped during discovery (0 tests) because it denied `tutorial_code`, including the pure standard-library scheduler. Its source, failure, stdout/stderr and snapshots are retained unchanged. The v2 guard admits only the exact namespace and five SHA-pinned CPU modules; all other device/backend, subprocess, network and signal operations remain denied. The complete three-module suite then passed **40/40**, with no errors, skips or denied operations and unchanged sources.

Actual inherited CPU preparation passed. The separate read-only controller preparation and sealed receipt reached **PREPARED**. The historical journal snapshot was saved before launch; this archive makes no claim about the current service or subsequent experiment.

- Owner plan: `c908eed8087cd72a86a4994a7fc37ffb41bfd719726ad04ff6d8b6f0768c14d2`.
- Child plan: `dfb6d3d306e6247e34dfdfe54746e12205b297e166e1f175e85024786d73163b`.
- PREPARED journal: `70a4269e63af463c721855a255c1c577b1fe8b124abba577cafcafecd08e1fcd`.
- Registered source/evidence pins: 76.
- Archive: **945 members**, **13,885,757 uncompressed bytes**, **1,580,680 compressed bytes**.
- `evidence.tar.xz` SHA256: `1e1f0622b7f300789dd7633fcfd56dc502b0261e1dfc69efdb061c6ae229798e`.

Every member was restricted to a unique regular relative path, safely extracted, then checked for exact SHA256 and size. All packed small inputs were rechecked. Both actual CPU attempts, original inherited preflight, registration, preregistered sources, retained runner versions, command receipts, and historical service preparation are included. No tests were rerun during archiving.

The [existing canary archive](../E029-gdn-prework-canary-v1/README.md) retains its 60 raw tensor files; this archive stores only its small evidence references. It does not duplicate the 176,734,788-byte tensor payload. Preparation does not qualify the planned 37-request serving workload, live HTTP, parameter replacement, E024/E028 composition, or performance. Those require subsequent closed evidence and independent audits.
