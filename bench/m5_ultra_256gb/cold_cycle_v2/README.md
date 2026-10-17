# Cold single-child lifecycle v2 — prepared, not executed

2026-10-17. Frozen v1 files remain unchanged. Controller, owner and auditor are copied from the registered v1 sources; only the lifecycle schema and predecessor admission change. The old source identities remain explicit. New imports do not operate services.

This version can enter from independently restored v1 single-child or Lithos D512 formal cycles, in addition to the original pinned v7 migration. Journal/audit files still require exact immutable plan hashes, reviewed controller/auditor identities, RESTORED, strict reap, successful monitored HTTP load, preserved visible configuration and no unresolved cleanup. A numerical PASS is neither required nor sufficient. It does not rewrite old journals or impersonate the old controller.

The OS/resource/stop/load/reap/recover methods and direct-child owner are unchanged, with the same 1830-second work, at most 1800-second child, 15-second TERM and 15-second KILL bounds. CPU tests were copied for later execution; new predecessor-positive/negative tests and root AST-diff review are still required. No preparation, stop, recovery or model execution has occurred under v2.
