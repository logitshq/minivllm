# E023 service-absent v1: fixed plan and read-only prepare

The new wrapper pins the unchanged 37-request numerical plan and the separately
reviewed service-absent owner/controller. Actual v2 `prepare` passed on
2026-10-17, observing the original service identity and exact loaded model set.
The saved journal is PREPARED, no owner directory exists, and no stop, bootstrap,
load POST or model/GPU work occurred. The `approved_contract` plan string names
the required future contract; it does not record user approval.

Plan SHA: `b38ae764738786734d173d94741d867b942516cad84fe287c5768b042a9391b2`.
Original numerical plan SHA:
`21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb`.

See [receipt](receipt.json), [exact wrapper](plan.json), and the historical
[read-only journal snapshot](read-only-prepared-journal.json). The live journal
path in the wrapper is authoritative for subsequent state; do not edit the
snapshot to simulate a completed phase. Fresh source/identity checks remain
mandatory before any authorized mutation.

The independent service-absent auditor is now frozen at SHA
`1b7b526dac98c43f5b77f915aec3173d755d1efb28ed49b989e7601293ddaf6d`.
Root reran 25 CPU tests, including producer-layout compatibility; independent
review repaired and rejected a stale-resource-time replay. The actual current
PREPARED journal correctly returns INCOMPLETE without opening a ledger.
[Exact test and rejection evidence](auditor-review/receipt.json) is retained.
These are preparation checks; no complete model workload or real recovery
has run. Final HTTP/PID observations remain pinned-controller receipts, while
physical snapshots and log bytes are independently checked.
Current pressure 2 does not satisfy the full-model/performance pressure-1 gate.
Explicit cold-restart approval remains pending; temporary suspend/resume
permission does not authorize this cold restart. Unknown spawn/forced-owner
termination outcomes retain the manual-reconciliation boundary described in
[controller v2](../../omlx_cold_restart/README-v2.md).
