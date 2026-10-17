# E026 real-weight layer v1: aborted before the first case

Preregistered in `387412e`; plan SHA `f01fa0ac7ce85975a92ef07188417dde096c8f4b37abcea9467408f516e3ce08`. This attempt is not numerically or performance qualified. OFF completed zero cases; neither ON arm started. No point is added to the throughput–latency Pareto chart.

The first external footprint probe reported top-level current **65,389,240 B**, auxiliary current **65,323,704 B**, and auxiliary historical peak **65,323,704 B**. All are below the unchanged 2 GiB budget. The old parser required top-level and auxiliary current to be equal; the 65,536 B difference triggered its guard. The origin of the difference is not established. It is an observation-contract failure, not evidence of excessive model memory or incorrect model output.

The owner ended after 0.793 s and its child after 0.124 s, interrupted while loading selected tensors before any case. The child's generic `TimeoutError` text also handles SIGTERM, so it does not establish a 120-second timeout. A separate v2 registration will preserve every raw field, bound each independently, and distinguish cancellation from a deadline; it must not modify this attempt or raise the budget.

Four raw system snapshots retained pressure 2, swap used 445.06 MiB and unchanged Swapouts. The failed physical probe prevented the running snapshot's derived free-pages field, but raw VM data is preserved. Child PID39604 was waited/reaped with strict absence; owner PID39583 was waited by the lease. The same original PID74384 service returned T→S. An additional read-only probe confirms both experiment PIDs absent and the original service identity running.

The unchanged preregistered auditor and a root rerun both return FAIL, qualified=false. An independent failure summary describes partial evidence without granting qualification. [Independent summary](independent-failure-summary.json) · [Root audit](root-audit.json) · [Final process proof](final-readonly.json).

`raw-evidence.tar.gz` retains the complete owner directory including frozen source copies, all raw resource probes, partial child output, exact plans, lease and audit receipts. `manifest.json` lists every member's SHA and size; root reread every archive member after packing. Large external runtime binaries and original model weights remain referenced by their frozen hashes, not duplicated in the archive.
