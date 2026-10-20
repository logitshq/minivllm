# E053: complete replay of actual candidate HTTP history

**Producer replay checks PASS, independent ledger audit PASS, saved resource and
restore review PASS.** This run qualifies the exact E052 v2 candidate HTTP history:
1,030 steps,16 drained requests and8,192 raw/committed output IDs. It does not grant
arbitrary-arrival, other-workload or performance/Pareto qualification. This is the
new candidate history, unlike the original E047 history replayed by E051.

| Independently audited source-bound receipts | Count |
| --- | --- |
| Original-dtype bit-equality checks | 1,324,530 |
| Finite-value checks | 2,649,060 |
| Full-vocabulary / state / ahead arrays | 2,064 / 1,319,424 / 3,042 |
| Complete requests / raw and committed IDs | 16 / 8,192 each |
| Ahead produced / carried / consumed | 1,014 each; zero invalidated |
| Route calls | 1,028; zero eager fallback |
| Python graph tracing / same-key reuse | 8 / 1,020 |

The auditor independently rejoins original raw SSE/trace inputs, checks all1,034
ledger lines, both route streams through EOF, every recorded step and final drain.
Batch histogram is B0:2, B2:4, B4:4, B6:4, B8:1,016. Six partial retirements/refills
are retained. Both bounds1023/1536 were traced at B2/4/6/8. The48 GDN layers are
installed/restored once. Tracing and reuse counts are Python/source-bound graph
route observations, not GPU kernel/completion counters.

Successful raw tensor payloads were not saved. Independent audit validates the
complete device producer's per-array bit/finite receipts and reconstructed
IDs/schedule/routes; it does not independently rerun MLX or reread successful full
tensor bytes. The553,638,749-byte ledger SHA is
`7e73fb53b5cb5118338eee873c041a6b81e9ceef1026eacd6741c23e7d50aee7`.

The independent artifact sets recorded_history_verified,
http_history_qualified and candidate_http_capture_qualified true only for this
history. Original producer numerical/http/performance=false and
independent_audit_required=true, and owner independent_numerical_audit_pending=true,
remain unchanged. Archive integrity is separate from both numerical qualification
and resource review. No performance number is promoted and formal v16 is unchanged.

## Resources and restoration

Actual parent time518.821436334s, base replay513.334622167s and independent audit
5.953630542s include checks/IO and are not serving throughput measurements.
Independent saved resource review reparses151 physical samples and151 metadata
samples, with final-sample→ack→wait/reap binding. Max gap3.367126667s, including
baseline→first2.118149083s. Recorded maxima: system245,604,433,920B, child
current72,306,334,208B and child historical peak77,319,018,864B. MLX allocator peak
75,192,321,312B is a separate producer accounting value; these quantities are not
added together.

All observed samples meet child80GiB/system240GiB sampled thresholds, pressure1,
AC power, thermal0, zero swap growth and fixed Swapouts807478. Sampling is not a
hard allocation cap and cannot bound between-sample peaks. Saved direct PID/PPID,
wait/reap/strict absence, lease closure and original oMLX PID52818 identity/resume
and all three health endpoints pass. This review performs no new live probe and
does not claim to reparse unavailable full historical process inventories.

## Reproducibility and preserved attempts

Parent plan SHA:
`4a2d6cc768545654a7931ea4dbc3afc8837b0bdc16a159021c9ade2f0104de47`.
Child plan SHA:
`26f5d003fc61f96a22939750e7732858beeaa9313f4de40b94a0e4a0009c9431`.
Independent auditor source SHA:
`1576466a27c3e5463401ce19c14c9c4cb936a342cf920e90dc232c364699e4ec`.
Actual device tool43778d/session32502 →8ddd62 exit0; actual independent audit
6d870c/session90617 →9ca609 exit0. Commands, stdout/stderr, returncodes and source
snapshots are preserved. The audit output is an extensionless **file**, not a
directory. A read-only metadata inspection initially assumed a directory and
raised NotADirectoryError; `inspection-note.json` retains this tooling mistake.
It caused no audit rerun and did not change the original PASS artifact.

All preparation attempts remain: source CPUv1 ran10 tests with one error; v2
10/10 PASS, v3/v4 11/11 PASS; owner CPUv1/v2 13/13 PASS; route auditor7/7 PASS;
old-pin auditor4/4 PASS and final auditor11/11 PASS. Original snapshots/pending
fields remain. No failed actual numerical or archive attempt was replaced.

`evidence.tar.gz` includes967 mapped members: full ledger/recording/routes,
producer/owner/lease and raw resource observations, source pins, all CPU versions,
and actual independent audit/execution.600,247,653 original bytes compress to
41,404,859B. Every member was independently reread and matched by size/SHA to
its original and manifest. No requested directory is missing; no payload was
excluded, and no model weights/full tensor snapshots are present.

Archive SHA256:
`746cfeace401d1bad00ab257a4b22c398e0574e68acce6cf4eb2dffe1414f588`.
Manifest SHA256:
`da3dc69afd18f8345d41b91faacdfdfc3f0150adc2667b1c1939eb219304bbc6`.
Outer artifacts are bound by `document-integrity.json`, excluding itself. The
preparation drafts remain; actual archive used `archive.py`. No GPU, service action,
raw join or numerical replay was repeated by this archival step.
