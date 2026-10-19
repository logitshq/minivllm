# E042: shared-head fixed forward-prefix qualification

**Producer PASS, independent original-byte audit PASS, parent/lease closed and service restored.**
Only `forward_prefix_qualified` is true. `full_model_qualified`, compiled, rollback, HTTP and
performance flags remain false. This is not an acceptance/rejection implementation, a general
model qualification, a speculative speed result or an engine Pareto point. The candidate is
default OFF and remains outside the production engine.

| Actual coverage | Result |
| --- | --- |
| T2/4/8 at used257/1023/1024 | 9 cases PASS |
| Retained candidate prefixes | 42, including full vocabulary and all KV/GDN/conv state |
| Independent output tensor pairs | 5,418 original-byte comparisons |
| Raw snapshot files checked | 96, all local originals |
| Saved vocabulary rows | 24 reference, 42 candidate |
| Source-bound evaluated head receipts | 9 |
| Before/after parameter value receipts | 1,349 unchanged |

The producer used the actual pinned Qwen3.8-27B NVFP4 model on MLX 0.32.3 / Apple M5 Ultra.
The candidate shares the native teacher's original model parameters and preserves the fixed
B1 arithmetic path, with the new shared kernel only at the head. The shortest original D
fixture has 886 prompt IDs; used257 uses its first256 IDs as an explicit diagnostic domain.
Teacher-forced blocks, every prefix output/state, complete KV padding, canonical/input
preservation and rejection-domain inputs are checked. The independent reader also verifies
all saved raw file hashes, finite values, original dtypes/shapes, causal IDs from full saved
vocabulary and exact source/plan identity. This is an offline saved-byte comparison, not a
second MLX computation. Pre-anchor history, head dispatch and live storage identities remain
source-bound producer evidence. Parameter payloads are not archived or independently reread
by this audit; their retained original-byte receipts match before/after.

Producer elapsed time was 58.171206 s; independent auditor elapsed
40.454174 s. These include qualification/storage work and are not speed
measurements. The parent, numerical child and independent auditor each completed with exit 0.
Actual lease recorded original oMLX PID52818, the same start identity, T→S resume and healthy
original-model before/after receipts. The child PID/PPID and direct wait record agree. The
saved source-bound parent also verifies child absence after wait; this archive performs no
present-time process query.

There are47 child physical samples plus the parent's admission sample. Saved parsed values:
child current max19,424,916,320B, recorded historical peak19,634,188,624B, system max
192,880,607,232B. Pressure is1 throughout; Swapouts stays807478. Swap bytes range
2,372,340,285–2,380,728,893, never above the admission value2,380,728,893. The archive rechecks
these saved parsed values, sample gaps and the registered observation gates; it does not
independently remeasure physical memory or turn observation limits into hard allocation caps.
Full raw monitor outputs remain included for inspection.

## Exact closure and preparation history

The actual child plan is `f6aeddb2af9e0958e16b45c03c1360cd96f5662ad65c43700809382f2ff3df92`.
The actual parent v2 plan is `55dbd68bb64a80b3091e629a20b6915a19d3ab92fb92c0b29417700aaefa6b7a`.
The shared head source is `d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e`.
The independent auditor source is `cbae7b60f1aa10c2be6965d5ab4cc8bdc104edf61822802c889e01a2969299af`.
All executed source maps and actual commands are retained. Parent v1 was prepared on CPU but
never executed; child-reap and environment review repairs are preserved alongside executed
v2. The model and numerical child plan were frozen. The 40-test CPU preparation and source
snapshots are retained as preparation evidence, separate from this device qualification.
Historical preparation files still say pending/not run; those statements describe their
original preparation snapshot, and the saved execution/audit records provide the later outcome.

## Metadata archive and local raw payloads

`evidence.tar.xz` has 474 explicitly mapped members: producer complete/events and every
raw sidecar, parameter value receipts, fixtures/plans, actual owner/lease/auditor execution,
all monitor observations, original preparation/CPU evidence, frozen algorithm and relevant
runtime text sources. Every archived member was reread and matched to both its original and
`manifest.json`: 21,200,060 metadata bytes → 573,008 archive bytes.
Archive SHA256: `1f1cc198e8632886c8f97a17295bae73fd06900ca69da3381f6c4705d535efc9`.

**The96 large raw safetensors files remain local and were neither copied into Git nor reread
by this metadata archiver.** `raw-file-references.json` lists each absolute path, current file
size, sidecar SHA and original full-file SHA validated by the completed independent auditor.
Their exact total size is 24,476,249,870B (about22.80GiB). These references preserve
the already audited raw identity; metadata archive integrity does not itself repeat that raw
qualification. Full offline reproduction requires retaining these local96 files and the
frozen source/plan paths, not merely extracting this metadata archive. Model weights and
runtime binaries are also excluded; their source-bound references remain.

First metadata archival attempt PASS; no failed E042 archive attempt was overwritten.
`references.json` separates coordinator-reported tool/session IDs from retained process
receipts. `archive-command.json`, stdout/stderr and `archive-execution.json` retain this actual
metadata-only command. `document-integrity.json` binds all outer artifacts except itself.
No GPU, service action, new numerical test or commit was performed by this archive step.

[E040 local speed screen](../E040-shared-projection-screen-v1/README.md) motivates only the
head T4/T8 direction. [Formal v16](../../../optimization-guide/progress/2026-10-19-v16/README.md)
remains unchanged. End-to-end verifier cost and acceptance/rollback/HTTP qualification are
still required before any engine performance conclusion.
