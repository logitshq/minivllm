# GLMv3 actual E033-bound preregistration

Status: **PREPARED; no GLM owner/model run**. The actual E033 predecessor is
RESTORED with independent restoration PASS. CPU registration, actual read-only
controller preparation and receipt sealing returned zero. The source chain and
61 receipt pins were checked again while this archive was written.

The broad CPU attempt previously found one test exception-class mismatch: the
controller correctly raised Refuse, but the test expected ValueError. Only tests
were repaired, with an explicit synthetic UNSEALED pin so later sealed checkouts
still exercise rejection. Both focused 16-test runs passed under backend/process/
network/signal guards; postseal also validated the real source/predecessor chain.
No GPU or performance result is claimed.

`summary.json` records exact future execution argv, original service identity,
plan/receipt hashes and output paths. `commands.json` is the unchanged registered
controller/owner/auditor argv. `prepared-journal.json` is a byte-for-byte PREPARED
snapshot. `evidence.tar.xz` retains the complete focused CPU source snapshots,
stdout/stderr/results, before/after seals, actual registration and controller
receipts. `manifest.json` maps every member to its original source/hash/size;
`archive-verification.json` confirms all copied members and original bytes.

Root commits and pushes this archive and sealed sources before executing the
actual driver, which always attempts finally recovery. Closure uses the saved
`restoration_audit` and `closed_audit` commands. A diagnostic failure is preserved;
restoration success alone cannot qualify the model or performance.
