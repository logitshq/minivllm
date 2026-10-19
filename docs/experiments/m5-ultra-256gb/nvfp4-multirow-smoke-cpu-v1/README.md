# NVFP4 smoke entry — one guarded CPU check

Actual first run: **PASS 1/1**, zero failures, errors, skips or forbidden
operations. Six tracked sources remained unchanged. Tool `5d14cc` exited 0;
harness elapsed time was 0.00913716695504263 s, a CPU test duration only.

The test checks default-inert entry and rejected arguments. Its guard prohibits
MLX/model imports, mmap, subprocess, process/network operations, signal changes
and timers. No actual Metal computation, model load or service action ran.
Independent source review found no blocker for the stated 36-case standalone
numerical smoke; installed MLX API signatures were inspected, not executed.

`actual/` copies the complete original receipt tree, command, result, stdout/
stderr, before/after SHA maps and six source snapshots. Every copied member and
original file was independently SHA-checked. No failed CPU attempt preceded this
run. The archived DEVICE_SMOKE.md still says the CPU test was unrun; it is the
unaltered test-time document, while the live document records the later PASS.

**The actual 36-case GPU smoke is still unrun.** This CPU PASS does not establish
arithmetic equality, the device memory envelope, a hard process-RSS bound,
compiler behavior, runtime cleanup, full-model correctness or performance.
Smoke source SHA256:
`cac89f54ce7ded33e09b7b8e6bfa61caa03af1ae80f2c36b4b2e658afa0e8213`.
