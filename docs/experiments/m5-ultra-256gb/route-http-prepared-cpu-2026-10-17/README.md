# Prepared E024 route HTTP CPU evidence — 2026-10-17

The first guarded run executed 23 tests: all 12 server tests passed; capture had 10 passes and one failed fixture assertion. The capture completed successfully, but its actual argv used the correctly resolved macOS `/private/var/...` path while the test expected `/var/...`.

The only repair was `Path(self.temp.name).resolve()` in the capture test fixture. Production sources and gates were unchanged. The focused second run executed all 11 capture tests successfully. The 12 server tests were not repeated. This is 34 total test executions across two attempts, not a fresh combined 23-test rerun.

Both attempts preserve their original command, runner, full stdout/stderr, source/metadata snapshots and before/after hashes. `fixture-fix.diff` is the exact one-line repair. `manifest.json` maps every copied file to its original source, size and SHA; copies were compared byte for byte.

The runner rejected model/MLX imports and real subprocess, socket and process-signal backends. Capture used fake direct children; the server tests used the actual donor worker and witness bookkeeping with fake callables. Neither attempt loaded a model, launched a server, used the GPU or granted numerical, real HTTP or performance qualification. Actual capture/replay and independent audits remain required.
