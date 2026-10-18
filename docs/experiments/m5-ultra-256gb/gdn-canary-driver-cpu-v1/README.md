# GDN canary driver: CPU wiring evidence

One guarded attempt ran all 12 whole-module fake CPU tests: PASS, zero failures, errors or skips. No backend imports, real subprocesses, network, signals, services or GPU work were allowed. The historical oMLX restoration was an evidence prerequisite, not an observation of the current service.

The full runner, command, stdout/stderr, per-test outcomes and before/after source snapshots are in `evidence.tar.xz`. `manifest.json` maps each original file to its exact archive member, SHA256 and size; `archive-verification.json` records complete member and original-file rechecks. Tests cover default-inert execution, fixed CLI without a child --run, source/PREPARED binding, failed/unknown launch and finally-recovery routing. They do not prove device execution, numerical correctness, physical resource safety or real restoration. Original artifacts remain untouched.
