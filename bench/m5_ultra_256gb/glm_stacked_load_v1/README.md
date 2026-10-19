# GLM stacked checkpoint: official loader diagnostic

Actual CPU preparation passed on 2026-10-19. This independent namespace admits
E036's real 58-shard derived checkpoint, retaining the original GLM v3 cold
lifecycle, direct child, dual resource monitor, cleanup and restoration. It does
not modify the old probe, lifecycle, prior journals, original model or service.

The old probe and inventory fixed the original Hugging Face revision, 62 shards,
203,992,076,296 file bytes, and original download/header receipts. Changing only
a model alias would fail those checks. This adapter explicitly uses E036's
PASS_BYTES_ONLY producer receipt, independent header/index/manifest proof,
source-plan/candidate hashes, actual 58-shard header/stat identities and unchanged
sidecar hashes. It never labels this as the original download or a new payload
hash. All 4,960 tensors and the remaining MTP layer-45 stack retain that scope.

`load_probe.py` copies the original probe and changes only checkpoint identity,
metadata/evidence validation and required source pins. A CPU AST comparison
verified unchanged runtime load, four plain completion requests (caps
8/32/32/32), observation, token accounting and shutdown functions. The real model
alias is GLM-5.3-Flash-4bit-stacked-v1. Official EnginePool defaults, ordinary
target-only decoding, prefix cache off, all MTP/speculation off, and existing
42-layer gate/up fusion remain. No allocator-limit override is introduced.

`owner.py`, `controller.py`, auditors, registrar and driver bind fixed existing
functions to their actual local source namespace. The old source modules are
unchanged. The owner keeps 220 GiB child / 240 GiB system ceilings, zero swap
growth, separate 1-second physical and 5-second metadata cadences, and bounded
owned-child/probe-command cleanup. Failed monitoring can restore only after
settled/reaped proof; it does not become a successful diagnostic.

The sealed predecessor is the actual E035 RESTORED journal and independent
restoration PASS. It includes the three real profile-phase child reap chains and
replacement identity PID 39988, PPID 1, Mon Oct 19 00:56:46 2026, omlx-server.
A read-only process check (`4cc2d0`) matched that identity. Preparation must still
verify the current service, settings and model. If another cold cycle replaces
this PID, this binding must be updated from that cycle's real closed evidence;
old predecessor records are never rewritten.

Actual validation:

- 12/12 guarded CPU tests PASS, zero failures/errors/skips/forbidden operations;
  unchanged sources, 1.816907959 s. Original receipts and snapshots:
  `.artifacts/m5-ultra-256gb/glm-stacked-cpu-v1`.
- Actual input preparation, CPU registration and owner inventory verification
  all PASS under the same backend/process/network/timer prohibitions;
  unchanged sources, 2.293989792 s. Original commands, logs, proof and snapshots:
  `.artifacts/m5-ultra-256gb/glm-stacked-input-registration-attempt-v1`.
- Independent source review found no blocker in the derived model binding,
  local function dependencies, auditor inventory path or recovery namespace.
  These are CPU/static results, not a loader, fit, numerical or performance result.

Prepared inputs are in `.artifacts/m5-ultra-256gb/glm-stacked-inputs-v1`.
Registration is `.artifacts/m5-ultra-256gb/glm-stacked-registration-v1`, with
76 pinned sources/inputs and owner-plan SHA256
`ca2a2e280c29aea239c640d41a5b1e9e6af71138cabfac6f02e22d887967affa`.
Its `commands.json` records the exact read-only controller preparation, owner,
stop, recovery and two offline audits. At this document's creation, controller
preparation/sealing and actual execution have not run. Root owns those operations.

After actual read-only PREPARED output and `preregister.py --seal-prepared`, root
must commit/push the source and registration before running:

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -I -B \
  /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/glm_stacked_load_v1/execute.py \
  --run \
  --registration /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/glm-stacked-registration-v1 \
  --out /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/glm-stacked-execution-v1
```

The driver retains original stop/direct-owner/finally-recover and unknown-launch
handling. After final closure, the saved restoration_audit and closed_audit
commands produce separate raw evidence conclusions. No diagnostic result here
can qualify reference-equivalent GLM inference, chat quality, HTTP serving,
throughput/latency or a Pareto point.
