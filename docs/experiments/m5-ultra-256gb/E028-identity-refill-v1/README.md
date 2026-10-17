# E028 v1: input generation failed before candidate rebuild

The experiment is **not numerically qualified**. The first case failed while
constructing its first BF16 source tensor: frozen `screen.py:182` evaluates
`0x3e00 | low`, which raises `TypeError: unsupported operand type(s) for |:
'int' and 'mlx.core.array'` on the recorded MLX 0.32.3 runtime. The candidate
factory and OFF batch initializer had executed; neither the candidate's
`rebuild` method nor its identity-refill branch was reached. This is an input
generator failure, not evidence of a candidate numerical mismatch.

Actual coverage is **0 completed cases, 0 comparisons, 0 tensor snapshots**, and
two self-resource phases (`initial`, `before_case`). The intended plan remains
10 cases, 88 comparisons and 53 phases. No gate, test input or raw record was
changed to turn this attempt into a pass. There is no throughput, latency,
full-model, HTTP or memory-saving result here.

Child 58427 exited 1. Its owner 58415 subsequently attempted a footprint probe,
which returned 66 because the child had already exited; that is a secondary
monitoring failure, not the original cause and not a 120-second timeout. The
child ran for 0.0741 s and the owner for 0.5952 s; these are failure durations,
not performance measurements. Child wait/reap and strict PID absence are
recorded; the outer lease waited for owner exit 1.

All three owner resource samples have pressure 2, Swapouts 316235 and swap used
445.06 MiB, with at least 50 GiB real free pages. Both child self-footprint
phases recorded 28,443,368 bytes current/peak and zero materialized MLX memory.
The owner obtained no successful child footprint sample. The archived lease
binds the exact owner command and plan, closes, and records the original
74384 identity (`Fri Oct 16 08:01:58 2026`, `omlx-server`) returning T→S.
Restoration is verified from the pinned lease implementation and its saved
receipt; the offline auditor does not claim a new live service probe.

`raw/` preserves every file from the four original owner, service-lease,
inputs and CPU-prepare directories. Their original file bytes were compared
with the copies. This includes the frozen v1 sources, failed owner physical
probe, all partial results, and the 20-test CPU-prepare receipt. CPU mock tests
had not exercised this real MLX reflected-operator path.

The independent verifier hashes all 1,078 archived source copies (539 in each
of owner and CPU prepare), binds all 570 pre/post pins, reparses physical and
resource records, and checks cleanup and the lease. Its first receipt also
rehashes the 570 original runtime/source paths (259,496,954 bytes); runtime
binaries themselves are not duplicated in this archive.

Important identities:

- Plan: `652486a29e021bd8009bc27a1bdede607c63aa56d9fb84bf70940860a8e437ee`
- Candidate: `bb9f8a461625e97c80e75140c9d9b7b8718a633595ab4f3a422267e752ded8f5`
- Screen v1: `160c8848026e8640827dee0e739d8d68040c520a5e1517041eac87d4116bace3`
- Owner v1: `524d8c6471cdf80c645205c27add36f8a881d674f36c3a71b3dd7a4594b3a61a`
- Lease source: `76108bc98300183658a2ad71cb542c0dc8e061253c5120daf86074e43e0b248f`

Reproduce the offline failure verification, without importing MLX or running
any measured code:

```sh
python3 -I -B docs/experiments/m5-ultra-256gb/E028-identity-refill-v1/verify.py
```

Add `--runtime-pins` only if the original installed paths still exist unchanged.
`independent-audit.json` distinguishes successful evidence verification from
the experiment's **INPUT_GENERATOR_FAILURE_NOT_QUALIFIED** outcome.
`manifest.json` records every archive input's size and SHA-256; `evidence.tar.gz`
contains those files plus the manifest. `tar.verify.json` records the separate
full member-byte verification and tar digest. The tar and its verification
receipt are intentionally outside their own manifest.
