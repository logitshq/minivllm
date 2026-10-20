# E046: B2/B8 real-activation GDN device canary — prepared, CPU PASS

**Prepared against the live machine; 12/12 owner CPU tests PASS; device NOT RUN.**
The actual CPU receipt is at
[`.artifacts/m5-ultra-256gb/gdn-batched-device-owner-cpu-v1`](../../../.artifacts/m5-ultra-256gb/gdn-batched-device-owner-cpu-v1),
with return code 0, `PASS_CPU_ONLY`, and all 12 cases recorded in `stderr.txt`.
This folder adds one small single-child
runner and a thin adapter to E045's repaired outer pause owner; it does not modify
`concurrency_config_v1` / `concurrency_config_v2`, the original qualifier/candidate/auditor, or the engine.

The original [canary contract](../gdn_prework_batched/CANARY.md) remains the
numerical authority: four B2/B8 early/late reference chains, GDN ordinals 0/24/47,
12 continued-state candidate chains and 36 exact calls. Normal EOS stays enabled;
early EOS is INCOMPLETE. Full output/conv/S original bytes, 20 prefill logits and
12 captured full batched logits remain in all **140 tensor files**. There is no
candidate full-model rollout, graph composition, HTTP or speed qualification.
This is a local B2/B8 candidate numerical canary, **not a diagnosis of C8 token
variance** observed in HTTP runs.

## Thin ownership and bounds

`execute.py` binds local copies of the exact E045 `run`, `bind_runner` and
`bounded_cleanup` function code into a local namespace. It replaces only the
working directory, single-child runner and preflight snapshot callback. E045
sources are pinned, not mutated. Its historical `final.json` schema remains
`E045-config-driver-result-v1`; `e046-summary.json` references that raw receipt
without rewriting it. PASS_CAPTURE_ONLY / PASS_CAPTURE_AND_RESTORE describe
producer/owner closure, not independent numerical qualification.

The service is PID **52818**, identity **Mon Oct 19 09:22:08 2026 omlx-server**.
The exact original health checks must pass before pause and after resume. Use the
existing authorized SIGSTOP/SIGCONT lease: retain the resident model and original
PID. No cold stop/reload. Earlier CANARY prose mentioned cold recovery; the
qualifier itself has no cold-service assertion. This owner explicitly uses a
paused-resident resource policy, without changing numerical gates.

Bounds are **64 GiB child / 240 GiB system**, pressure 1, zero swap-byte growth and
zero new Swapouts relative to pre-pause admission, AC power, thermal nominal/fair,
and no other recognized active inference process. Physical and metadata probes
are individually bounded to 5 seconds; physical observations start at most
12 seconds after the previous completion, with a 3-second requested interval.
These are sampled limits, not a hard allocation cap or proof of unobserved peaks.

The exact qualifier is the runner's direct Popen child; its process identity,
stdout/stderr, actual wait/absence and result are retained. The runner reuses the
existing physical guard's direct-Popen reap proof. On unresolved reap it stays
alive, preserving the pause lease for E045's bounded fallback; that fallback
resolves recorded descendants before killing the runner and never signals the
lease during restoration. Unknown descendants remain an explicit reconciliation
failure rather than an assumed clean resume. A short `runpy` wrapper runs the unchanged qualifier in the same PID/PPID. After
its original SystemExit and complete.json it writes an atomic done receipt and
waits at most 60 seconds. The owner checks that receipt, takes a final complete
physical + metadata sample while the child remains alive, then atomically acks
exit and waits/reaps it. This eliminates the ordinary numerical-completion race
without waiving terminal resource sampling. Unexpected child death or timeout
still fails; missing/FAIL/INCOMPLETE numerical output never becomes PASS.

The child deadline is **1500 s**, runner work **1650 s**, outer work **1800 s**,
plus existing bounded cleanup. Reserve roughly **10–25 minutes** for the device
phase and a separate CPU audit window; this is a scheduling estimate, not measured
runtime. The 20 full prompts and two 131-step batched histories, original model
hashes before/after, and about 1.7 GiB retained tensors can dominate time. The outer
adapter checks runtime/source bytes but does not add a third full-model rehash.
The independent original-byte audit runs only after service resume.

## Actual CPU result and prepared interface

The 12 owner CPU tests passed with backend/process/service actions forbidden or
mocked: original SystemExit 0/1, required final-sample ack, child/owner/command/done
hash binding, changed parent, the 60-second deadline, child MLX environment
isolation, failure-path direct-child reap, and the FunctionType namespace binding.
These tests do not execute the qualifier or device work. The original qualifier's
12 and auditor's 8 earlier CPU tests remain separate; the new tests do not claim
every possible timeout, signal or OS failure branch is covered.

Preparation completed after actual E045 v2 closure. The saved
[plan](prereg-v1/plan.json) SHA is
`08325135360d35faa5bc0b9294aca21ee4dc8fa7c974c49b752682184bc13e3b`;
the original five-field [child plan](prereg-v1/child-plan.json) SHA is
`7df6cff99425698a1fb134df4f25dc8de4e717482f73039e56ec5a0a00930003`.
[commands.json](prereg-v1/commands.json) retains the exact run and independent audit
argv. This README is absent from both plan pin maps and was updated after
preparation without changing runtime sources or either plan.

The completed preparation used the following command; do not rerun it over the
existing fresh-only `prereg-v1` directory:

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -I -B \
  /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/gdn_batched_device_v1/execute.py --prepare
```

This requires the actual **E045 v2** lease at
`.artifacts/m5-ultra-256gb/concurrency-config-v2-lease` and execution final at
`concurrency-config-v2-execution` to be closed, bound to predecessor plan SHA
`7b14c40ddf20381dd6fc130addfe18706eb0935d6c54fb105bea39b1667a057b`. It validates the original E024 plan, checks
fixed source identities, and checks the same live service identity/health without
pausing it. It creates only fresh `prereg-v1/{child-plan,plan,commands}.json`.
The numerical child gets the inherited **ops50 / MB50 / BFS20 / sync0** environment,
replacing all inherited `MLX_*` values explicitly. No E045 tuned configuration is
silently inherited, and the parent environment is not changed.

Commit/push the reviewed source and prepared plans before running. Execute the
exact `run` argv saved in `prereg-v1/commands.json`; its equivalent is:

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -I -B \
  /Users/ryanli/code/minivllm/bench/m5_ultra_256gb/gdn_batched_device_v1/execute.py \
  --run --plan-sha256 08325135360d35faa5bc0b9294aca21ee4dc8fa7c974c49b752682184bc13e3b
```

Only after terminal owner/lease closure and restoration, execute the saved
`audit` argv (the unchanged `gdn_prework_batched/audit.py`). Preserve its real exit
and report alongside the owner/resource/restoration result. It independently
checks 140 files, 108 output/state pairs, 48 continued-state pairs, 80 greedy rows
and 324 device comparison array receipts. Do not turn producer PASS alone into
independent canary qualification.

Outputs use fresh `.artifacts/m5-ultra-256gb/gdn-batched-v1-{execution,owner,lease,numerical}`
and `gdn-batched-v1-independent-numerical.json`. The original qualifier's full source
and all original outputs are retained; model weights are not copied. After closure,
archive these artifacts including every failure and member hashes as E046. No
archive or next HTTP benchmark is launched by this adapter.
