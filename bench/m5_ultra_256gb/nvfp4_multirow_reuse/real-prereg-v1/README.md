# E038 real-weight projection screening

Frozen before GPU execution. Plan SHA f203aa842594a6b02bf787b0e780c89f59e7a2956a97a115a0ba380ef5f4b714.

Two actual Qwen projections (head and layer0 up), T2/4/8; synthetic BF16 activations, not recorded hidden states. Complete raw-output equality against independent native M1 and custom B1 precedes timing. Five warmups per route, five groups of twenty alternating AB/BA pairs per baseline, 2400 retained measurements total. Fresh operation construction, eval, synchronization, baseline concatenation and candidate staging count. This is one-process local screening, not engine performance or Pareto.

The existing user-authorized SIGSTOP/SIGCONT lease temporarily pauses original oMLX PID39988 with exact start identity. Its finally and watchdog restore it. Outer owner is bounded to160 seconds; child hard limit is120 seconds and4GiB MLX guideline. Resource/power/thermal snapshots and original service/model health are retained. No model settings are changed.

CPU entry guard actually passed twice: console attempt467bbd and captured attemptbc3828; the latter raw command, stdout/stderr, returncode and identical source snapshots are in cpu-evidence. No GPU was executed by those checks. Selected model-weight slices remain local and must be excluded from Git archives.
