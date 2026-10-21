# E058: three Logits coding agents, real feature validation

Default OFF. This is a local verifier/capture experiment for Qwen3.8-27B NVFP4
on the M5 Ultra, not an HTTP throughput or Pareto benchmark.

The feature candidate reuses the already checked E057 shared MLP gate/up and
vocabulary projections. It captures complete decoder-layer outputs at
`(5,19,33,47,61)` during the same verification forward, avoiding a separate
feature recomputation. Hidden rows concatenate to `[T,25600]`.

Three actual Codex CLI sessions use installed Logits profiles through the local
Responses bridge: DeepSeek implements the independent token-major reference and
candidate interface; Kimi implements an independent safetensors byte auditor;
Muse implements the serial rotated timing collector/reducer. They write and test
code in separate local directories. Root reviews and integrates code and owns the
single bounded GPU queue. See the experiment record for original failures and
revisions; passing API connectivity does not imply correct code.

The screen checks T4/T8 and **every** selected prefix k=1..T. Both the selected
output and one native continuation must match in complete logits, five-layer
features and all128 cache arrays, including KV padding. The independent reference
must additionally match the original frozen model's actual capture forward.
Twenty-four pairs /48 raw files /3120 tensor pairs are retained for Kimi's audit.
Sources/model weights, parameter bytes, canonical state and actual hardware are
checked. The raw auditor runs only after the GPU child exits and oMLX is restored.

Timing starts after numerical checks. Two warmups per arm/width precede60
samples: T4/T8, five groups, two repeats, three rotating arms. The primary native
arm is the frozen model with capture on supplied causal teacher tokens; the
candidate shares MLP/head projections and captures features in the same forward.
The third arm omits features only to diagnose capture overhead. All arms evaluate
complete logits and final selected state; capture arms evaluate features too.
Native is teacher-forced, not device-dependent autoregression. Timings include
Python construction/materialization/synchronization, and exclude disk logging.

The owner is a narrow adaptation of the tested E057 lifecycle. It pauses the
same authorized service identity, enforces bounded deadlines and sampled resource
ceilings, retains the numerical child through a final resource observation,
reaps it, then restores and checks oMLX. A failed numerical check cannot produce a
passing owner. No production engine dispatch is changed.

No drafter, real acceptance distribution, full speculative round, scheduling or
HTTP cost is measured here. T is verifier input width, not concurrent requests.
C1/C8 serving Pareto figures remain unchanged until a full benchmark qualifies.

Run CPU tests from this directory with `PYTHONDONTWRITEBYTECODE=1` and the repository
venv. Explicit `execute.py --prepare` creates a source-pinned plan; `--run` requires
its exact SHA. An invocation without a flag is inert. Never rerun into an existing
output directory or change pinned code after preparation.
