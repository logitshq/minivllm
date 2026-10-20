# E058 — expose target features for a complete speculative round

Source prototype, default OFF. No GPU qualification, serving integration or
performance result yet. The E057 frozen candidate and measurements are unchanged.

The shared verifier already provides full logits and a private cache for every
input prefix. DFlash also needs target hidden features: five complete decoder
layer outputs at layers `(5,19,33,47,61)`, concatenated on the hidden dimension.
`model/qwen3_5.py` captures after `x = layer(...)`. Capturing the MLP input
normalization or final model norm would be wrong even if logits still matched.

The candidate is an explicit small extension of the frozen E057 candidate SHA
`02200afa33fd6daa3f0f4dac3436b717616cf620f4b6cd7e59f1d8f263658950`.
It appends the same `x` after each selected complete decoder layer. There is no
new arithmetic inside a layer and no extra forward pass. `capture=()` omits
feature concatenation; `enabled=False` returns without inspecting inputs.
Only the fixed five-layer tuple is accepted. The shared MLP projector is required
when requesting features. T2/T4/T8 remain the candidate's admitted widths; the
existing E057 device evidence covers T4/T8 only.

`round.select_greedy` is a pure selection helper. For inputs `[anchor]+drafts`,
it finds the first draft/posterior disagreement, accepts `n` drafts, emits
`drafts[:n]+[posterior[n]]`, and selects `prefixes[n]` and `features[:n+1]`.
That cache contains the old anchor and accepted drafts; the new correction or
bonus is pending, not already computed. Full rejection still advances one input
and emits one correction. All accepted emits T tokens and keeps T inputs.
The helper does not mutate or install any cache. EOS/output-cap truncation,
vocabulary admission and batch/staging conversion remain the caller's job;
this is not a production SpecRunner replacement.

CPU scalar-fingerprint tests cover independent token-major layer-output capture,
logits and every cache prefix across T2/T4/T8 and positions spanning 1023/1024;
capture OFF; shared head ON; fixed capture admission; every first-rejection
position and all accepted; and canonical cache immutability. These do not prove
MLX aliasing, BF16 byte equality, GPU performance or a real drafter's acceptance.

Next device gate: use T4/T8, the same real E057 seed/weights, and compare full
feature bytes against a native token-major oracle capturing complete layers.
Keep existing full-logit/all-prefix-state/native-continuation checks, plus
feature slicing at every possible accepted prefix. Repeat multiple rounds with
real draft IDs, including forced first/middle rejection and all accepted.
Only after that passes, measure complete draft+verify+selection+commit cost and
actual produced token counts. Native throughput is the baseline; optional
head-only/shared-MLP arms must run in the same paired experiment. No new GPU
owner plan is prepared at this source checkpoint.

Actual CPU runs: attempt1 7/7 PASS; after adding shared-head feature coverage, attempt2 8/8 PASS. Evidence: `docs/experiments/m5-ultra-256gb/E058-feature-interface-source/`.
