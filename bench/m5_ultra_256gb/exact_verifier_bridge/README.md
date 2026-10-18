# Native decode / verifier bridge

Source preparation only. No CPU tests or device calls have run for this entry.
The original model, SpecRunner, E027 experiment and all current timing sources
remain unchanged. E027's failed native multirow bridge remains a failure.

`qualify.py` defaults to describing the contract. An existing cold/resource owner
may later launch it with the standard three arguments:

```text
<pinned Python> -I -B bench/m5_ultra_256gb/exact_verifier_bridge/qualify.py \
  --plan <absolute new plan.json> --plan-sha256 <actual plan SHA> --out <new directory>
```

The plan has exactly four fields: `schema: Exact-verifier-bridge-plan-v1`, the
actual `qualifier_sha256`, `e024_plan: {path, sha256}`, and `deadline_s` in (0,1800].
The E024 plan SHA is fixed in the source; it binds the original E023 model,
complete source manifest and eight D512 fixtures. Its old diagnostic execution
flags are **not** substituted for the bridge's explicit native settings:
compile/fuse decode ON, gates ON, norm reuse OFF, max_len2048, target-only C1.
There is no service controller here. Outer admission, live resource monitoring,
hard TERM/KILL/wait and strict child absence are prerequisites, not results of
this script. Do not execute it during another timing run.

The shortest and longest original fixtures each supply decode anchors1 and129.
At each anchor, the original `runner.decode_model` first generates eight actual
teacher tokens. Early normal EOS/cap means INCOMPLETE; no synthetic tokens or
ignore-EOS continuation is allowed. All prompt bytes/IDs and cap512 are retained.
Each T2/4/8 case reloads two physically independent copies of the same saved
canonical state, including all128 original-dtype state arrays and KV padding.
T includes the anchor, so these widths represent1/3/7 possible draft tokens.

The oracle performs T ordinary C1 decode calls. The other arm executes the
existing SpecRunner target expression: snapshot each staging cache, then
`model(block, ForwardContext('prefill', caches), capture=())`. No fabricated
drafter is installed. Empty capture deliberately excludes drafter feature
retention from this first mathematical bridge. The model's prefill path already
has norm reuse disabled regardless of capture. The two arms consume identical
fixed IDs; candidate argmax is recorded only and never becomes an input.

The gate compares T complete vocabulary vectors and all128 **final** state
arrays using the unchanged E023 finite/original-bit Comparator. Cache class and
record/tape bookkeeping are separately recorded. Representation conversion only
renames fields and restores the singleton batch dimension: it does not cast,
mask or trim KV backing. Both full output artifacts are saved before the bit
comparison; the Comparator retains its first differing tensors. The input seed
arrays must remain unchanged. All parameter values and source/model files are
checked before/after a complete run; parameter handle IDs are evidence only.

A PASS would qualify only the12 fixed teacher-forced forward blocks. It does
not qualify intermediate accepted prefixes, rollback, drafter features,
normal-EOS acceptance, scheduler/ahead, HTTP, performance, or E024/GDN/E028
combinations. `forward_qualified` is separate from the permanently false
`rollback_qualified`, `full_model_qualified`, `http_qualified` and
`performance_qualified` flags. A failure stops immediately and retains the
prefix and original raw tensors; postchecks not reached remain explicitly false.
Raw complete state artifacts may total several GiB; archive policies must retain
them without putting an oversized single payload into Git.

The next step after a failure is a separate diagnostic of that exact failed
block, locating the first operation with equal input but different output.
Likely boundaries include M1 versus multirow quantized projections, conv sum
order, compiled gates, chunked GDN versus recurrent Metal, and attention's
different RoPE/SDPA shape. None has been identified as the actual first failure
by this unrun source. Rollback must later be tested independently: GDN partial
rollback uses a different chunk size, and attention rollback leaves rejected KV
tail bytes. No tolerance or valid-prefix-only substitution is allowed here.

`test_qualify_cpu.py` contains eight stdlib fake mechanism tests for inert CLI,
partial arguments, teacher-token causality, normal EOS, the12-case selection,
snapshot/prefill dispatch, full-state representation and failure retention.
They are prepared but **not executed**, and cannot establish MLX/model equality.
