# B2/B8 real-activation canary — prepared, not run

`qualify.py --plan FILE --plan-sha256 SHA --out NEW` is a local-layer numerical
screen. It does not install a server or mutate the frozen candidate, B1 canary,
core, E024 or E028. Importing it loads only SHA-bound standard-library helpers.
No device run or numerical result exists yet. The qualifier's 12 focused fake
CPU tests and the independent auditor's 8 tests both passed on their first
guarded run, with zero errors/skips/forbidden operations and unchanged sources.
Their complete evidence is in [the CPU archive](../../../docs/experiments/m5-ultra-256gb/gdn-batched-canary-prepared-cpu-v1/README.md).

The exact plan has five fields: `schema` =
`GDN-batched-prework-real-layer-plan-v1`, `qualifier_sha256`,
`candidate: {path, sha256}`, `e024_plan: {path, sha256}`, and
`deadline_s` in `(0, 1800]`. The inherited E024 plan keeps its original model,
core, runtime environment and eight D512 fixtures. The new source pins the
frozen B1 helper and batched candidate; candidate pins include `gdn_metal.py`.

Four reference chains run in order:

| Chain | Original fixtures, sorted by prompt length then ID | Decode window |
| --- | --- | --- |
| B2 early | Shortest two | 1–3 |
| B2 late | Longest two | 129–131 |
| B8 early | All eight | 1–3 |
| B8 late | All eight | 129–131 |

Each row is prefilled from its complete original prompt and consumes its own
actual native greedy token at every subsequent batched forward. Native batched
Linear arithmetic is preserved; there is no forced per-row M1 oracle. Ordinary
EOS remains enabled. If any row ends before the fixed window finishes, the
result is INCOMPLETE; rows are not silently replaced, retired or force-continued.

The reference is one ordinary full 64-layer model, gates ON, norm reuse OFF,
compile/fuse decode OFF. At GDN ordinals 0/24/47 it saves full input x/conv/S and
output/conv/S in original BF16/FP32. The idle local candidate initializes each
state once, then consumes its own preceding conv/S for three calls; later x
comes from the reference. All 36 calls require evaluated candidate counter
increments, finite arrays and exact unsigned-byte equality. Any execution error,
input change or first bit difference stops immediately. The three layers' full
parameter-byte receipts are checked before and after; this is not an offline
rehash of every model parameter array.

Full retained tensors comprise 108 layer files (36 reference inputs, 36
reference outputs, 36 candidate outputs), 20 complete prefill logits and 12
captured complete batched logits. Exact file/tensor hashes and metadata are
stored with every tensor file. All intermediate greedy IDs are recorded; full
logits for the intervening late-history steps are not archived. After actual closure, `audit.py --child CLOSED --plan-sha256 SHA --out NEW_JSON`
independently checks all 140 original-dtype files, 108 output/state pairs,
48 continued-state pairs, 80 full-vocabulary greedy rows and 324 device comparison
array receipts. Its eight CPU tests prove parser/causality rejection mechanisms,
not actual device equality. Intermediate-history IDs and selected-layer parameter
hashes remain source-pinned producer receipts; their full payloads are not
reconstructed offline.

The auditor reuses the frozen B1 raw parser in a separate namespace, changing
only its single-file ceiling from 16 MiB to 32 MiB. B8 FP32 S alone is 25,165,824
bytes; S plus BF16 conv/output and header is below 32 MiB. Header/span/shape/dtype,
finite and exact-byte checks are unchanged. This bounded file-size increase is
not a model-memory budget change. An actual audit needs a separate bounded CPU
window for reading finite FP32 words; no actual audit or timing claim exists yet.

`input_at` intentionally loads the stored input twice: the reference and candidate
must have independent MLX storage. Candidate state is initialized independently
once per chain; its later state comes only from its previous invocation. This
extra load is outside any benchmark and does not change projection arithmetic.

`complete.json` uses `GDN-batched-prework-real-layer-result-v1`. PASS sets
`canary_qualified=true` only after all four reference chains, twelve candidate
state chains and 36 calls complete, hooks restore, parameters match and model/
source pins recheck. `full_model_qualified`, `http_qualified` and
`performance_qualified` remain false. This does not qualify B3–7, row-changing
schedules, ahead reuse, changed parameters, strided inputs or combined switches.
The root must provide the unchanged hard owner deadline, physical/resource
checks and cold-service recovery before any device execution.
