# Rowwise native qualifier: independent source review

Reviewed `bench/m5_ultra_256gb/rowwise_native_verify/{qualify.py,test_qualify_cpu.py,candidate.py}`
against the actual frozen cache, model, compiled decoder and retained bridge/helper
implementations. Source reads only: no imports, AST execution, hashes, tests or
backend/device actions. No files under the experiment were changed.

No concrete execution-contract or false-PASS blocker found in this review.
UNSEALED source constants correctly prevent execution and are intentional.

- `qualify.py:98–108,134–155`: used257 means pending token position256 after a
  separately declared 256-token diagnostic prompt. `DecodeBatch.rebuild` receives
  required length1536; its native ceiling-by256 gives exactly1536. The committed
  batch length remains the real prompt offset256, not1536. The other anchors use
  the entire original shortest fixture at used1023/1024. All actual decode bounds
  exceed256, so frozen `CompiledDecode.__call__` falls back to ordinary model
  execution. The terminal empty graph/pack/ahead check matches that route.
- `qualify.py:141–167` and `qualify_native.snapshot:71–85`: the canonical snapshot
  retains independent `array[:]` descriptors. The native attention writes replace
  its own K/V handles; GDN replaces complete conv/S values. Each reference output
  is fully evaluated and written to its unique prefix file before the next
  reference step. Later teacher steps cannot rewrite those saved raw files.
  The retained canonical view is compared to the saved canonical after all eight
  teacher steps, so this ownership property is itself checked on device later.
- Candidate prefixes own distinct wrappers and K/V descriptors. Each layer uses
  its own previous-prefix state, B1-shaped input, real absolute position/lens and
  the native layer arithmetic. The true model call supplies the same
  `_norm_reuse_normal=True`; norm reuse is OFF and native gate compilation is ON.
  Additional concatenation/row views can still affect actual dispatch or numerical
  behavior: source equivalence is a hypothesis for the exact gate, not a PASS.
- `compare_case` evaluates logits and all states of all T prefixes before writing
  input-after and comparing the supplied inputs and original loaded canonical
  arrays. Every prefix compares one full vocabulary row and all128 original-dtype
  state arrays, including complete1536 KV padding. The original Comparator checks
  metadata, dtype, shape, finite flags and unsigned bit equality. Greedy IDs are
  an extra causal check rather than the numerical gate.
- The unchanged `reference_tokens` generates fixed inputs from original C1 only.
  EOS before the necessary teacher window yields INCOMPLETE, with no fabricated
  continuation; EOS emitted by the final required reference call may remain its
  output. No sampler/acceptance/rollback/retirement is installed or qualified.
- Each candidate exception still runs input-after/invariance handling and never
  retries the advanced call. Device errors retain the current case and raw
  first-failure artifacts, clear unwound tensor-bearing frames, then clear runner
  state/references and synchronize/clear cache before the release event. The
  owner remains responsible for process-level reap. Cleanup itself can fail, in
  which case cleanup-completed/PASS cannot be granted.
- Successful completion requires all9 cases/all42 prefixes, original-byte full
  model parameter receipts before and after with exact values/shape/dtype/finite
  equality (handle IDs excluded), source/model rechecks and cleanup. A failed
  comparison can exit before the after-parameter receipt; it receives no forward
  qualification. Neither parameter replacement nor an independent raw audit is
  supplied by these producer receipts.

The fake tests cover interfaces and failure preservation; they do not establish
real MLX view ownership, numerical equality or memory release. A future independent
raw auditor is still required to reread saved canonical/input/reference/candidate
payloads. Only this fixed teacher-forced prefix-forward scope could be qualified;
no full-model rollout, scheduler/ahead, HTTP or performance conclusion follows.
