# Three Logits workers: actual code review and failure record

2026-10-20. The user explicitly authorized sending related E055/E057 and C1/C8
sources, documents and experiment data to api.logits.com. The earlier automatic
approval rejection is preserved; this authorization resolved it.

| API model / installed profile | Actual Codex review | Completed shell calls | Duration |
|---|---|---:|---:|
| deepseek-v4.1-flash / logits-deepseek | finished |12|13.491 s|
| kimi-k3 / logits-kimi | finished |16 (1 failed command)|38.022 s|
| muse-spark-open / logits-muse | finished |6|12.279 s|

All three sessions identify `logits-local` and the requested model in actual
session metadata. They used the installed profiles and local Responses adapter.
This extends prior synthetic API/streaming/coding/follow-up tests to actual
repository review. GPT's configuration was not edited. Existing native GPT
subagents reported an account usage limit; no fresh GPT worker result is claimed.

Connectivity is successful; research accuracy is a separate test. The original
answers in v1/v2/v3 are untrusted model output, preserved as text, NOT accepted
technical guidance:

- DeepSeek confused the E056 prerequisite auditor (`3ce4...`) with the E057 cost
  auditor (`09e9...`) and recommended a false hash fix. Local source review found
  the paths/schemas correctly distinct; DeepSeek explicitly retracted in v2.
- The first prompt included frozen preparation READMEs, which are historical,
  so all workers missed some completed results. The prompt-only six-file bound
  also failed: broad searches exceeded it. Follow-ups supplied bounded exact
  excerpts with current execution reports and no tools.
- Kimi's first review correctly used range/median as OFF spread. The root
  reviewer's v2/v3 prompt incorrectly told it the reducer uses max/min-1.
  Direct source inspection confirms `(max-min)/median`. These formulas are
  not equivalent; Kimi v3 wrongly reconciled them as equivalent under that bad
  correction. This is a recorded root prompt error, not a defect in the original
  Kimi calculation. v3 also mislabels pair1's 1.877% as the median 2.112%.
  Its fallback explanation is internally
  inconsistent: a different valid regular path delegates to the original resolver.
  Those statements are rejected. Three positive pairs do not establish a
  population-level speedup. There is no reason to rerun an unchanged closed audit.
- Kimi v2 returned HTTP200 but no final content: all6000 completion tokens were
  reasoning, with `finish_reason=length`. v3 explicitly set `reasoning_effort=none`
  and returned nonempty final content in5.110s. Preserve this as an actual failure;
  HTTP200 alone is insufficient. The installed adapter profiles already disable
  reasoning; the failed direct-API packet had omitted this setting.
- Muse's first bandwidth equation had wrong dimensions. v2 identified the real
  missing feature interface but proposed MLP-input/final-norm captures. The actual
  model captures complete decoder-layer outputs. v3 corrected that location,
  yet mixed the old snapshot/rollback interface with private prefixes and gave
  inconsistent break-even formulas. Only the independently checked interface
  observation is adopted. Captured feature bytes, not just posterior IDs, must match.

For a complete speculative round, throughput is total emitted tokens divided by
total elapsed time. Compare `sum(round_cost)/sum(actual_useful_outputs)` with
native cost per output under the same workload. Count correction/bonus outputs,
respect terminal/output-cap conventions, and include draft, verify, selection,
commit and scheduling. Byte traffic divided by bandwidth has units of time.
Local E057 speed ratios alone are not serving gains.

The actionable result is the default-OFF
[`rowwise_mlp_shared_features`](../../../../bench/m5_ultra_256gb/rowwise_mlp_shared_features/README.md)
prototype and CPU tests. No service/GPU operations were performed by these workers.
Review packets, provider responses, commands, stdout/stderr and original failures
remain in local `.artifacts/m5-ultra-256gb/logits-research-2026-10-20-v{1,2,3}`.
`local-evidence-manifest.json` hashes those local non-session files. Compact
reports, route/results and reproducer scripts are committed here;
credentials and Codex session homes are excluded. One small review task per model
does not establish a general model ranking.

Export-only cleanup: the first staged whitespace check (`6026de`) rejected trailing spaces/blank EOF lines in model responses. Markdown exports now normalize line-end whitespace; original bytes remain in `.artifacts`, with before/after hashes in `markdown-export-normalization.json`. No source or technical content changed.
