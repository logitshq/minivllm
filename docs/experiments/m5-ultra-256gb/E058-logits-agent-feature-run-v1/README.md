# E058: actual Logits coding agents and feature verification

This extends the earlier read-only Logits reviews into code-writing workers.
The three models use their installed Codex profiles and the local Responses
adapter. No GPT/native worker is relabeled as a Logits worker.

| Worker | Actual API model | Code responsibility |
|---|---|---|
| DeepSeek | deepseek-v4.1-flash | independent token-major feature reference and candidate interface |
| Kimi | kimi-k3 | independent raw safetensors parser, finite checks, shape checks and exact-byte audit |
| Muse | muse-spark-open | rotated serial timing schedule, strict reducer and failure propagation |

Root owns integration, code review, source pinning and the single GPU queue.
Parallel code research is useful; parallel GPU timing would contaminate these
measurements. Each worker has a private output directory, recorded commands,
stdout/stderr, session model metadata, exit status and revision snapshots.

## Failure and correction record

- DeepSeek's first reference concatenated five `[T,5120]` feature arrays along
  the token axis, producing `[5T,5120]`. Some attempted test-scaffold repairs
  adapted expectations instead of fixing the source. Root interrupted that
  worker after191.455s and resumed the same session with the concrete shape
  counterexample. The repaired reference concatenates on the hidden axis and
  slices both GDN arrays when making cache snapshots. A rejected candidate now
  returns `None` explicitly. The interrupted attempt is retained as failed.
- Kimi's first parser misused `object_pairs_hook=list`; JSON returned tuple
  pairs and converted nested metadata to lists, so valid files were rejected.
  Its own tests found this. It changed to a duplicate-rejecting dict hook. The
  original finite scan looped over every scalar; root requested a vectorized
  NumPy bitmask scan with all65536 BF16 bit patterns and sampled F32 equivalence
  checks. Root also required hashing the same file bytes that are parsed,
  replacing a two-read gap. Test-fixture scope and class-layout errors during
  this revision are preserved in the worker log.
- Muse's first collector rotated only by repeat index and never used the third
  rotation. Its reducer did not verify the exact arm order and its callback
  wrapper swallowed log-write failures. Root supplied counterexamples; Muse
  corrected the schedule to `(group*2+repeat)%3`, validates exact schedule rows,
  and propagates failures. Root independently reran all8 tests.
- Muse's extra source review wrongly proposed resetting seed position to0,
  misattributed cleanup functions to other files, and suggested a redundant
  final partial-log write. Root rejected these recommendations. The actual
  seed position remains1022/lens1023; successful samples are logged immediately.
  Muse's subsequent review explicitly retracts those claims.
- Root integration tests caught its own `T`/`width` field mismatch and JSON
  integer-key normalization issue; these were corrected before preparation.
  Root static inspection also corrected Python import-path order so frozen
  model sources take precedence over the current repository copy.

These failures are part of the experiment record, not claims about general
model rankings. A small worker task or passing connectivity test cannot establish
research accuracy. Original local logs are hashed in the evidence manifest;
compact reports and revision snapshots are committed. Session homes and secrets
are excluded.

## Measurement boundary

The test checks actual T4/T8 features, all12 prefix choices and their12 native
continuations, with full vocabulary logits and all128 cache arrays per record.
The independent reference must also match the original model's capture method.
Then60 rotated timings compare the original token-major capture forward with
shared MLP/head + capture, plus shared MLP/head without capture as an overhead
control. All arms use the same supplied causal teacher tokens and cache seed.

T is verification width, not request concurrency. These are local materialized
forward costs, not end-to-end speculative decoding or serving throughput.
Drafter cost, actual acceptance, selected-state installation, scheduling,
TTFT/TPOT and full engine throughput remain outside this test. The existing
C1/C8 comparison with oMLX and Lithos is unchanged by this experiment.

## Prepared checkpoint

All63 root-run CPU tests passed:19 owner lifecycle,30 Kimi auditor,8 Muse timing,6 root integration. DeepSeek's own test scaffolding remained failing after repeated repairs and is retained separately; no passing result is claimed for it. Its corrected implementation passed root's independent checks.

Parent plan `de5b72f580b9dcdfe09dcf07afa7703d3080d28d6e0d3471a94c9a1469dc4ebf`; child plan `89f4b815bd706a16012d351b88e6b3490fe514db0d834f36a59c62a7c41ec29e`. GPU execution pending at this checkpoint.
