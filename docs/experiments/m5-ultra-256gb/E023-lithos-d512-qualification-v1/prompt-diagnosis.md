# Lithos D512 v1: confirmed double-template input failure

CPU diagnostic only. The eight completed C1 requests are not qualified; C8 was
not executed. No old source, input, output, or failure receipt was modified.

The builder used rendered `fixture.prompt_text` as `/api/chat` user content.
Official `monolith/serve.py:293` applies the same chat template again with
`add_generation_prompt=True, enable_thinking=False`. Frozen fixtures were already
rendered with those exact options (`prepare_fixtures.py:53–57`). This is an input
construction bug, not a different official template default or inference result.

| Fixture | Actual IDs | Frozen IDs | First differing index (zero based) |
| --- | ---: | ---: | ---: |
| D01-ttl-lru | 989 | 978 | 3 |
| D02-dag-scheduler | 897 | 886 | 3 |
| D03-event-rollup | 937 | 926 | 3 |
| D04-frame-parser | 932 | 921 | 3 |
| D05-retry-budget | 965 | 954 | 3 |
| D06-interval-booking | 946 | 935 | 3 |
| D07-versioned-store | 914 | 903 | 3 |
| D08-sliding-limit | 969 | 958 | 3 |

All eight differ first at index3: actual248045 (`<|im_start|>`) versus expected60856
(`Implement`). Each actual prompt is11 tokens longer:7549 total versus7461.
The observer closed all34 events (open1, eight each start/return/body/wire,
sealed1). Actual raw/committed output counts are4096/4096 and observedEOS0.
These outputs cannot become fair-workload benchmark points by changing the oracle.

Official CPU tokenizer versions are transformers5.18.0, tokenizers0.23.2 and
jinja2 3.1.6. For all eight fixtures, applying the official template to original
`fixture.messages` exactly reproduces every frozen prompt ID; applying it to the
incorrect old body exactly reproduces every observed old prompt ID. New retry
`jobs_for` was also invoked for actual8/C1 and16/C8 jobs, and all24 new request
bodies reproduce the frozen IDs exactly. No MLX/Torch/JAX/TensorFlow import.

Minimal proposed patch changes only job messages and the exact join body check;
the original full prompt-ID gate and all route/count/cache/EOS gates remain.
Retry regression suite18/18 PASS. Its first run passed17/18 and failed only on
the copied saved-log fixture's relative location; only the test's path changed.
The earlier diagnostic guard failed before tokenization on a package-availability
lookup; the actual-import guard then succeeded. All attempts are retained in JSON.

Artifacts: `/tmp/lithos-d512-v1-prompt-diagnosis.py`, `.json`, this `.md`, and
`/tmp/lithos-d512-prompt-minimal.patch`. JSON pins original raw artifacts, original
and retry sources, tokenizer inputs, versions, full actual/frozen prompt IDs,
and the diagnostic script itself. No large model payload was read.
