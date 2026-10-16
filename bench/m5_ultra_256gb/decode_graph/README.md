# M5 Ultra decode graph experiment

`compile_decode=True` reuses the full single-step decode graph with explicit
cache inputs/outputs and captured parameter inputs. The opt-in is bounded to
MLX 0.32.3, Apple M5 Ultra, this 27B NVFP4/BF16 model, batches 1/2/4 and used
length <=256. Other calls retain the eager model. It does not reuse prompts.

Qualification compares all logits and 128 cache arrays at each step, including
mixed lengths, the 256-position boundary, fallback, array-weight replacement,
request insertion/removal, lookahead, and stopping. Recreate the engine after
changing model structure or non-array configuration. Array weights can change.

The performance candidate combines graph reuse with a 2 GiB MLX allocator pool.
The baseline uses the eager model and a zero-size allocator pool, as previous
HTTP harnesses did. Both enable compiled GDN gates and disable V19 norm reuse.
The allocator pool retains unused buffers, not request contents or KV prefixes.
`screen.py` separately measures the allocation-pool and graph effects; those
same-process exploratory runs are not fresh-process adoption evidence.

`frontier.py --prepare` freezes sources after `qualify.py` succeeds. Run it
without `--prepare` for three fresh processes per configuration: C1 eager,
candidate, oMLX and Lithos; C2/C4 candidate and oMLX. Each has three warm requests
and three measured groups. No retry or discarded negative pair. The four MLX
controls are 50/50/20/0, with TF32 unset. Lithos uses the previously qualified
local M5 Ultra profile and accelerator=on. Its OpenAI endpoint supports C1 only.

Owners use `../lithos/guarded.py` for qualification/screens and the frontier's
own process/resource guards for HTTP. Stop only our child if the other model
comparison chat begins GPU benchmarking. Preserve every attempt and verify
owner/server PID absence after the controlling tool returns.

Example core-engine use (different timing boundary from HTTP):

```sh
.venv/bin/python -m tutorial_code.benchmark_core \
  --model-dir models/Qwen3.8-27B-nvfp4 --compile-decode --cache-limit-gb 2 \
  --prompt-json .artifacts/m5-ultra-256gb/fullqual-build2/workload.json \
  --max-new-tokens 32 --warmup-runs 3 --repeat 3
```

Warming excludes initial graph tracing from the reported steady-state median;
all warm request timings remain in the raw results. The ordinary engine
default remains eager. A short-context result is not a long-context claim.
