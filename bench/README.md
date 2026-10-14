# bench/

本地复刻远程契约里的那条跑分：`vllm bench serve --dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 100 --max-concurrency 32`。
Mac 上没有 vLLM，所以 `bench_serve.py` 自己实现了同样的协议：100 条随机 token 的 prompt、t=0 全部入队、最多 32 条在飞、流式 `/v1/completions`、temperature 0，
报 request / output-token 吞吐和 TTFT / TPOT / ITL 的 mean / median / p99，JSON 写到 `results/`。EOS 抑制：vLLM 用 `ignore_eos`，其他 server 用 `logit_bias` 把所有 EOS 压到 -100。

```bash
# 基线：mlx-lm server（.venv-mlx）
.venv-mlx/bin/python -m mlx_lm.server --model mlx-community/Qwen3.8-27B-nvfp4 --port 8080 --decode-concurrency 32 --prompt-concurrency 8
.venv/bin/python bench/bench_serve.py --base-url http://localhost:8080 --model mlx-community/Qwen3.8-27B-nvfp4 \
    --num-prompts 100 --input-len 1024 --output-len 1024 --max-concurrency 32 --name mlxlm-27b-nvfp4-c32 --out bench/results/mlxlm-27b-nvfp4-c32.json

# 单流投机解码基线：dflash-mlx（.venv-mlx）
.venv-mlx/bin/dflash benchmark --model mlx-community/Qwen3.8-27B-4bit --draft z-lab/Qwen3.8-27B-DFlash2 --block-tokens 8 --max-tokens 512
```

每次跑分前：GPU 上没有别的东西（`mlx_lm.server` 只开一个），同一 checkpoint，同一 client。结果的 JSON 和 log 一起提交。
