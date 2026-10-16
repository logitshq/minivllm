#!/bin/zsh
set -eu
cd /Users/ryanli/code/minivllm
mkdir -p .artifacts/m5-ultra-256gb/pareto-local-server
exec env -u MLX_ENABLE_TF32 \
  MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 \
  MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
  .venv/bin/python -B .artifacts/m5-ultra-256gb/pareto-batch-launcher-v1/native_child.py \
  --model models/Qwen3.8-27B-nvfp4 --port 8003 --batch 32 --graph --fuse --pool 2 \
  --trace .artifacts/m5-ultra-256gb/pareto-local-server/trace.json \
  --package-root .artifacts/m5-ultra-256gb/pareto-batch-source-v1 \
  --source-manifest .artifacts/m5-ultra-256gb/pareto-batch-frontier-v1/inputs/source.json
