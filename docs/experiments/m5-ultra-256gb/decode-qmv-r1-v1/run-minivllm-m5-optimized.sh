#!/bin/zsh
# Frozen qualified M5 Ultra benchmark configuration. Ctrl-C stops this server.
set -euo pipefail
cd /Users/ryanli/code/minivllm
unset MLX_ENABLE_TF32
export MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50
export MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0
mkdir -p .artifacts/m5-ultra-256gb/qmv-r1-local-server
exec .venv/bin/python -B bench/m5_ultra_256gb/decode_qmv/native_child.py \
  --model models/Qwen3.8-27B-nvfp4 --port 8003 --batch 4 --graph --fuse --pool 2 \
  --trace .artifacts/m5-ultra-256gb/qmv-r1-local-server/trace.json \
  --package-root .artifacts/m5-ultra-256gb/qmv-r1-source-v1 \
  --source-manifest .artifacts/m5-ultra-256gb/qmv-r1-inputs-v1/source.json
