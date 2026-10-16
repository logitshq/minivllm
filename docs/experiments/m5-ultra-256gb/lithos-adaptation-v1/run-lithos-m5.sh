#!/bin/zsh
set -euo pipefail
export MONOLITH_ACCELERATOR=on
export MLX_MAX_OPS_PER_BUFFER=50
export MLX_MAX_MB_PER_BUFFER=50
export MLX_BFS_MAX_WIDTH=20
export MLX_METAL_FAST_SYNCH=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /Users/ryanli/code/lithos-metal
exec .venv/bin/python -B -m monolith.serve \
  --model /Users/ryanli/code/minivllm/models/Qwen3.8-27B-nvfp4 \
  --no-draft \
  --pack /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/lithos-adaptation-v1/packs \
  --local-files-only --host 127.0.0.1 --port 8002 \
  --max-context 2048 --keep-warm 0 --no-warmup "$@"
