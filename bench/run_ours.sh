#!/bin/zsh
# 我们的 server 跑缩短协议（和 mlx-lm 那组一样：32 条 prompt，1024 进 256 出，并发 32 和 8）。
set -u
cd ~/code/minivllm
R=bench/results
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/*)
NAME=minivllm-27b-nvfp4
memline() { echo "$(date +%H:%M:%S) $(sysctl -n vm.swapusage | sed 's/  */ /g') | $(memory_pressure 2>/dev/null | grep -o 'free percentage: [0-9]*%') | server_footprint=$(top -l 1 -stats pid,mem,command 2>/dev/null | grep 'python3.13' | sort -k2 -h | tail -1 | awk '{print $2}')"; }
for C in 32 8; do
  echo "=== minivllm server c=$C $(date +%H:%M:%S)"
  nohup uv run python -m minivllm.server --model "$MODEL" --served-model-name $NAME --port 8000 --max-seqs $C --max-len 1536 --max-batched-tokens 512 > $R/server-ours-c$C.log 2>&1 &
  for i in $(seq 1 90); do curl -s -o /dev/null localhost:8000/health && break; sleep 2; done
  ( while pgrep -f "minivllm.serve[r]" > /dev/null; do memline; sleep 20; done ) > $R/mem-ours-c$C.log 2>&1 &
  uv run bench/bench_serve.py --base-url http://localhost:8000 --model $NAME --tokenizer "$MODEL" --num-prompts 32 --input-len 1024 --output-len 256 \
      --max-concurrency $C --eos-mode vllm --name ours-27b-nvfp4-c$C-short --out $R/ours-27b-nvfp4-c$C-short.json 2>&1 | grep -v -i warn | tail -22
  pkill -f "minivllm.serve[r]"; sleep 3
  echo "--- peak footprint:"; sort -t= -k4 -h $R/mem-ours-c$C.log | tail -1
done
echo "=== done $(date +%H:%M:%S)"
