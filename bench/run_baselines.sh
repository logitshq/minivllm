#!/bin/zsh
# Reduced-protocol baselines on the M4 Max. v2: memory-conscious mlx-lm settings + memory monitor.
set -u
cd ~/code/minivllm
R=bench/results
MLX=.venv-mlx/bin/python
PY=.venv/bin/python
MODEL=mlx-community/Qwen3.8-27B-nvfp4

memline() { echo "$(date +%H:%M:%S) $(sysctl -n vm.swapusage | sed 's/  */ /g') | $(memory_pressure 2>/dev/null | grep -o 'free percentage: [0-9]*%') | server_footprint=$(top -l 1 -stats pid,mem,command 2>/dev/null | grep 'python3.13' | sort -k2 -h | tail -1 | awk '{print $2}')"; }
start_server() {  # $1 = decode concurrency
  nohup $MLX -m mlx_lm.server --model $MODEL --port 8080 --decode-concurrency $1 --prompt-concurrency 2 --prompt-cache-bytes 1 > $R/server-27b-nvfp4-c$1.log 2>&1 &
  for i in $(seq 1 60); do curl -s -o /dev/null localhost:8080/v1/models && break; sleep 2; done
}
stop_server() { pkill -f "mlx_lm.serve[r]"; sleep 3; }

echo "=== idle memory: $(memline)"
for C in 32 8; do
  echo "=== mlx-lm server c=$C $(date +%H:%M:%S)"
  start_server $C
  ( while pgrep -f "mlx_lm.serve[r]" > /dev/null; do memline; sleep 20; done ) > $R/mem-c$C.log 2>&1 &
  $PY bench/bench_serve.py --base-url http://localhost:8080 --model $MODEL --num-prompts 32 --input-len 1024 --output-len 256 \
      --max-concurrency $C --name mlxlm-27b-nvfp4-c$C-short --out $R/mlxlm-27b-nvfp4-c$C-short.json 2>&1 | grep -v -i warn | tail -22
  stop_server
  echo "--- memory during c=$C run:"; sed -n '1p;$p' $R/mem-c$C.log; sort -t= -k2 -h $R/mem-c$C.log | tail -1 | sed 's/^/peak: /'
done

echo "=== mlx_lm.benchmark 4bit vs nvfp4 b=1 $(date +%H:%M:%S)"
for M in mlx-community/Qwen3.8-27B-4bit mlx-community/Qwen3.8-27B-nvfp4; do
  $MLX -m mlx_lm.benchmark --model $M -p 256 -g 128 -b 1 -n 1 2>&1 | grep -E "^Trial" | sed "s|^|$M |"
done

for B in 8 5; do
  echo "=== dflash-mlx DFlash2 block $B $(date +%H:%M:%S)"
  .venv-mlx/bin/dflash benchmark --model mlx-community/Qwen3.8-27B-4bit --draft z-lab/Qwen3.8-27B-DFlash2 --block-tokens $B --max-tokens 512 --repeat 1 --cooldown 5 2>&1 | grep -v -i warn | tail -30
done
echo "=== done $(date +%H:%M:%S)"
