#!/bin/zsh
# InferenceX 式的 Pareto 数据：扫并发，记每个点的 (每用户 tok/s, 整机 tok/s)。我们的引擎 + mlx-lm 静态 batch 作对照。
set -u
cd ~/code/minivllm
R=bench/results
MODEL=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/*)
echo "=== minivllm engine sweep $(date +%H:%M:%S)"
uv run bench/throughput.py "$MODEL" --batch 1 2 4 8 16 24 32 --input-len 1024 --output-len 128 2>&1 | grep -v -i warn | tee $R/pareto-ours.txt
echo "=== mlx-lm static batch sweep $(date +%H:%M:%S)"
for b in 1 2 4 8 16 24 32; do
  .venv-mlx/bin/python -m mlx_lm.benchmark --model "$MODEL" -p 1024 -g 128 -b $b -n 1 2>&1 | grep -E "^Trial|rror|emory" | sed "s/^/b=$b /" | tail -1
done 2>&1 | tee $R/pareto-mlxlm.txt
echo "=== done $(date +%H:%M:%S)"
