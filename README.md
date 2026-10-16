# minivllm

从零手搓一个 vLLM 式推理引擎，跑 Qwen3.8-27B（nvfp4）在 Apple Silicon 上，教学用、自包含。
设计跟 vLLM V1（统一调度、paged KV、continuous batching），后端是 MLX。远程还有一份 CUDA/Triton 版，接口契约相同。

- **逐章重写教程**：[从第一章开始](docs/tutorial-v2/README.md)，每章一个 commit/tag，教学代码在 `tutorial_code/`。
- 原计划和实验记录：`docs/PLAN.md`、`docs/tutorial/`
- 引擎只依赖 `mlx`，分词用 `tokenizers`；torch/transformers 只用来生成参考答案；基线（mlx-lm、dflash-mlx）单独一个环境

## 现状（2026-10-14，27B nvfp4，M4 Max 48 GB）

| 场景 | minivllm | 对照 |
|---|---|---|
| prefill 1024 token | 248 tok/s | mlx-lm 同 checkpoint 138，affine 4bit 241 |
| 单流 decode | 25 tok/s | mlx-lm 24–29 |
| 单流 + DFlash2 投机解码（无损） | 56–62 tok/s | dflash-mlx 64 |
| 并发 8 decode 阶段 | 90 tok/s | mlx-lm server 48 |
| 并发 32 | 跑得起来，峰值 24 GB，持续 115–127 tok/s | mlx-lm server Metal OOM；其静态 batch 142（比我们快 20%） |

教程按步骤在 `docs/tutorial/00`–`06`；吞吐 vs 每用户速度的 Pareto 曲线在 `docs/pareto.html`（`uv run bench/pareto.py` 重画）。

## 环境

```bash
uv sync                     # 引擎 + 开发依赖 → .venv，锁在 uv.lock
uv run pytest -q            # 跑测试
./bench/setup_baseline.sh   # 基线环境 → .venv-mlx（mlx-lm、dflash-mlx）
```

模型：`uv run hf download Qwen/Qwen3.5-0.8B`（开发用，和 27B 同架构）、`uv run hf download mlx-community/Qwen3.8-27B-nvfp4`（目标）。
0.8B 的 nvfp4 开发版用 mlx-lm 转：见 `docs/tutorial/00-环境与基线.md`。

## 目录

```
minivllm/          引擎（只 import mlx）
  config.py        从 config.json 读模型配置
  model/           Qwen3.5 架构的前向：layers / attention / gdn / qwen3_5 / loader
tests/             pytest；tests/make_refs.py 用 HF 生成参考答案到 tests/data/
bench/             跑分 client（复刻 vllm bench serve）、基线脚本、结果
docs/              计划、教程、远程版的契约和 notes
```

## 独立实验与未采用结果

- [18：先确认优化真的更快](docs/tutorial-v2/18-先确认优化真的更快.md)：GDN/MLP输出行拼接的独立评估，两方向均STOP、未采用；没有修改当前26个引擎源码或V19 default OFF。
- [投影拼接证据与评估](docs/experiments/18-projection-packing-evaluation.md)：旧AC顺序混杂、新Battery局部配对和控制、原始SHA及独立审查收据；不列为已实现能力。

- [19：把量化矩阵乘拆开看](docs/tutorial-v2/19-把量化矩阵乘拆开看.md)：原QMM→手写R4控制→R2几何实验，27项数值通过但未过5%收益门，STOP、未采用。

- [20：沿实际实现读三条执行路径](docs/experiments/20-donor-implementation-map.md)：oMLX预处理、已有ahead、Lithos ICB与allocator策略的源码证据；仅CPU阅读，无新增性能结论。

- [21：把GDN真卷积、SiLU和原始tail合成一核](docs/experiments/21-gdn-conv-fusion.md)：synthetic严格数值与局部micro通过，host-wall中位减少12.21%；真实preconv、完整模型及E2E资格待做，原型未接入引擎。

[实验22：融合和图复用的真实 HTTP 检查](docs/experiments/22-combined-decode-screen.md)保存组合版源码和完整结果；C1交互速度+1.54%、总吞吐+0.76%，尚未超过官方前沿，默认关闭。
