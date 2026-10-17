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

[27：少搬 state，再把真实 Metal 命令留在 ICB 里](docs/experiments/27-native-icb-pingpong.md)记录V36 S轮换、V37独立克隆负结果与V38实际ICB；该章当时测得的27.633tok/s仅属B-only探索，尚无HTTP/Pareto获胜结论。

[28：把卷积、归一化复用和state保留接到真实ICB](docs/experiments/28-native-gdn-composition.md)记录V39/V40/V44完整严格资格、V40真实绑定失败及修复、独立B-only计时；34.699ms/步、28.820tok/s尚非HTTP或Pareto获胜结论。

[29：把原生 decode 接进调度器，再测真实请求](docs/experiments/29-native-runner-http.md)验证 C1 排队、重置和回退；记录无屏障减少的负结果及真实 HTTP 电源变化 STOP，没有新的 Pareto 获胜结论。

[30：提前提交一步](docs/experiments/30-native-ahead.md)保存 FIFO 输出所有权、提前停止丢弃、12,540 项完整模型检查与未完成 HTTP 结果；保留量化矩阵乘和屏障实验的负结果。

[31：直接反馈、采样扰动与局部投影实验](docs/experiments/31-native-direct-profiling.md)区分逐步诊断与生产末尾状态证据，保存 GDN 局部负结果和 GPU 采样边界；28.826tok/s 为 B-only 引擎筛查，尚无新的 HTTP/Pareto 获胜结论。

[32：残差 carry 与四投影](docs/experiments/32-native-residual-projection.md)记录严格数值通过后的 B-only 筛查，以及 R8 的 6/6 胜却未过 5% 门槛；这些路径仍未采用，没有新的 Pareto 胜出。

归档中引用的上游源码许可见 [Third-party source notices](THIRD_PARTY_NOTICES.md)。

[33：ICB范围与整步编译](docs/experiments/33-native-range-compiled-step.md)保留 R8 在原生 ICB 下慢0.52%的结果，以及整步编译的数值资格和 B-only 筛查；没有新的 HTTP 或 Pareto 胜出。

[34：BA小投影与门控融合](docs/experiments/34-native-ba-gates-prework.md)记录720项局部检查、12,276项整模检查，以及28.693tok/s的B-only筛查；减少192条dispatch仍没有证明提速，候选默认关闭。

[35：局部优化的采用门槛](docs/experiments/35-local-probe-stops.md)保留 half2、unroll2 与 scaled-ready GDN 的数值检查和负结果；三个 micro 均未过预设提速门槛。

[36：CAP64与RMS舍入版本](docs/experiments/36-cap64-and-rms-profile.md)验证真实CAP64缓存，并保留RMS舍入差异、旧序列检查失败与新参考诊断；原生RMS版本的速度未显示提升。

[37：SIMD4、异步提交与投机解码接手](docs/experiments/37-simd4-async-and-spec-handoff.md)记录新版源码、完整正确性资格与HTTP热状态中止；整体尚未胜出。


[M4 与 M5 持续同步](docs/handoffs/m4-max-to-m5-ultra.md)：按数值／性能检查点交换已推送源码、正负结果及下一步，两个硬件域分别验证。

[38：投机事务与量化乘法](docs/experiments/38-spec-transaction-and-qmm.md)保存V108旧ID门失败、官方MTP数值资格及三对HTTP负结果、同输入逐算子诊断；187项源码身份与449份公开运行文本已归档（417份原字节、32份凭据字段脱敏view），尚无新的完整Pareto胜出。

[39：保留逐行量化乘法的结果](docs/experiments/39-verify-broadcast-m1.md)附冻结代码与前后diff；[进展图与台账](docs/progress/README.md)保存PNG/SVG、数据及复现脚本，[模型计划](docs/progress/MODEL-PLAN.md)明确Qwen3.8和GLM-5.3-Flash的机器与对照范围。

[40：整模通过后再测吞吐和延迟](docs/experiments/40-m1-verify-and-bounded-loader.md)保存 V117 实际正确性结果、V120/V122 资源中止及代码；当前主图为 [M4 throughput–latency Pareto](docs/progress/2026-10-17-m4-pareto/README.md)。
