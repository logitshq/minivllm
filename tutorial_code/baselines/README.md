# 两条 baseline：HF 验证正确性，oMLX 比较性能

HF 是我们逐步复写模型数学时的参照；oMLX 是已经有调度、缓存和原生 kernel 的性能对照。两个用途在教程中分开写。

| 基线 | 主要问题 | 当前可运行入口 |
|---|---|---|
| HF Transformers | 同样输入与权重，手写计算是否正确？ | [generate.py](../generate.py)、每章的组件对照脚本 |
| oMLX | 优化后的 Apple Silicon 引擎能做到什么速度、内存与并发？ | [run_omlx.py](run_omlx.py) |

第一章的 HF 循环设置 `use_cache=False`，方便观察每轮重算的过程。它是教学和正确性参照，不能代表 HF 开启 cache 后的性能。比较性能时，手写引擎与 oMLX 使用相同的 MLX checkpoint 和相同测试条件。

oMLX 的整体架构、continuous batching 与分层缓存见 [官方说明](https://github.com/jundot/omlx)。本地固定为官方 `0.7.1.dev1` 的 Python 3.13 预编译 wheel，MLX 固定为 `0.32.3`。所有依赖记录在 [omlx-requirements.lock](omlx-requirements.lock)，官方 wheel 使用 SHA256 校验。

## 1. HF baseline 已在第一章跑通

```bash
cd /Users/ryanli/code/minivllm-tutorial
/Users/ryanli/code/minivllm/.venv/bin/python tutorial_code/generate.py
```

它加载指定 revision 的 `Qwen/Qwen3.5-0.8B`，以 fp32 在 MPS 上运行，自己写 greedy 循环。参考结果保存在 [chapter01-trace.json](../examples/chapter01-trace.json)，生成 8/12 个 token 的共同前缀已经验证。

每章进行组件重写时，先比较同一份输入、同一份权重和同一精度下的数值与 token。现有第二章的完整验证代码固定在 `tutorial-02-v2`，查看固定版本：

```bash
git show tutorial-02-v2:tutorial_code/check_layers.py
```

## 2. oMLX 使用独立环境

当前机器已经安装好。重新准备环境时，从已有的 Python 3.13 环境运行：

```bash
/Users/ryanli/code/minivllm/.venv/bin/python tutorial_code/baselines/setup_omlx.py
```

脚本将依赖安装到 `.artifacts/omlx-baseline/.venv`，不改变第一章的环境。固定 wheel 包含预编译 native kernel，因此当前机器不需要安装完整 Xcode。

安装脚本检查 Qwen prefill 和 decode kernel 可加载。可加载表示运行时具备该实现；某个具体形状实际选择哪条 kernel 路径，需要进一步 profile。

## 3. 跑通同一条输入

```bash
/Users/ryanli/code/minivllm/.venv/bin/python tutorial_code/baselines/run_omlx.py \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --max-new-tokens 8 \
  --out .artifacts/omlx-baseline/latest-smoke.json
```

脚本会启动自己的本地前台子进程，指定独立的设置和模型目录，完成请求后关闭该进程。模型目录用符号链接复用已有权重，不再下载一份。

当前 runner 的设置：单请求、greedy、无投机解码、关闭 oMLX 的跨请求分页缓存；先用与正式请求相同的 `max_tokens` 上限加载权重并预热，报告同时保存预热实际输出数。模型仍可在 EOS 处提前结束。decode 内部仍然使用 KV/GDN 状态，`--no-cache` 不会把它变成第一章的无 cache 循环。

历史 [omlx-smoke.json](omlx-smoke.json) 使用 2-token 预热与 8-token 正式请求，保留原始记录。第十一章的重复测量使用当前完整工作负载预热。

为了固定输入，脚本使用第一章的 19 个 IDs，还原含特殊标记的文本，通过 `/v1/completions` 发送；验证本地再编码和服务返回的 prompt token 数都为 19。

HTTP 在这里仅用于驱动外部 baseline，教程的核心引擎仍以 token IDs 为输入。客户端耗时包含 HTTP/SSE 与服务层开销，不能标成纯模型 decode 吞吐。

每个正式流保存 `finish_reason`，并确认服务返回的 `cached_tokens=0`。oMLX 的 `completion_tokens` 排除终端 EOS；我们的核心引擎交付的 ID 数包含该 EOS，因此核心报告另外给出排除终端 EOS 的 `logical_completion_tokens`。速度字段各自沿用报告声明的计数方式；不能靠相同输出上限推定两边实际输出数相同。

## 4. 这次已经验证的内容

完整记录见 [omlx-smoke.json](omlx-smoke.json)：

- 固定版本可安装，五组 native kernel 报告可加载。
- 同一条 19-token 输入完成生成，返回 8 个输出 token，流包含 usage 和 DONE。
- 本次请求的 `cached_tokens=0`。
- 安装环境、设置和临时服务进程均与 HF 教学环境分开。

这次是短请求的功能验证。记录里保留了客户端时间和 oMLX 的服务端统计，但不足以代表持续吞吐或 27B 性能。

HF 此处是原始权重转 fp32，oMLX smoke 使用 NVFP4 转换权重。权重表示和数值路径不同，本次不要求两个输出文本完全相同，也不把二者相除作为优化收益。

## 5. 后续章节怎样使用两个 baseline

| 章节阶段 | HF 侧要展示 | oMLX 侧要展示 |
|---|---|---|
| 基础组件、Attention、GDN | 原始计算 → 手写对应计算；同输入误差 | 说明对应优化路径，暂不据组件测试宣称整体提速 |
| 完整模型与单流生成 | 固定 prompt 的 logits/greedy 对齐 | 同一 MLX checkpoint 的单流性能 |
| cache、调度、batch | 验证保存状态与合批前后结果 | 同长度、并发、输出长度下的吞吐、TTFT、内存 |
| 投机解码 | 相同采样规则下的目标生成结果 | 分别比较普通 decode 和投机模式，记录 drafter 与接受率 |

正式性能记录至少包括：checkpoint、量化格式、运行版本、输入/输出 token 数、采样与 EOS 规则、cache 命中、并发、预热、重复次数、原生 kernel 状态和内存。单流、批量、纯 decode、端到端测量分开标记。

本次新增 baseline 工具，不改变 `generate.py` 的 HF 前向，也不修改第一、二章已经固定的代码 tag。

## Lithos 的独立环境

第三条外部 baseline 已在隔离 Python 3.12 环境准备好；完整官方固定源码构建、CPU checkpoint 资格、EOS 元数据视图与 API 计时说明见 [Lithos README](lithos-README.md)。CPU 准备通过与实际 GPU 测量分别记录。当前已安装，并在应用清理后完成 0.8B/27B target-only 真实 GPU 测量；详见第十一章扩展版与各自 JSON。

本机 M4 官方 accelerator-on 的独立实验已记录，见 [实验说明](m4-accelerator-README.md)；完整 M5 配置被官方设备检查拒绝，未绕过验证。推荐第十一章阅读版本为 `tutorial-11-v3`，默认 profile/off 与原 `tutorial-11-v2` 记录保留。
