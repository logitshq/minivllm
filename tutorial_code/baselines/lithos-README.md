# 第三条 baseline：Lithos Metal

Lithos 用于观察另一种 Apple GPU 引擎设计：预先编译的执行图、GPU 状态控制和 DSpark 投机解码。HF 仍是数学正确性参照；oMLX 与 Lithos 是外部性能对照。它们有各自的目标模型、kernel 路径和计时边界。

本机使用官方源码固定在 `5485ad6bac0a5649d66a347f3e01a655ea29d937`，distribution version 为 `0.1.2`。官方 `v0.1.2` 的预编译 wheel 缺少 M4 Max 登记和 MLX untied output head 的名字映射；因此安装脚本构建完整的固定源码。我们没有修改 Lithos 模型或 kernel。Python 为隔离的 `3.12.13`，依赖来自 SHA256 校验的官方二进制归档；源码归档也独立校验。源码构建只需已有 Command Line Tools。

固定源码归档 SHA256 为 `2b33f72e22d3b392df72cda547f27ffbc03ea0597e17acde243fb53bb5081c6f`。初期仅复制 head adapter 的实验已废弃；当前 native/runtime、Python compiler 和 Metal 源码均来自完整的官方 `5485ad6` 构建。

同一 adapter 也接受第二至十一章的本地 `Qwen3.5-0.8B-nvfp4`。把下面命令的 `--model-dir` 换成 `/Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4`，便可在同一个 0.8B checkpoint 上补充第三条基线。该目录缺少 `generation_config.json`：runner 复用教程的纯 JSON `read_eos_ids`，从嵌套模型 config 和 tokenizer 特殊 token 元数据读出 `{248044,248046}`。随后在隔离目录创建一个只含符号链接的 checkpoint 视图，补上这个 EOS 集合，避免上游只读取 text_config 中的单个 EOS。它不复制或修改 tensor、config 与 tokenizer；报告保存这条元数据设置。加 `--prepare-only` 可只验证这些 metadata、19 个输入 IDs 和 CPU weight bindings，既不启动 server，也不创建 GPU device。

```bash
cd /Users/ryanli/code/minivllm-tutorial
/Users/ryanli/code/minivllm/.venv/bin/python tutorial_code/baselines/setup_lithos.py
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --model-dir /Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6 \
  --max-context 2048 --max-new-tokens 128 --repeat 3 \
  --out .artifacts/lithos-baseline/latest.json
```

当前仅完成 CPU 准备，GPU **尚未实测**。两个 CPU 报告为 `/private/tmp/minivllm-remaining-20261014/lithos-0.8b-cpu-prepared.json` 与 `/private/tmp/minivllm-remaining-20261014/lithos-27b-cpu-prepared.json`；它们分别验证 187/498 个 NVFP4 bindings、相同的 19 个 prompt IDs 和 EOS 集合，并明确记录没有启动 server 或创建 GPU device。

重新做 CPU 准备时：

```bash
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --prepare-only --out .artifacts/lithos-baseline/0.8b-prepared.json
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --prepare-only --out .artifacts/lithos-baseline/27b-prepared.json
```

解锁 macOS、退出其他占内存的应用并取得 GPU 运行时段后，按下面顺序执行。每条命令结束都会在 `finally` 中正常终止并等待自己的 server 子进程；仅该 owned 子进程超过 15 秒未退出时才强制结束它。

```bash
# 0.8B：16-token 功能 smoke，再完整预热和三次重复测量
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --max-new-tokens 16 --repeat 1 --out .artifacts/lithos-baseline/0.8b-smoke.json
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --max-new-tokens 128 --repeat 3 --out tutorial_code/examples/chapter11-lithos-0.8b.json

# 27B：默认是已缓存且固定 revision 的目标，始终 --no-draft
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --max-new-tokens 16 --repeat 1 --out .artifacts/lithos-baseline/27b-smoke.json
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --max-new-tokens 128 --repeat 3 --out tutorial_code/examples/chapter11-lithos-27b.json
```

这里复用已缓存的 MLX NVFP4 权重，group size 16、4 bits，固定 checkpoint revision `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`。CPU 资格检查确认 498 个量化 tensor binding 均由官方 NVFP4 adapter 接受。官方 loader 对 packed U32、U8 scale、norm 值与 text/head 名称有明确转换，runner 不自行改变权重。原始 checkpoint 有约 16.05 GB；Lithos 会另外生成自己的权重 pack。这一磁盘缓存和 GPU 内存容量是不同的量。

为了先确认目标模型可运行，命令明确使用 `--no-draft`。原 minivllm 的 DFlash2 checkpoint 不能代替 Lithos 的 DSpark head；两者是不同的 drafter。

runner 通过 Lithos 实际提供的 `/v1/chat/completions` 发送第一章的同一个用户问题。本地 chat template 产生的 19 个 IDs 必须逐个等于第一章基线，服务的 prompt usage 也必须是 19。greedy、正常 EOS 停止、并发为 1。一次完整工作负载预热之后重复测量；`128` 是输出上限，实际 EOS 可能使完成 token 数更少。

报告的客户端时间从请求开始到 SSE `DONE`，包括 HTTP、服务层、prefill 和 decode；首字时间是首次非空文本抵达。权重下载、pack、首次加载与显式预热不计入重复测量。服务日志另外记录 GPU 时间和 cache 计数，runner 验证每次 `cached_prompt_tokens=0` 与 `verify_tokens=1`。服务的 prefix cache 本身开启，只是此处短输入没有命中。

Lithos 的 completion usage 来自 committed token IDs，包括终端 EOS；显示文本隐藏特殊 token。oMLX 的 usage 不包括终端 EOS，因此报告必须保留各自计数定义。

本机真实芯片与 core 数以报告中原生设备信息为准。M4 Max 的官方 profile 是 `unmeasured` fallback，关闭 automatic mixer megakernel fusion 和 tensor accelerator。这是上游 recipe 的资格标签：完成本机测量后只能报告此次工作负载的结果，不能据此宣称整套 profile 已校准。参见 [官方硬件说明](https://github.com/lithos-ai/lithos-metal#models-and-hardware) 与 [固定 M4 Max 配置](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/backends/metal/m4_max/config-40c.json)。

上游公开的约 168 tok/s 数字是 M5 Max 的 target-compute projection，假设每次八行验证接受六个 token，并排除 draft、acceptance 与 rewind。它不能填进本机端到端测试表。第十一章既有的 0.8B 数字也不能直接与此 27B 数字排速度；要对比，先补同一个 27B checkpoint 的 oMLX 与手写引擎测量，保留核心/API 边界。

脚本只启动并关闭自己的 loopback 子进程，使用随机端口和临时 API key，不修改任何 coding agent 的全局配置。安装 manifest、完整依赖记录和每次 server log 保存在 `.artifacts/lithos-baseline/`。
