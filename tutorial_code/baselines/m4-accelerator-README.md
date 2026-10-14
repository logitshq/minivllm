# M4 Max 的 accelerator A/B：不套整份 M5 配置

官方 `m5_max_40c` 配置不能直接用于此机。CPU 实际调用 `get_backend(...).validate_device(...)` 返回：`backend m5_max_40c requires Apple M5 Max with 40 GPU cores; device is Apple M4 Max with 40 GPU cores`。记录见 [M5 profile 拒绝](../examples/chapter11-m5-profile-rejection.json)，该检查没有创建 GPU device。40 个 core 相同不代表 GPU family 相同：本机 Apple9，M5 Max Apple10。Apple10 的 staged/split fill 配置也不能直接套给 Apple9。

本实验保持官方 `apple-m4-max-40c` / `m4_max` / Apple9 身份，threadgroups_per_core=1、attention=v1、commute_norm=True、原 NVFP4 权重及 EOS。只通过官方 `MONOLITH_ACCELERATOR=on` 打开 shared compiler 已有的 Apple9 matrix tile 路径。没有修改已安装 Lithos 源码、芯片 JSON、M5 cost table、fusion recipe 或设备校验。

代码使用完整官方源码 revision `5485ad6bac0a5649d66a347f3e01a655ea29d937`，与已归档正式基线相同。权重 pack 从现有 isolated cache 复用，没有再复制一份大 tensor。

`lithos_proof_server.py` 只在官方 `Backend.complete()` 返回后读取 live Engines 的 `program`，记录 kernel 函数、`COOP_A9`、predicated T 范围和 MSL 版本，随后写诊断日志。原 generation 与 GPU 决策不变。实际成功运行的 0.8B / 27B prefill program 分别有 97 / 321 个 `gemm_tile` op，均为 `COOP_A9=1`、MSL4，实际 T=19 落在 [1,128] 内。T=1 decode program 没有 `gemm_tile`；同一个官方开关还会选择部分单行 row kernel 和 GDN prepared 路径，因此不能将整个收益只归因于 prefill tile。证据是已完成生成的 live-program 编译、编码与有效 predicate 范围，不是每个 op 独立 GPU timestamp profile，也不是 M5 神经加速器测量。

## 实测

19 个输入 IDs、同一 checkpoint/config SHA、greedy、正常 EOS、单请求、max context2048、输出上限128，一次 full-workload warmup 后三轮测量。全部 cache0 / verify1 / SSE usage+DONE，未加载 drafter。

| 目标 | 历史 profile-off | 本次 accelerator-on |
|---|---:|---:|
|0.8B 实际完成 IDs（含 EOS）|34 /34 /34|29 /29 /29|
|0.8B 首字中位数|1.460s|0.223s|
|0.8B 请求耗时中位数|1.658s|0.354s|
|0.8B prefill GPU command-buffer 中位数|1449.8ms|214.4ms|
|27B 实际完成 IDs（含 EOS）|45 /45 /45|45 /45 /45|
|27B 首字中位数|42.781s|0.434s|
|27B 请求耗时中位数|46.193s|2.208s|
|27B prefill GPU command-buffer 中位数|42769.9ms|420.3ms|
|27B decode GPU command-buffer 中位数|3341.7ms|1766.3ms|

27B 每轮是 44 个非 EOS token，on/off 可见文本完全相同，且各自三轮稳定：它解释缓存历史词元的 Key/Value 以避免重复计算，生成完整的一句话。HTTP API 没有给出逐个 generated token ID，不能据此宣称独立验证了完整 ID 数组。

0.8B 的 on/off 可见文本不同，分别是 28 /33 个非 EOS token。本次 on 文本与已归档 oMLX 0.8B 的 28-token 文本完全一致。它生成完整句子，但关于“减少显存占用”的表述不准确；模型输出不能替代教程中的 KV cache 定义，也不是 kernel 数值正确性证明。不得把 0.8B 全请求 TPS 比例写成相同生成序列的无损加速。

off 来自已归档的历史正式运行，on 使用上述诊断 launcher，因此这不是同一时段交替、同仪器的配对 A/B。后置读取/打印会为客户端耗时增加少量 CPU 开销；已有 GPU command-buffer 计时不含这段后置诊断。这里保留原始数值与差异，不宣称这一个 prompt 足以校准整份 M4 profile 或所有模型。

session 当前 unique buffer bytes：0.8B on703,742,032，27B on17,045,438,688；这是当前 GPU session buffers，不是 RSS 或峰值系统内存。开始 memory_pressure system-free78%，结束69%。所有四个自有 server 已关闭，已知 PID 检查全部 absent，见 [比较与 cleanup](../examples/chapter11-m4-accelerator-comparison.json)。

## 复现

从教学仓库根目录执行下列命令，保持模型串行。`--accelerator profile` 使用正式 profile（此机 off）；`on` / `off` 为显式官方开关，任何继承的同名环境变量会先清除。

```bash
cd /Users/ryanli/code/minivllm-tutorial
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --workspace /Users/ryanli/code/minivllm-tutorial \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --accelerator on --max-new-tokens 16 --repeat 1 \
  --out tutorial_code/examples/chapter11-m4-accelerator-0.8b-smoke.json
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --workspace /Users/ryanli/code/minivllm-tutorial \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --accelerator on --max-new-tokens 128 --repeat 3 \
  --out tutorial_code/examples/chapter11-m4-accelerator-0.8b.json
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --workspace /Users/ryanli/code/minivllm-tutorial \
  --accelerator on --max-new-tokens 16 --repeat 1 \
  --out tutorial_code/examples/chapter11-m4-accelerator-27b-smoke.json
.artifacts/lithos-baseline/.venv/bin/python -m tutorial_code.baselines.run_lithos \
  --workspace /Users/ryanli/code/minivllm-tutorial \
  --accelerator on --max-new-tokens 128 --repeat 3 \
  --out tutorial_code/examples/chapter11-m4-accelerator-27b.json
```

本实验以 `tutorial-11-v3` / `baseline-m4-accelerator-ab-v1` 保存。正式 `tutorial-11-v2` 默认 profile-off 数据保留，独立 CPU 审阅已通过；没有修改安装包源码或芯片配置。

## 默认运行与观察开销

`run_lithos --accelerator profile` 是默认值：清除继承的 `MONOLITH_ACCELERATOR`，让官方 profile 决定（本机为 off）。`on/off` 只影响这个自有子进程的官方 override，不写安装包、环境全局设置或芯片 JSON。

当前 wrapper 通过同目录的 `lithos_proof_server.py` 启动官方服务，完整请求返回后读取 live program。即使选择 profile，后置诊断也可能增加客户端总耗时；旧 profile-off 测量没有这层诊断，不能把它们说成同仪器交替实验。编译配置里的 off 与实际 Session override on 在原始报告中分别记录。

完整原始记录：[0.8B on](../examples/chapter11-m4-accelerator-0.8b.json)、[27B on](../examples/chapter11-m4-accelerator-27b.json)、[独立算术审阅](../examples/chapter11-m4-accelerator-arithmetic.json)。官方 [M5 配置快照](../examples/chapter11-m5-profile-config.json) 的 memory_gb=48、gpu_working_set_gb=37.44、max_buffer_gb=28.08 是该 profile 的容量字段，不是本实验 GPU buffers 或进程 RSS。
