# E017：给 Lithos 加上真实 DSpark 强对照

2026-10-17；状态：下载、原生 pack、两臂实际 engine 短请求资格与独立 CPU 审计均通过。
下面保留准备阶段的设计与扩展计划，已执行的范围及结果见文末；尚未形成性能排名。

此前 Lithos 对照使用 `--no-draft`。要判断最快单用户推理，必须再测它实际支持的投机路径。
这里固定同一个 target checkpoint，显式选择 DSpark；已有 DFlash2 权重不能替代 DSpark。

## 固定的来源与输入

本机 Lithos commit `223658090ae549e59220bdec459957cbdad7926a`，包含此前 M5 Ultra 原生
identity 适配；上游基线 `5485ad6bac0a5649d66a347f3e01a655ea29d937`。工作树须保持 clean。
target 沿用 `mlx-community/Qwen3.8-27B-nvfp4` 的本地 snapshot
`5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`，不替换为 NVIDIA mixed-precision checkpoint。

[官方 DSpark checkpoint](https://huggingface.co/LithosAI/Qwen3.8-27B-DSpark-NVFP4/tree/b169bc4c8dc805a9e6178fa82f6a7b84b169a457)
固定 revision `b169bc4c8dc805a9e6178fa82f6a7b84b169a457`，三个 safetensors 共
1,136,258,502 bytes。metadata 的 shard LFS SHA 与 conversion 声明一致；下载之后仍须
对实际权重逐文件算 SHA，远端声明不等于本地字节检查。

[配置](https://huggingface.co/LithosAI/Qwen3.8-27B-DSpark-NVFP4/blob/b169bc4c8dc805a9e6178fa82f6a7b84b169a457/config.json)
定义五层、hidden 5120、vocab 248320、tap `[5,19,33,47,61]`、七个 proposal、Markov rank 256、
YaRN factor 32、原始位置 8192，且没有 sliding window。
它与我们的 DFlash2 block/selector/窗口合同不同。运行时必须保留 checkpoint 自己的 YaRN。
shard index 不含 embedding 或 lm_head；Lithos 会共享选中 target 的这两个模块。
37 个矩阵已为 NVFP4，Markov W1 与辅助参数保留源精度。

这些配置形状与固定 target 静态匹配，但最终 pairing 要由实际 bind、token 和状态资格证明。
上游只给 NVIDIA NVFP4 与 MLX affine4bit 两个 27B ID 配了自动 catalog。
本地 MLX NVFP4 路径不会自动匹配，必须显式 `--draft`，不能只看服务成功启动。

## 可复用和必须重新验证的部分

此前 M5 Ultra identity 为 Apple M5 Ultra、80 cores、Apple family 10；backend 的
`validate_device` 逐项检查，不能伪装为 Max。之前的有限 scale、tensor tile、GEMM kernel
资格可作依赖，但不证明 DSpark、八行 verify 或 accepted-prefix GDN 状态。

目标加载器已有 MLX tensor name/数值约定适配，之前同 checkpoint target-only 已跑过。
本次仍固定 config/index/全部权重与 source hash，并验证新 pack 的 tensor formats、layout
及 cache identity。`--draft-quantization none` 表示保留 checkpoint 已有 NVFP4；它不会
恢复 BF16，也不会新增 requantization。pack 时应禁止 CPU quantize/dequantize 并检查可成功
直接打包；只有真实本地结果能证明这一点。

现有 target pack capacity 是 2048。七 proposals 的 serving `capacity=max_context+6`，
因此 `max_context=2048` 需要 **2054**，旧 pack 不合格。使用新的 cache root 重建，记录
新的 pack SHA 和磁盘成本；不降低 context 或修改旧 manifest 规避检查。

当前 M5 Ultra backend 的 `serving_recipes` 为空，初始 DSpark 将使用共享 fallback。
不能把这次初始路径称为最佳调优 Lithos。Max 的已测 27B recipe 选择器还要求 target 同时有
NVFP4 与 FP8，而本 target pack 全是 NVFP4，不能直接移植整份 mixed-target recipe。
后续可独立试只含 draft 部分的显式 `--kernel-config`，保留真实 Ultra identity 与原 target；
需要新建假设、源码/recipe pin、真实 fused-route 记录和全数值资格。不得为了 recipe 偷换权重。

## 最少准备与启动命令

CPU-only metadata 脚本不导入 MLX、Metal 或模型 loader，不取权重。它记录官方 immutable
revision 的 JSON、README SHA 与来源、关键源码 hash、静态 pairing 和未来启动 argv。
输出目录必须全新：

```sh
cd /Users/ryanli/code/minivllm
.venv/bin/python -B bench/m5_ultra_256gb/lithos_spec/prepare_metadata.py \
  --out bench/m5_ultra_256gb/lithos_spec/metadata-v1
```

准备阶段的固定 revision 下载命令如下；实际执行已由 `download.py` 完成，保留全部 shard/config/index SHA。
只需 JSON 和 safetensors，不需要 remote Python code：

```sh
/Users/ryanli/code/lithos-metal/.venv/bin/python -c \
 'from huggingface_hub import snapshot_download; print(snapshot_download("LithosAI/Qwen3.8-27B-DSpark-NVFP4", revision="b169bc4c8dc805a9e6178fa82f6a7b84b169a457", allow_patterns=["*.json", "*.safetensors"]))'
```

准备完成后使用 metadata `complete.json` 中完整 argv，由主线程既有 guarded owner 和已授权
暂停/恢复 lease 执行。核心 CLI 为：

```sh
cd /Users/ryanli/code/lithos-metal
MONOLITH_ACCELERATOR=on HF_HUB_OFFLINE=1 .venv/bin/python -B -m monolith.serve \
  --model /Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6 \
  --draft LithosAI/Qwen3.8-27B-DSpark-NVFP4 \
  --draft-revision b169bc4c8dc805a9e6178fa82f6a7b84b169a457 \
  --draft-kind dspark --draft-quantization none --draft-block-size 7 --draft-sampling argmax \
  --max-context 2048 --prefill-chunk-size 128-exact \
  --pack /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/lithos-spec-packs-v1 \
  --local-files-only --host 127.0.0.1 --port 18017 --keep-warm 0
```

此命令会初始化 GPU/编译，必须在 GPU 队列取得所有权后执行，不能与性能实验并跑。

## GPU 前必须固定的验证计划

1. 固定上述 target/draft bytes、packing、真实 hardware profile 与 compile options；保存实际
   `ServingAssets.options`、选择的 recipe、编译 program/kernel 名称、八行 target verification
   及 DSpark Markov 路径。HTTP `verify_tokens=8` 只证明配置，不能单独证明已执行路线。
2. 初始目标为 greedy、正常 EOS、同 prompt IDs、同 token cap、同 tokenizer。先用短数学、
   代码、中文和现有 19→32 workload；再测 127/128/129 的 prefill 边界与长度 128 的输出。
   同 Lithos target-only 对照 raw token IDs；拿到 raw IDs 用低层 `Session.generate` 的
   `Generation.tokens`，不要把 HTTP decode/re-tokenize 当成真实原始 token。
3. 保存每轮 `accepted`、`committed`、`verify_len`、correction/bonus 与停止原因。至少覆盖
   零接受、部分拒绝、全接受、EOS 和输出上限。正常输入未触发某分支就明确缺失，并在独立
   diagnostic 中安排强制候选；不能把 diagnostic 的接受率或耗时混进性能。
4. 对拒绝后的 target attention/GDN recurrent/conv state 及 draft context 的逻辑长度做
   accepted-prefix replay。已有 0.8B synthetic rollback 测试是覆盖参考，不是此 27B
   权重、八行 verify 和本机 runtime 的证明。若块浮点顺序导致 greedy 差异，保留首处分歧、
   logits 和状态差异，暂停“严格无损”资格，不静默放宽阈值。
5. 跨 engine 数学实现仍可能有浮点差异。质量报告同时区分同 Lithos greedy 合同与
   minivllm/oMLX 跨 engine 输出结果；只共享 target 权重不能自动推出逐 token 一样。
   尚未具备可比较质量证据时不宣称新的 Pareto 支配。

## 性能计划

资格通过后才跑 C1 新进程轮转：Lithos target-only、Lithos DSpark、当前 minivllm，
同样本三次 fresh process，顺序轮换，每进程固定 warmup 与三次测量，全部原始样本保留。
初始长度/输出格子固定 `19→32` 与真实任务三 prompts→128；prompt cache 不得产生命中，
通过每请求 `cached_prompt_tokens=0` 证据确认。保存冷启动/编译与 warm request 两类时间。

同时报告 client TTFT、可见输出 token/s、完整 wall time、GPU decode time、真实产出、
accepted proposals 与 committed tokens/round。Lithos `Generation.ms_per_token` 分母可能
包含请求上限之外已经提交的 token，因此不直接当客户端输出吞吐。保留原始计数并分别算。
server 当前拒绝同时请求，不能从 C1 推导 C32，也不能把排队请求当实际 GPU batch。
初筛通过标准是三次 process 中 DSpark 对本引擎 target-only 的端到端 wall 改善稳定，
所有质量合同成立；是否胜过我们的实现按全样本报告，失败同样归档。

## E017 engine资格实际入口（运行前）

固定revision的DSpark已通过snapshot解析与三shard SHA/大小校验；download-v1记录
1,136,258,502 bytes，尚不代表模型资格。`qualify.py`实际调用serving.prepare、
assets.options与load_session，在两个独立进程各跑三次原始19→32请求。普通臂明确no_draft，
DSpark固定gamma7；正常EOS、prefix_cache=False、无自动Max recipe。两臂从各自新pack目录
创建2048/2054容量，由原始pack逻辑处理，不替换目标量化权重。

观察器保存真实编译Program和kernel SHA、accept/conf二进制日志、StepState与Generation。
独立compare再解析accept日志，核对actual route/验证长度/计数及fresh普通控制的192个
rawIDs。最后一个GPU块可能多提交，不能把它当作返回的输出数；全部实际工作照样记录。
释放session engines/引用并由guard wait/reap证明进程结束。9项CPU腐败/CLI检查通过。
此阶段只证明实际engine路由和短输入输出一致，不证明全部状态、HTTP或性能。


## E017 实际结果与独立归档

[不可变证据包](../../experiments/m5-ultra-256gb/E017-lithos-dspark-v1/README.md) 的 CPU 审计通过：
两个 fresh process（none / DSpark）各运行三次同一个 19→32 raw prompt，六次共 **192 个 raw token IDs**
完全一致。这是同 Lithos 普通 greedy 的短请求资格，尚未覆盖上文规划的三任务、长输出或全状态。

每次 DSpark 实际 14 个 decode rounds，共验证 98 proposals、接受 17、提交 31 decode tokens；
prefill 另产出 1 token，返回数为 32。每次接受分布为 0×7、1×2、2×4、7×1，已有零接受、
部分拒绝与全接受；本样本没有 EOS，也没有最后一块超预算，不能宣称这两个分支已覆盖。
普通臂每次为 31 decode rounds。减少轮数不是端到端加速结论。

实际 Program 独立重建的结构为普通 decode 535 ops、bound 1；DSpark decode 586 ops、
dynamic bound 8、43 个 draft ops，包括五层 backbone、Markov W2、verify_select、accept_scan。
四份完整 Program 与所有 kernel source SHA/macros、accept/conf 原始二进制日志均重新解析验证。
EOS 宏实际为 `(tok==248044||tok==248046)`，与 model config 一致，没有禁用 EOS。

原生 Apple M5 Ultra 80 cores / Apple10、`m5_ultra_80c`、空 recipes 保持不变。
target packs capacity 为 2048 / 2054，draft pack 为 2054，Program 有效 target context 均为 2048。
两个 target packs 都有 258 个 NVFP4 slabs；draft pack 为 27 个 NVFP4 + 1 个 BF16 slab，
这是合并后的 packed slabs，不能与 checkpoint 的矩阵个数混为一谈。未请求 requantization，
实际 layout 为 interleaved16/block。config/index/非权重 metadata 与 pack manifest 已独立复核；
模型/pack 大 payload 的前后 SHA 由运行记录保存，审计没有再次读取三十余 GiB 权重。

19 个资源快照均 pressure=1、thermal=0，swap=268.81 MiB、Swapouts=303419 稳定。
两个模型子进程 exit 0，guard wait/reap 与严格不存在检查通过；授权暂停的 omlx-server PID 74384
身份匹配并恢复至 S。session engines、buffers 和 last-engine 引用均释放。

归档含完整 run/owner/lease/jobs/download/comparison/metadata、221 个实际 Lithos source/native binary、
3 个被测 harness、模型 metadata 和原始 pack manifest；327 个 archive members 的大小与 SHA 全部通过。
这些资格请求有观察器、哈希、Program 序列化与新 pack 创建，不用于速度或 Pareto 排名。
下一步应运行独立的无观察器官方 HTTP/engine 性能流程，并继续补足质量与状态覆盖。
