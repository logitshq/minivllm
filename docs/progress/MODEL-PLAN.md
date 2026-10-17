# 模型范围与三家对照 · 2026-10-17

按用户最新要求，以Qwen3.8、GLM-5.3-Flash为两条模型主线。先做文本推理；图像、服务功能及模型能力榜不与当前速度结果混为一谈。

| 模型 | 机器与优先级 | 我们当前状态 | oMLX | Lithos固定安装 |
|---|---|---|---|---|
| Qwen3.8-27B，当前同一NVFP4 checkpoint | M4/M5；当前正确性与速度主线 | 已实现；旧完整轮次有成绩，新投机整模尚未通过 | 本机已有普通/MTP资格及计时 | 本机C1已测，当前协议C2/C4受容量限制 |
| GLM-5.3-Flash | M5 256GB；并行做架构与baseline准备 | 尚未实现，不沿用Qwen成绩 | 固定安装有glm5_next发现及vendor路由；模型实测待做 | 5485ad6固定安装缺GLM注册；记N/A，不画成0 |
| Qwen3.8-Flash-Next | M5；Qwen MoE扩展候选 | 尚未实现和实测，单独资格 | qwen4_exp vendor及兼容入口存在，checkpoint实测待做 | 此5485固定注册表也缺入口；不外推最新版 |

Qwen3.8-27B的官方结构为64层：每3层Gated DeltaNet后接1层attention，共48层GDN、16层attention；当前实现对应其文本路径。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)

GLM-5.3-Flash为320B总参数、18B激活，包含稀疏与线性attention及mHC。它需要新增模型实现，不是换一个目录名就能加载。[官方模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash)

现有MLX community混合4/5/6bit、group64版本为62 shards、203,992,076,296 bytes，约190GiB；模型卡要求256GB及以上。此体积在M4 48GiB全驻留不适用。M5也必须先检查系统、KV、prefill临时空间与实际加载峰值；权重可容纳不是长上下文运行资格。[转换权重与精度说明](https://huggingface.co/mlx-community/GLM-5.3-Flash-4bit)

Qwen3.8-Flash-Next为125B主模型、6B激活，另有51B n-gram embedding和4B MTP；有MoE、稀疏attention和gated residual。不能把它标成当前27B实现的另一个checkpoint。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

oMLX已合入GLM-5.3-Flash支持，包括DSA/MLA、MoE和线性attention等；上游支持与本机实际权重加载、输出和性能资格分开。[上游PR3184](https://github.com/jundot/omlx/pull/3184)

[固定安装的源码矩阵与SHA](model-readiness-20261017/README.md)保留详细证据。当前本机固定oMLX安装的glm5_next发现、VLM路由及vendor兼容分支已进行源码核对；Lithos固定5485ad6的architecture注册与unknown-architecture拒绝分支也已核对。该结论只针对固定安装，不声称Lithos最新版永远不支持。两台机器的实际模型/版本/量化参数、spec开关和资源状态均须分别落盘。

## 当前可比成绩

下表是M4 Max 48GiB此前完整同机轮次的三次arm中位数，Qwen3.8-27B NVFP4、19输入/32输出；不是V113之后的新测，也不是M5成绩。270请求、8640输出token，正式测量135请求、4320token。原始重复有明显波动，不以这些中位数宣布完整前沿胜出。

| 实现 | C1首文本后客户端速度 | C1整请求吞吐 | C2总吞吐 | C4总吞吐 |
|---|---:|---:|---:|---:|
| 手搓实现 | 27.95 | 25.14 | 22.63 | 38.89 |
| oMLX已测普通配置 | 29.65 | 25.40 | 20.30 | 34.45 |
| Lithos已测配置 | 29.49 | 21.77 | N/A | N/A |

单位tok/s。C1仍落后oMLX的两个指标；Lithos在客户端速度与整请求吞吐各有取舍；C2/C4的中位数领先oMLX。Lithos容量限制不是0 tok/s。数字取自[已归档的21份validated summary及延迟—吞吐重算](2026-10-17-native15/README.md)，源机控制结果为`m4-final-native15-official-attempt2/output/suite-complete.json`。同轮次主图已用实际HTTP start→DONE完整请求延迟与总吞吐重新计算，首文本后速率只保留为旧辅助指标。后续需要新的可比完整轮次，不混入最近的独立MTP筛查。

## M5最新已测范围

M5最新已推送检查点为`390d98f976cddc90e0cf5e4ec9de3c42bc7fc4ad`，其[吞吐—完整请求延迟图及逐源记录](https://github.com/logitshq/minivllm/tree/390d98f976cddc90e0cf5e4ec9de3c42bc7fc4ad/docs/optimization-guide/progress/2026-10-17-v3)基于48进程、2304正式请求：native C1为622.15ms/51.43tok/s、C4为999.27ms/126.67tok/s、C32为5286.72ms/190.44tok/s。其9个此前实测非spec外部配置中位点均被native点覆盖。外部未在该次画图时重跑，Lithos只有C1；结论仅限同机Qwen27B、19→32、target-only，不覆盖MTP/DSpark、GLM或长上下文。M4只读了Git记录，并复算其公开chartdata的聚合和支配关系，没有在本机重跑M5。

## 接下来的测量顺序

M4继续完成广播M1、低峰值草稿加载与整模事务资格，再重新跑三家端到端对照。M5并行准备GLM-5.3-Flash的固定权重和oMLX baseline，先证明实际加载、正确输出、资源余量；我们补齐其MoE/attention/cache实现后再加入性能表。Lithos只在其固定版本实际支持的模型上比较。

每个模型先固定同一权重、dtype、量化与请求定义，比较非spec路径；再单独资格并测各家兼容的spec配置。19→32继续作回归探针，新增512/4K输入、128输出和资源允许的C1/2/4，之后再扩16K及更高并发；不以短探针宣布真实长上下文已胜出。
