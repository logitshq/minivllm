# 模型范围与三家对照 · 2026-10-17

按用户最新要求，以Qwen3.8、GLM-5.3-Flash为两条模型主线。先做文本推理；图像、服务功能及模型能力榜不与当前速度结果混为一谈。

| 模型 | 机器与优先级 | 我们当前状态 | oMLX | Lithos固定安装 |
|---|---|---|---|---|
| Qwen3.8-27B，当前同一NVFP4 checkpoint | M4/M5；当前正确性与速度主线 | 已实现；V117 投机整模严格通过，端到端性能尚未取得 | 本机已有普通/MTP资格及计时 | 本机C1已测，当前协议C2/C4受容量限制 |
| GLM-5.3-Flash | M5 256GB；并行做架构与baseline准备 | 尚未实现，不沿用Qwen成绩 | 固定安装有glm5_next发现及vendor路由；模型实测待做 | 5485ad6固定安装缺GLM注册；记N/A，不画成0 |
| Qwen3.8-Flash-Next | M5；Qwen MoE扩展候选 | 尚未实现和实测，单独资格 | qwen4_exp vendor及兼容入口存在，checkpoint实测待做 | 此5485固定注册表也缺入口；不外推最新版 |

Qwen3.8-27B的官方结构为64层：每3层Gated DeltaNet后接1层attention，共48层GDN、16层attention；当前实现对应其文本路径。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)

GLM-5.3-Flash为320B总参数、18B激活，包含稀疏与线性attention及mHC。它需要新增模型实现，不是换一个目录名就能加载。[官方模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash)

现有MLX community混合4/5/6bit、group64版本为62 shards、203,992,076,296 bytes，约190GiB；模型卡要求256GB及以上。此体积在M4 48GiB全驻留不适用。M5也必须先检查系统、KV、prefill临时空间与实际加载峰值；权重可容纳不是长上下文运行资格。[转换权重与精度说明](https://huggingface.co/mlx-community/GLM-5.3-Flash-4bit)

Qwen3.8-Flash-Next为125B主模型、6B激活，另有51B n-gram embedding和4B MTP；有MoE、稀疏attention和gated residual。不能把它标成当前27B实现的另一个checkpoint。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

oMLX已合入GLM-5.3-Flash支持，包括DSA/MLA、MoE和线性attention等；上游支持与本机实际权重加载、输出和性能资格分开。[上游PR3184](https://github.com/jundot/omlx/pull/3184)

[固定安装的源码矩阵与SHA](model-readiness-20261017/README.md)保留详细证据。当前本机固定oMLX安装的glm5_next发现、VLM路由及vendor兼容分支已进行源码核对；Lithos固定5485ad6的architecture注册与unknown-architecture拒绝分支也已核对。该结论只针对固定安装，不声称Lithos最新版永远不支持。两台机器的实际模型/版本/量化参数、spec开关和资源状态均须分别落盘。

## 优化目标与入图条件

正确性是进入候选集合的硬条件，不是Pareto坐标，也不与速度交换。只有通过该模型/路径既定的输出、缓存和调度正确性检查的配置，才进入端到端性能比较；未通过整模检查的新投机候选不入图。局部算子一致性、源码检查或内存加载通过，仅用于研发诊断，不算Pareto改善。

主图横轴为完整HTTP请求延迟p50（越小越好），纵轴为总输出吞吐（越大越好）。在相同机器、权重/量化、输入输出长度及资源条件下，一个配置若两项均不差且至少一项更好，就支配另一个配置。目标是推进这些不被支配的实测点；TTFT、ITL、内存和局部kernel时间作为辅助诊断。基准配置名称、正确性合同、三次重复及负结果随数据保留。

## 当前可比成绩

[查看M4 throughput–latency Pareto主图](2026-10-17-m4-pareto/README.md)。

下表是M4 Max 48GiB此前完整同机轮次的三次arm中位数，Qwen3.8-27B NVFP4、19输入/32输出；不是V113之后的新测，也不是M5成绩。270请求、8640输出token，正式测量135请求、4320token。原始重复有明显波动，不以这些中位数宣布完整前沿胜出。

| 实现 | 并发 | 完整请求延迟 p50（ms） | 总吞吐（tok/s） | 联合Pareto前沿 |
|---|---:|---:|---:|---|
| 手搓实现 | 1 | 1273.09 | 25.14 | 否 |
| 手搓实现 | 2 | 2799.68 | 22.63 | 否 |
| 手搓实现 | 4 | 3291.09 | 38.89 | 是 |
| oMLX已测普通配置 | 1 | 1259.77 | 25.40 | 是 |
| oMLX已测普通配置 | 2 | 3115.39 | 20.30 | 否 |
| oMLX已测普通配置 | 4 | 3640.56 | 34.45 | 否 |
| Lithos已测配置 | 1 | 1470.18 | 21.77 | 否 |

在这组配置中，只有oMLX C1和本方C4没有被其他点同时以更低延迟和更高吞吐覆盖。本方C2被本方C1支配；不能因为C2比oMLX C2快，就把它当作前沿进步。Lithos C2/C4为容量限制，没有可画的性能点。

延迟由实际HTTP start→DONE计算，吞吐为组输出token总数/完整请求窗口；请求p50→每进程三组中位数→三个fresh进程中位数。数据见[21份聚合记录及时间戳重算](2026-10-17-native15/README.md)。该几何前沿只描述旧轮次的配置中位数；原始热状态和重复波动保留，不声称统计显著性或新候选已经领先。

## M5最新已测范围

M5 此前检查点 `390d98f976cddc90e0cf5e4ec9de3c42bc7fc4ad` 的[吞吐—完整请求延迟图及逐源记录](https://github.com/logitshq/minivllm/tree/390d98f976cddc90e0cf5e4ec9de3c42bc7fc4ad/docs/optimization-guide/progress/2026-10-17-v3)基于48进程、2304正式请求：native C1为622.15ms/51.43tok/s、C4为999.27ms/126.67tok/s、C32为5286.72ms/190.44tok/s。其9个此前实测非spec外部配置中位点均被native点覆盖。外部未在该次画图时重跑，Lithos只有C1；结论仅限同机Qwen27B、19→32、target-only，不覆盖MTP/DSpark、GLM或长上下文。M4只读了Git记录，并复算其公开chartdata的聚合和支配关系，没有在本机重跑M5。

本轮实际 fetch 到 `c01624d43ddb51415cd6971353d9cac6a313fe5f`。较新的 [E020 同轮 C1 对照](https://github.com/logitshq/minivllm/tree/9f1e331a7afdb31889e7f007c9eef21ecde5b8d9/docs/optimization-guide/progress/2026-10-17-v4)报告：native 716.10 ms / 44.69 tok/s、oMLX 普通 679.32 / 47.11、oMLX MTP 752.06 / 42.55、Lithos 普通 812.92 / 39.36、Lithos DSpark 741.96 / 43.13。15 个独立进程、45 条正式请求；该轮最快是 oMLX 普通配置。M4 已读其源码、设置和报告，没有独立重审全部原始测量，不能继续把旧轮次优势当作当前全面领先。两个轮次分别保留，不混成一条前沿。

该检查点报告 GLM-5.3-Flash 固定 revision 的 62 shards 下载已启动；尚无加载或性能资格。

## 接下来的测量顺序

M4 的 V117 已完成 A/B 各 96 个旧 token ID、49 次实际事务及同提交位置缓存/logits 严格检查。V120 两次投机服务和 V122 小缓存配置均在首请求前因内存压力中止；只有 V120 普通路径的一组 1250.37 ms / 25.59 tok/s，不能据此宣布配对提速。下一步独立测 HTTP 发送选项，并继续排查投机服务的加载资源；已有前沿不增加未测点。M5并行准备GLM-5.3-Flash的固定权重和oMLX baseline，先证明实际加载、正确输出、资源余量；我们补齐其MoE/attention/cache实现后再加入性能表。Lithos只在其固定版本实际支持的模型上比较。

每个模型先固定同一权重、dtype、量化与请求定义，比较非spec路径；再单独资格并测各家兼容的spec配置。19→32继续作回归探针，新增512/4K输入、128输出和资源允许的C1/2/4，之后再扩16K及更高并发；不以短探针宣布真实长上下文已胜出。
