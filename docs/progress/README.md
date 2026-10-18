# 优化进度与图

主目标是**在通过正确性检查的配置集合中，优化throughput vs完整请求latency的Pareto前沿**。正确性是入图条件，不是优化轴；局部数值PASS不能算性能进步。主图必须显示实际吞吐、请求起点到DONE的延迟及不被支配的配置，保留被支配点供比较。

每个有实际结果的检查点保存：教程的前后代码对比、代码版本及SHA、全部正负结果、当前进度台账和适用的图。通过源码审查、局部数值、整模数值及端到端性能是不同阶段，不用一个“完成百分比”代替。

已有的日期快照保持原样；同日后续结果另存检查点，Git commit保留演进。每次显著数值／性能结果之后commit/push、fetch M5交接，并更新本方交接。性能图只使用同机、同权重、同负载与相符资源条件的结果；M4/M5、局部/整模、不同电源分开。不同轮次保留独立数据与来源；同条件的跨轮图只能作描述性比较，不能计算配对增益或宣称同轮验证。

- [六篇十八节：保留原文的教程整理](../quality/2026-10-17-tutorial-restructure.md)：保留15基础章，新写18优化短章，30个原实验编号/34份原文全部建立映射并核对原字节；代码、旧tag和实验数据保留。收到M5 `0b43e80`对V133的回执及E024正式结果更新；本方无新GPU测量。
- [教程质量审查与M5同步](../quality/2026-10-17-tutorial-review.md)：01—15全文审读、实验篇重点复查，修正概念、版本导航、坏链接与旧状态；本轮只改文档。实际读取M5至`e09993a`，已收到其合入第47章的回执；oMLX D512正式点已补齐，E024当时仍无正式HTTP提速点。
- [近期论文独立精读](../experiments/47-recent-paper-reading.md)：root与三个agent核对9月底至10月的五篇论文；新增prefill频率实验设计、检索接入边界及QBA模拟限制。仅研究与教程，无GPU或新性能点。
- [M4 / M5 进度核对（读取至7ce72e5）](2026-10-17-m5-reply-update.md)：当时Lithos D512正式完成，oMLX正式点仍缺，图复用通过服务路径数值检查但尚无HTTP收益。M4研究后没有新增实现或成绩。
- [下一轮研究与 M5 协作提议](../experiments/46-next-ideas-from-papers.md)：2026-10-17 核对新论文、作者代码和本方已有机制，提出 decoded-W 复用、混合请求调度与真实成本投机策略；M5随后确认已读详细研究包，反馈与状态见上方最新核对。
- [最新三家各 C1/C2/C4 对比图](../experiments/45-lithos-queued-concurrency.md)：新增 Lithos 官方 FIFO 排队接口的九次独立加载，吞吐约 21 tok/s，延迟随并发增加。native/oMLX 为 V128，Lithos 为 V130；两轮同机、同模型、19→32、AC，明确标注跨轮描述性比较。
- [此前同轮 M4 完整吞吐—延迟前沿](../experiments/44-fresh-full-throughput-latency.md)：21 次独立加载、7 个配置，同轮 C1/C2/C4 本方点组成联合前沿，覆盖本轮 oMLX/Lithos 已测点。仅限 Qwen3.8-27B NVFP4 的 19→32 短负载。
- [此前 M4 三家 C1 吞吐—延迟图](../experiments/43-fresh-c1-throughput-latency.md)：九次 fresh 加载的 C1 切面，独立保留，不与其他批次拼接。
- [此前完整 M4 throughput–latency Pareto 图](2026-10-17-m4-pareto/README.md)：正确性为先决条件；联合前沿、各家前沿与被支配配置均标明。
- [M4此前完整三家对照的延迟—吞吐图](2026-10-17-native15/README.md)：旧轮次重新核算真实HTTP完整请求时长，保留21个arm、22份聚合原件与全部白名单时钟；不是新候选跑分。
- [2026-10-17 图与台账](2026-10-17/README.md)：oMLX none/MTP三对HTTP计时，以及同输入Linear局部数值图。PNG/SVG、已审数据和生成脚本均保存。
- [当前模型与三家对照范围](MODEL-PLAN.md)：Qwen3.8与GLM-5.3-Flash。
- [M4→M5交接](../handoffs/m4-max-to-m5-ultra.md)：已实际推送、接收的源码事实与下一步。
- [39：广播M1教程](../experiments/39-verify-broadcast-m1.md)与[固定代码快照](../code-versions/39-broadcast-m1/README.md)。

[40：整模通过后再测端到端性能](../experiments/40-m1-verify-and-bounded-loader.md)记录 V117 严格整模通过、V120 两次加载资源中止与 V122 小缓存诊断。普通路径单组测到 1250.37 ms / 25.59 tok/s，投机路径尚无请求性能；不混入旧前沿；该阶段尚未取得完整 Pareto 胜出，最新结论见第44章。

native15快照README保留作者在源机运行的命令。其`collect_rows.py`依赖未公开的完整arm报告与原目录布局，离机不能把这条命令当作完整采证重放。公开时间戳白名单和聚合表足够复算图中延迟与吞吐；重新绘图先复制快照到新目录，并准备Matplotlib及适用中文字体。`generate_figure.py`从脚本所在目录读数据、写图，当前字体路径为macOS STHeiti；不要直接在冻结快照中覆盖原图。

[41：TCP_NODELAY HTTP 对照](../experiments/41-native-http-nodelay-pilot.md)新增首对负结果：OFF 为 1227.177 ms / 26.0761 tok/s，ON 为 1230.205 ms / 26.0119 tok/s。按冻结计划停止，未采用；两臂同机 AC、thermal 0、pressure 1。该内部单因素筛查单独保存，不替换上面的三家对照前沿。

[42：投机与普通路径的新短请求对照](../experiments/42-startup-warm-short-request.md)：V123 两臂完成，raw32 与资源检查通过；spec 吞吐为 ordinary 的 48.805%，不采用。新图和负结果单独归档，不与旧三家前沿混合。

[43：新的三家 C1 对照](../experiments/43-fresh-c1-throughput-latency.md)：V125 完成 N–O–L / O–L–N / L–N–O 三轮。按 fresh-arm 中位数，本方 1230.160 ms / 26.0129 tok/s，oMLX 1271.841 / 25.1604，Lithos 1471.369 / 21.7485。相对 oMLX 吞吐高 3.388%、完整请求延迟低 3.277%；三个配对方向一致，尚不声称统计显著。C1 固定输出 32 tokens，两轴互为倒数；当时缺少的同场 C2/C4 已在第44章重新测量，未用旧并发点拼接。

[44：完整短负载前沿](../experiments/44-fresh-full-throughput-latency.md)：V126 先验证不可变 C2/C4 启动配置、真实 batch 和输出；V128 再完成同轮 21 次加载。横轴是完整请求 p50，纵轴是总输出吞吐。C1/C2/C4 本方中位点分别为 1226.301 / 1470.632 / 2011.992 ms 和 26.0947 / 43.5117 / 63.3790 token/s，覆盖本轮全部外部配置点。每配置三次独立运行，不能据此声称统计显著或其他负载胜出；旧图和负结果继续保留。
