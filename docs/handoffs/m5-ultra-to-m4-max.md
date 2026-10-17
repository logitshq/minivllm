# M5 Ultra → M4 Max 同步说明

2026-10-16。共享仓库 `logitshq/minivllm`；M5 分支为
`experiment/m5-ultra-256gb`。本分支已合入 M4 分支
`tutorial/18-gdn-projection-packing` 的 `0ed8df8`（2026-10-16最终fetch），
包含第22–37章原生执行、投影/归一化、CAP64、SIMD4及失败记录。此次合并没有改变
M5 已测量的引擎源码；M4 的原型仍是未接入状态。

## 先看这些提交和结果

- `dc745bc`：一次清理同一步结束的全部请求。只改两个核心文件，净增11行。
  `ModelRunner.drop_many` 最多压缩一次剩余 KV/GDN 行；整批结束直接释放。
  这是最适合先在 M4 独立评估的通用改动，不依赖 M5 kernel。
- `044a263`：B1–B8 的有界 decode 图、投影/GDN 融合扩展。
- `f44bc5a`：单行 NVFP4 QMV 及编译器参数别名处理。
- `b0d83c3`：完整 Pareto 报告归档。
- [最终报告](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-pareto-report.txt)、
  [图](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-pareto.png)、
  [结果 JSON](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-pareto-results.json)、
  [设计取舍](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-design-notes.txt)。

改动在 `tutorial_code/miniengine`，不是根目录的 `minivllm` 包。
源码检查用 `git show dc745bc -- tutorial_code/miniengine`。在 M4 自己的分支
先 `git fetch origin`，再选择合并 M5 分支或单独移植需要的改动；M5 继续只推送
自己的实验分支，不更新 M4 的工作分支或 main。

## 已测范围与跨机器边界

M5 Ultra 80 GPU cores / 256 GiB / MLX0.32.3；同一 Qwen3.8-27B NVFP4，
19输入/32输出、greedy、正常EOS、无草稿模型和前缀缓存。69个独立进程，
3,843条正式测量请求，9个外部配置中位数均被实测 native 配置覆盖。
Lithos 官方接口仅测 C1，使用本机 M5 配置；oMLX C1含 balanced/aggressive。
这不是长上下文、其他模型或投机解码的胜出结论。

批量清理的 C64 吞吐中位数为175.83 tok/s，旧版本119.69；三轮范围
155.09–176.38。C32约+2.95%，C8约-4.35%，全部保留。新旧版本未逐对配对。
该改动通过24个真实引擎步骤的输出及全部剩余缓存字节签名比对，覆盖
B2/8/32/64、交错退出、迟到请求和stop token，以及4项CPU生命周期检查。
在 M4 上仍需按本机内存和负载独立验证，不照搬 M5 的 C64 队列或性能结论。

`compile_decode/fuse_decode` 默认OFF。M5的融合路径有明确硬件/版本/
模型/形状检查，范围B1..8、used length≤256；其他情况回到原路径。
派生权重额外约8.714 GiB。不要为了在 M4 命中而删除硬件 guard；M4上
需要单独证明数值、资源和收益。M5上的结果不推翻M4 v20/v21的STOP。

## 可复用与留在 M5 的文件

Git包含引擎源码、基准脚本、汇总JSON、图、校验清单、失败/拒绝候选说明。
`.artifacts/`、`.venv/`、模型和约196MB本地原始证据不在Git；清单中的绝对
路径只是M5证据索引。归档的启动脚本也绑定M5的路径和冻结源码目录，不能
直接在M4执行。跨机复现需在本机构建路径、冻结环境和模型后重跑。

CPU生命周期检查（仓库根目录、已安装MLX的Python环境）：

```sh
python -B bench/m5_ultra_256gb/retire_batch/test_retirement.py
```

本次同步还核对了安装来源：M5的oMLX是官方0.7.1.dev1 wheel，无法证明其
VCS提交；本地源码checkout的SHA不代表被测wheel。已纠正报告中的这一
元数据归属，原始计时和性能数字未变。

## 系统优化攻略与后续实验（2026-10-16）

新增[攻略入口](../optimization-guide/README.md)、[论文笔记](../optimization-guide/papers.md)
和[实验账本](../optimization-guide/experiments.md)。后续每项优化先落盘计划，
再追加正确性、全部计时、资源、失败和取舍；生产热路径保持小。

M5新实验E007（独立大batch GDN融合）、E008（allocator 2/16 GiB配对）、
E009（线程内状态读取复用）均未过预声明收益门，保留原型和证据，未改生产引擎。
E008有12个新HTTP进程、1,170条正式请求，C64配对吞吐几何平均仅+0.131%；
更大池多占约14 GiB空闲buffer，不采用。随后E010已完成M5成组prefill/NAX资格，
见下文；没有直接删除旧backend guard。

M4的`17bd7be`[卷积融合报告](../experiments/21-gdn-conv-fusion.md)已合入：局部micro下降12.21%是继续做模型资格的
依据，不是HTTP收益；与这里大batch的E007/E009不是同一个实验，不能互相替代。

## E010已过采用门：M5保留M=T的NAX分组

29组模型正确性、448层级检查和16步真实调度通过。核心候选只改layers及runner，
用广播packed weight保留单序列的NAX tile；旧split-K投影仍逐行。M4继续使用原legacy
路径，不能移除硬件/版本guard套用M5结论。

初筛完整prefill约1.26–1.42×，但HTTP发现另一活跃`omlx-server`被旧matcher漏识别，
计时受干扰。该阶段的受干扰数据完整保留；随后使用可执行名探针、用户授权暂停/恢复服务，
直接对最终核心完成24进程配对和独立重算。C8/C32/C64吞吐几何平均分别
+14.68%/+12.30%/+14.54%，TTFT −26.29%/−24.42%/−23.02%；通过预声明门，
采用核心`b33b282`。C64最后一对增益仅6.73%，保留所有样本。
C2/C4/C16另外各三新进程补齐当前前沿；外部对照沿用归档并明确标注历史来源。
[实验账本E010](../optimization-guide/experiments.md)和[NAX设计攻略](../optimization-guide/prefill-design.md)包含范围与复现边界。

[最终报告与33进程证据](../experiments/m5-ultra-256gb/prefill-nax-v2/README.md)及
[当前前沿图](../experiments/m5-ultra-256gb/prefill-nax-v2/current-native-frontier.png)已保存。
当前native C1/2/4/8/32组成联合前沿，覆盖9/9历史非spec外部配置中位数；
仅适用于此19→32短负载，不声称覆盖MTP/DFlash2或Lithos DSpark。
外部oMLX服务已恢复，同一PID/start身份通过，`/v1/models`返回HTTP200。

已阅读M4的[最新交接](256gb-mac/PROGRESS-20261016.md)，并运行第37章stdlib verifier：
258源码身份、566文本成员、797依赖绑定通过。V105 thermal STOP及V107 HOLD保持原样；
这些归档不构成M5上CAP64/SIMD4/投机路径的运行资格。M5本轮继续target-only，
没有与M4正在进行的V108/V109混合，也没有将两机数据pool。

## 2026-10-17：E011拒绝、E012独立selector模块

E011大batch图复用仅B16=1.023×，未过1.10×门，原型与失败已归档，未接入核心。
E012原型`e96c1a5`移除draft选路每个位置的两次主机同步，保持FP32 score运算次序。
33组score/IDs及9组完整draft/KV严格通过；完整draft九点几何平均1.177892×。
不是整请求收益，也没有证明当前SpecRunner无损。

正在接入的边界是`spec/device_selector.py`、`spec/dflash2.py`的selector开关，以及
`engine/llm_engine.py`的`draft_device_selection=False`参数。默认关闭。
不改GDN缓存、SpecRunner或literal-prefix-history事务；M4继续V108/V109可独立推进。
三次新进程实际核心复测已预声明。M4取用前请先看
[E012 tutorial](../optimization-guide/tutorials/012-device-draft-selector.md)中的最新资格状态，
并在M4重测收益；不复制M5性能结论。之后通过本交接文件和各自分支同步。

已fetch、阅读并合入M4的`4862215`，此次只有README/交接，没有改变M5被测核心。
已知V109官方MTP数值PASS但计时不排名；V108 tiny通过、旧ID硬门失败仍保持FAIL；
V112在定位同输入逐算子差异。M5不重复修改这条事务/诊断线，等待其完整源码与证据。

**E012实际核心三进程复测已通过。** 核心`f05c3cf`，完整draft九点几何平均分别
1.185085/1.185097/1.190796×，最弱单点仍1.094524×。99组score/IDs、27组完整draft/KV
通过，97文件完整归档见[E012接入报告](../experiments/m5-ultra-256gb/E012-draft-selector-integrated-v1/README.md)。
默认OFF；可用`load_drafter(..., device_selection=True)`独立评估，不必接入旧SpecRunner。
M4的事务与本模块组合时仍须新资格。记录、基准和失败均推本M5分支；不声称已通过聊天送达。

## E013定位与并行研究边界

E013已测32/256/1024历史、block3/5/8：5层draft backbone始终为最大组件，lm_head占
完整draft约22.9%–28.9%。108数组严格通过，首次命名导致的导入失败也保留。用户授权
多agent并行后，本机三条CPU实现线分别为E014 bounded draft KV、E015官方MTP对照资格、
E016 draft backbone图复用；GPU测试仍由本主线程单队列隔离运行。

## E014/E016结果与下一轮

E014 v1全部136数值检查通过，长历史回放加速但2048边界0.9102/0.9556×，拒绝。
另发现跨臂输出释放计时归属值得纠正。v2改为超过2W才裁到W，明确本臂teardown；
预注册2/16 GiB两个fresh process，以16 GiB强reference作采用门。这里只是合成KV回放，
不宣称真实长上下文生成或target verify资格。

E016保持原数学、显式输入输出KV的backbone图：24组312数组严格通过，完整draft几何
平均1.040008×，低于1.10×门；首次图调用205.702 ms。拒绝，不继续堆bucket。
原始数据、全部首调、108正式计时和资源恢复证据已归档。

官方E015 MTP正在准备fresh普通engine对照，使用本机专用0.7.1.dev1环境；
注意官方 `_MTPResetBindingProxy` 的累计计数写在proxy，不能读underlying model冒充计数。
跨机器仍请独立资格。此轮fetch后M4分支仍为4862215，未发现新的代码提交。

E014 v2两pool实测过门：16 GiB强reference 32K=1.19–1.31×，131K=1.83–2.22×，
短点min0.9834；368数值全过。默认OFF核心开关接入与四组合GPU资格分开继续。
16 GiB末采样Swapins+3已披露，Swapouts和swap用量不变，没有为此挑选重跑。

E015 M5官方MTP engine资格已过：fresh两臂192rawIDs相同，每MTP请求16verify/31proposed/
15accepted/32delivered。与M4的17/34不同，不能混同；本机对照原始ID在index4为60277。
资格只证明同机fresh官方对照与MTP一致，观察器写盘时长不排名。完整安装包身份和事件
已归档。Lithos显式DSpark b169bc4的3权重shard已SHA验证，尚待实际Session资格。

E014实际核心05d75cf净增20行：空cache配置`draft_bounded_kv`，禁止live toggle，
与selector四组合共2944数值通过。三个16 GiB进程的短点最小0.979995/0.946272/0.970806；
第二进程失败，所以整体收益门未过，暂留默认OFF实验开关。请勿按原型v2过门自动采用。
2 GiB兼容性short min0.971258。全部4进程保留，未追加挑选重跑。

新检索[ReaDy 2026-10-14](https://arxiv.org/html/2610.20503v1)已读§3/4.1：
需经训练reader读取committed GDN states，当前权重不能直接套开关。本机E018下一步
只做保持原helper算术的短T persistent GDN核与literalprefix history，不替代你们的
projection几何诊断。目标是少读写递归state和少dispatch，先单层精确验证/再谈整模。

## 2026-10-17 新同步与进展图

已实际fetch、阅读并合入M4的`aea008c`，包含第38章完整源码/公开证据；没有改变M5核心。
V112定位同输入Linear的shape差异，以及V113全部497个broadcast-M1与concat-M1相等，
给后续整模verify提供了方向；仍不把它当成完整ID修复或速度证明。M5继续独立E018
literal GDN状态驻留，与M4的V114事务资格分工。V111官方MTP负收益也保留，不移植成M5结论。

M5 E017实际Lithos native M5 Ultra、empty recipe普通/DSpark六请求192raw IDs完全相等；
真实gamma7生成与日志已独立审计：每请求14轮/98verified/17accepted/31decode committed，
另有prefill的1 token；327项归档SHA核验通过。它与E015一样只有插桩engine数值/路由资格，
没有HTTP速度结论。target相同但本机Lithos index4为100606、oMLX为60277，跨框架不混oracle。

用户要求常画图，已保存[第一版进展快照](../optimization-guide/progress/2026-10-17-v1/README.md)：
7张PNG+SVG、840条正式样本和逐源SHA。保留E014全部4进程与短点FAIL；HTTP和模块分图。
后续每个有结论的实验批次创建新版本，不覆盖旧快照。

E018隔离literal GDN核已实际初筛+三fresh复测，T1/3/5/8×真实/零/随机状态，
每步输出/S和全部keep前缀共660个严格比较通过。三fresh的T3/5/8几何平均
1.397486/1.384694/1.410092×；第二进程T1=0.939084，未过原无回退门，所以整体
PASS/FAIL/PASS，当前候选不采用。它不修projection几何、不证明整模事务；
T>1方向可与将来V114组合验证，T1应另测直接保留原路径的分派，不事后改本轮门。

## 2026-10-17 吞吐–延迟主图与模型范围

用户明确主目标为通过正确性门之后的 **throughput vs latency Pareto**。
已从48进程、2304正式请求的HTTP start/DONE重算完整请求延迟，保存
[v3快照](../optimization-guide/progress/2026-10-17-v3/README.md)；不再把首文本后交付速率作为主图横轴。
采用group p50→三组中位数→三个fresh进程中位数，与吞吐聚合层次一致。
native C1/2/4/8/32为当前联合前沿；C1=622.15ms/51.43tok/s，C32=5286.72ms/190.44tok/s。
外部仍为此前target-only对照，MTP/DSpark无observer HTTP正在准备，不能混入旧图。
固定调度数值资格不等同跨框架或异步batch全部raw IDs恒同，v3保留原合同和限制。

模型双重点改为Qwen3.8-27B与GLM-5.3-Flash。后者320B/18B active，现有MLX混合
4/5/6bit权重约189.98GiB，先做C1短上下文加载/质量资格；本轮未下载或测GLM。
当前自研Qwen核心和Lithos注册表没有glm5_next直接适配，本机oMLX/MLX-VLM有源码支持但未资格。
详见[模型路线](../optimization-guide/model-roadmap.md)。全量GLM5.3的MLX4bit约389.59GiB不全驻留256GiB。

E019已按预注册be12f82完成GPU提取micro：150数值点通过，120计时点，长尾GM7.247945×，
但72其他点中27点低于0.97×，整体REJECT，核心未改。服务74384已恢复S。
只是synthetic hidden-history提取，不能写成整模7×；完整输入/样本归档正在独立CPU审计。
本轮fetch后M4分支仍为aea008c。

## 2026-10-17 E020 新 HTTP 结果与 GLM 准备

E019 的完整 310-file 原始归档与独立审计已提交 `09568bc`，全部数值通过但性能门失败，
没有接入核心。E020 也在该提交冻结后完成全部 15 fresh servers、90 请求，45 formal。
当前同轮 C1 的完整请求延迟/总吞吐中位数为：native 716.10 ms / 44.69 tok/s；
oMLX none 679.32 / 47.11、MTP 752.06 / 42.55；Lithos none 812.92 / 39.36、
DSpark 741.96 / 43.13。所有预注册输出/路由合同通过，不能再以旧 E010 native 锚点
声称新一轮三方全面领先。MTP 与 DSpark 的成对吞吐 GM 分别 0.905527 / 1.085991×。
完整图表见 [v4](../optimization-guide/progress/2026-10-17-v4/README.md)。

用户再次明确 Pareto 只画 throughput vs latency，正确性是硬前提。v3 保留此前并发扫描；
v4 是新的 C1 切面，两批不池化。新旧 native HTTP adapter payload/观察器不同，尚未因果
归因。E022 独立 CPU 原型测试 stdlib TCP_NODELAY，只换 accepted-socket 选项；
原始结果和失败不改，未执行前不宣称收益。没有更改 M4 的 projection / transaction 代码。

GLM-5.3-Flash MLX 固定 revision `fa04cbc3622b16aa7475404060b05e506d81bd4e` 的
62-shard、203,992,076,296-byte 下载已启动；尚无运行资格。首个参考必须走真实
oMLX vendor preload/sanitize/EnginePool 路径，它已有 C1 MoE/KDA/mHC 融合。
5-bit routed down 是否落在慢路需要实际资格与成本测量，静态边界见
[GLM first-principles 审查](../../bench/reviews/glm-flash-first-principles.md)。
下载/SHA 期间不跑 GPU 性能测量。本轮再次 fetch，M4 分支仍为 `aea008c`，已合入。

E022 的 16 项 CPU 资格已由实现者和主线程分别通过，固定三对 OFF/ON、36 请求，
plan SHA `b239f6117c4a2b7799b5cf366b31c0a8277182f77773b5cc78c49f218f321fc1`。
仍未运行 GPU；见 [E022 tutorial](../optimization-guide/tutorials/022-native-transport.md)。
下一次完整并发曲线采用新候选工作负载方案，先修正 adapter 的实际 admission：
E020 C1 shim 是单请求排空，不能只提高 client C 就声称 batching。oMLX MTP 在 contention
下会回退，Lithos 单在途限制须真实报告。见 [下一轮 Pareto 方案](../../bench/reviews/pareto-next-workloads.md)。

## 2026-10-17 已接收第39章与长任务入口

实际fetch到M4 `89b9bd2`（含`8bc68da`），通过merge `269fa1e`合入本分支。
已读V113 broadcast-M1、V114整模资源STOP、V116 bounded loader及V117待资格边界；
运行第39章`verify_sources.py`，31份源码/许可SHA与语法检查通过，无MLX/GPU工作。
这是Git可验证的接收记录，不表示已在M5授予runtime资格。M4的事务/verify主线继续由M4推进。

M5下一条完整曲线为E023长编码任务，八份输入已在输出或计时前冻结：886–978 prompt tokens，
cap512、正常EOS、thinking off；fixture SHA
`6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db`。
新单worker HTTP入口每step接收pending、独立rid退休；异步client固定C槽补位、实际token计数、
完整POST→DONE计时，不复用旧C1串行入口冒充batch。CPU mocks不等于长形状数值资格；
真实同调度logits/cache/IDs与外部policy覆盖验证仍待执行。

静态审查发现当前compiled/fused decode的长度guard≤256，长任务从首decode就fallback；
grouped prefill还要求同长度、T≤512，当前八份自然长度也不走该快路。这是后续独立优化
候选，不能先放宽guard或改任务再宣称已有收益。[E023教程](../optimization-guide/tutorials/023-realistic-throughput-latency.md)
记录实际范围。用户要求的主图仍是正确性过门后的throughput–完整请求latency。

GLM固定190 GiB下载继续，无本机速度点。新的[加载预算审查](../../bench/reviews/glm-flash-load-budget.md)
确认官方后置gate/up已逐层eval/drop/clear；更早的113,446个tensor与逐expert stack在整模型首次
eval的峰值尚未知，不能把190+外部服务RSS简单相减当作安全加载证明。新增CPU header inventory
等待下载全部SHA通过后运行。研究笔记补Edge0：其预测替换路由及recovery LoRA不进入本次
同权重正确性主线，可独立学习固定槽位复用；BaseRT当前核心为独立binary，未声称读到其kernel。

## 2026-10-17 接收第40章、失败留档与下一轮曲线

已 fetch M4 `058bdad`，合入 M5 的 merge `326b32b`。根线程独立核对第40章
41份 published files的大小/SHA与Python语法，V117原receipt解压SHA亦一致。
V117的49次真实verify、A/B各96 tokens和提交状态严格资格已读；V120两次及V122
在HTTP投机计时前因资源门停止，不能当作速度点。尚未在M5运行这条路径。

M5 E022 v1只完成首OFF六请求，ON readiness把实际TCP_NODELAY=4误当不符合1而退出70；
矩阵停止，两个PID/端口已回收，外部服务同identity T→S恢复。全部66文件已归档，
不复用partial OFF。v2按布尔语义修正；Apple source及真实CPU socket0/4已验证，
18 tests root通过，独立plan SHA `e850dcdcc1d160ecddf512529758b32fc9a8241c6af515fb72901c4a6b28c81c`。
原三对顺序和性能门不变；此交接时尚无v2运行结果。

E023实际长任务同调度数值资格实现完成，全部52 CPU机制测试root通过；GPU资格仍待跑。
E024默认OFF固定L1536 graph-only候选已独立实现，15 CPU mocks通过，未改生产core；
真实masked SDPA/logits/state与整请求收益都未授予资格。

GLM固定revision的62 shards（189.982GiB）下载和全部SHA校验已完成，CPU header已清点，
没有加载或性能点。外部服务physical footprint约142.82GiB，不能用RSS或SIGSTOP
冒充释放内存；临时卸载/恢复路线已审读，还未执行卸载。

新Pareto审计确认旧图的实际throughput/完整latency重算一致，但历史外部C1资源matcher
没充分排除已启动的74384；新图将历史C2–64与受控E020 C1分面，既不拼接两批前沿，
也不把正确性当可交换轴。详细审计见 [v3/v4 review](../../bench/reviews/pareto-v3-v4-audit.md)。

后续E022 v2已完成：36请求全部正确、6进程全部reap、独立audit0 errors、外部服务恢复S。
完整请求paired GM=1.013777×，TTFT首对0.899891×，未过原采用门，未接入core；
223文件完整归档。v5分面图已独立逐点复算和视检，217份来源SHA一致；
保持“正确性是硬门，吞吐/完整延迟才是Pareto两轴”的口径。


## 2026-10-17 接收第41/42章，长负载资格运行中

已实际fetch并merge M4 `ee9f243`（包含`eba6320`），merge为`97273be`。
root阅读两章并运行标准库保存证据校验：V121的0.997538× transport比、V123的
0.4880478× spec/ordinary比、全部请求/资源/退出证据均通过。保留无收益结论，不在M5
冒充已运行这条路径，也不混合两机器的绝对速度。

M5已推送`ed32cab`，新增E023 owner与冻结数值计划（此前commit `e9600a8`）；本轮正在
真实GPU运行37个固定请求、C1/C8/混合生命周期，plan SHA
`21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb`。
逐步PASS不能当整组PASS，带观察器计时不用于Pareto。E024单独的1536图诊断已实现，
32个形状点+真实参数替换+B1→B8→B1+actual ahead检查，12CPU tests经过root和独立复核。
GPU诊断未调度；后续完整D512/gatesON资格仍必需。

cache关键事实：`_sync_batch`的新行容量参数是prompt+output budget，D任务首次decode
已预留1536，不需先改KV增长策略。现有256图的closure读取长度同样固定256，不能只放宽guard。
GLM190GiB权重已全部SHA通过，但原服务142.82GiB物理驻留不因SIGSTOP释放；恢复预检发现
admin主key未配置且匿名admin卸载401。尚未修改认证、卸载或加载GLM；Qwen工作独立继续。

## 2026-10-17 E023资源中止，Pareto口径与下一步

E023 v1已终止并完整归档：C1完成8请求/4096 IDs；C8最后step515，未完成16请求与13请求lifecycle。首个失败是pressure2（sample164，swap324→469.06MB），无已保存数值不一致。child TERM/reap、owner absence及原74384同identity T→S恢复均有证据。独立stream audit与root重跑完全一致，qualified=false；已完成请求均cap结束，无实际EOS早退覆盖，末尾model/source recheck未执行。不得把观察器计时或正确前缀加入性能图。

[完整归档](../experiments/m5-ultra-256gb/E023-realistic-qualification-v1/README.md)保存332成员，压缩7.8MB的xz保留原1.4GB JSONL。E024仍冻结未运行，当前资源压力仍2，先查buffer ownership。E025已准备独立、默认OFF的T19 same-split NAX部分乘法诊断，root10项CPU tests通过，未编译Metal/加载GPU，不接入core。

最新可报告的仍是[v5吞吐/完整延迟图](../optimization-guide/progress/2026-10-17-v5/qualified-throughput-latency.png)：历史C2–64与受控E020 C1分开；正确性是入场门，不作可交换轴。C1 oMLX仍领先，不能宣称全面前沿获胜。


## 2026-10-17 E026默认OFF状态所有权候选

E023失败完整归档与E025独立NAX诊断已push `6968805`。新的E026源码审查证实MLX0.32.3的
GDN `xp[T:]`保留整块concat buffer；只把最后3行用uint16 contiguous复制，可避开修改算术。
复制与立即materialize分别开关，root7项CPU检查通过，尚未GPU资格/接core/产生性能点。
[教程](../optimization-guide/tutorials/026-conv-tail-ownership.md)与容量推导均保留；约3.46GiB是
step515两arm的名义多余backing引用量，不是实测节省或E023根因。独立无权重buffer诊断正在准备。

Apple10月15日MLX文章已读，Qwen3.8数据为BF16/512GB/长上下文，不能混入当前Pareto；
TreeWY生产实现未承诺逐位轨迹，且本仓早已单checkpoint+tape，不直接移植其算术或套store_all倍数。
服务pressure2仍在；仅SIGSTOP不释放143GiB，coldrestart具体恢复方案已准备，用户明确授权待回复。


## 2026-10-17 接收第43章与V124源码请求

已实际fetch `057355a` 并merge为 `e3d7b25`。root审读第43章与verify.py，再运行标准库离机
复算PASS：9 fresh arms、54HTTP/1728tokens+3startup/96、155区间外资源样本、清理证据一致。
M4 native C1配置中位1230.159875ms/26.012879 tok/s，相对已测oMLX1.033882452×、
Lithos1.196079257×，三配对方向一致；不扩称显著/全并发前沿或三家raw IDs相同，不混入M5图。

看到了V124共享W多行M1的40case/80矩阵逐位PASS交接。请后续发布可复用kernel源码、
冻结plan/geometry/编译参数、完整资格证据及尚未覆盖的497真实projection列表，让M5能按同一
累加顺序独立复现；当前这次commit只含说明，M5尚未运行或声称已接收该kernel实现。

M5这边E026压紧3行conv尾helper已push `a56f9f1`，tiny设备诊断正在独立冻结；它不会改变
qkv/GDN数学。若M4也出现prefill状态驻留，可复用该helper和source推导，但设备分派guard需
独立注册M4，不要删除M5 guard后直接声称支持。大型实验coldrestart授权仍等待用户回复。

E026 tiny实机已PASS并独立复核：3fresh arms48case，T978完整backing20,103,168→65,536B，所有原位保留；parent/source释放及最终active0有阶段证据。注意finite验证临时buffer使OFF也有98,440B active下降，所以归因看retained backing而非总active delta。[完整归档](../experiments/m5-ultra-256gb/E026-conv-tail-tiny-v1/README.md)。无整模型/性能资格，原74384同identity恢复。

E026后续：scoped ordinary-prefill hook已通过12项root CPU checks及独立review；仅改tail ownership，copy/materialize独立且默认OFF。尚未GPU集成，不可直接拿CPU结果入性能图。约62MiB单GDN真实权重路径可独立诊断，注意不得mx.load整5GiB shard再筛。

E026 layer v1准备完成：14个真实layer0张量共65,239,488B，原5.34GB shard完整SHA已核，再独立对照全部抽取bytes；608源/runtime/bundle pins冻结。OFF/ON-lazy/ON-materialize各8case，root27 CPU检查+独立审计31负测PASS，设备结果尚无。严格逐位输出/状态通过后才推进整模型；单层时间不入Pareto。当前目标始终是正确性作为硬门槛的throughput–latency前沿，oMLX/Lithos均保留强合法配置。

E026 layer v1首case前被physical parser误杀：top current65389240、auxcurrent/peak65323704，均<2GiB，旧相等断言失败；非数值错误/内存超限。0case、ON未启动，owner/child已reap，原74384恢复。完整失败已归档；v2只修读数假设与取消文案，预算和原GDN算术不变。
