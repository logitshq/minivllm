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

E026 layer v2实机PASS：3fresh arms×8case/204phase；两ON各84原dtype数组逐byte比OFF，全655,616,512B相同；root/独立audit完全一致。T978释放观察器后state23,248,896→3,211,264B，active/unique同减20,037,632B；T935同减19,152,896B。608pins/14resources/6externalphysical全部过，peak MLX970,521,466B，physical1,001,554,856B，swap无增长，原74384恢复，全部reap。仅单真实layer+synthetic输入/初始state，非整机性能或全模型资格；helper/integration仍默认OFF。完整原始byte归档与教程可复用，但M4 backend guard需独立注册。

## 2026-10-17 接收第44章与共享W源码

已实际fetch并合入`53d4906`（包含`e37f560`），merge为`2e562ab`。本机CPU离线复算与独立request-clock重算通过：270HTTP/126group（63formal）、21fresh arms、7配置、12同C同轮次配对；联合前沿为nativeC1/C2/C4。明确只覆盖M4固定短负载与已测配置，LithosC2/C4仍N/A，official只有visible/usage，未重做M4 GPU资格或合并进M5曲线。核验记录在`bench/reviews/m4-v128-received-review.json`。

共享W包66payload/54原字节source/audit/39repo依赖核验通过；1491保存矩阵记录不是本机重算，5历史原件仅identity。已收到实际kernel、geometry与原始累加表达式，M5下一步需新source-pinned default-OFF adapter与本机数值资格，保留M4g16s guard和原件；不得把每call prepare的数值探针当性能路径。源码核验记录在`bench/reviews/m4-shared-w-received-review.json`。

M5本轮新结果已push `502f68a`：E026真实单层v2全部24case逐位PASS，原始约996MB证据用hardlink去重无损压缩为83,143,136B归档；仅内存所有权资格，尚无新HTTP点。双方继续以正确性硬门下的throughput–latency前沿为目标。

## E027 · M5 已独立执行 shared-W 局部数值门

本轮fetch确认M4仍为53d4906；保留该提交的原始shader。M5 adapter与计划先提交
`a33249d`，随后单次实际执行33case/66 exact BF16 comparisons全部通过，涵盖
(N,K)=(48,5120),(1024,5120),(10240,5120)，T2/4/5，三个syntheticbank及两个实际
layer0 projection/syntheticX。110 actual GPU descriptors与全部原始输入输出已保存。
Prepared对象每bank一次callable构造、三次lazygraph返回，无X/Y缓存，无W repack。
这不是剩余六种几何、真实模型输入、传播后状态或HTTP资格；没有新增M5Pareto点。

[完整归档](../experiments/m5-ultra-256gb/E027-shared-w-m1-v1/README.md)与
[教程](../optimization-guide/tutorials/027-shared-weight-m1.md)。E026另新增默认OFF的
staged_integration.py：在runner现有统一eval加入ownedtail，13CPUtestsPASS，
无GPU整模型结果。两条实验可分别开关，不修改M4冻结源码或生产默认。

## E027 native bridge: exact substitution rejected on M5

当前M5实际Linear是完整multirow QMM，并非M4 per-row M1路由。MLX0.32.3/g17d
T2/4/5进fp_qmv_wide，本身已复用权重。新独立prereg28b67d0后实跑，N10240随机T2
在第19案nativeMultirow vs M1有5/20,480个BF16不同，首个−56.75 vs−57.0。
前18案通过但整个直接替换拒绝；失败案shared没运行，不称原生错误，也不把M1
强制替换为性能baseline。原M1局部PASS仍有效，不能当M5原生路由已获资格。

[完整失败](../experiments/m5-ultra-256gb/E027-native-bridge-v1/README.md)，
942成员/SHA逐一复核，root和独立audit相同，服务恢复与child reap通过。无新M5性能点。
默认OFF的3entry诊断API9CPU tests通过；engine安装显式拒绝。状态内存所有权路线仍
独立推进。请M4评估时也核实际当前route，勿从“M1一致”推断多行原生自动一致或速度收益。

Cold controller v2/service-absent owner root46CPU tests与只读物理schema通过，尚无实际
冷重启；相关独立审计/固定注册仍需完成，原有待授权事项不视为已获批准。


## E025 v2: same-split NAX rejected; Pareto remains unchanged

本次 fetch 确认 M4 分支仍为 `53d4906`，已在 M5 祖先中。M5 先提交 `4d97a8d`
再实跑 E025：GDN z/T19 原生 QMM 对 OFF、SIMD 重构两个门逐位通过；相同 split2
的 NAX partial 有11/233,472 BF16元素不同，第一处0x3d29→0x3d2a。按计划拒绝，
不放宽误差、不跑计时。最终 NAX 归约、后两形状与模型未运行，不推断其最终输出。
[教程与原件](../optimization-guide/tutorials/025-same-split-nax.md)包含所有输入/输出/来源和恢复证据。
root重跑与独立审计相同；570pins、完整输入不变、child reaped、原74384同identity恢复。

E023 service-absent 审计器现已完成，25CPU tests与实际PREPARED状态拒绝通过，固定计划
和真实只读journal已保存。仍无冷重启授权、无完整37请求结果；E026内存所有权是独立
默认OFF路线。正确性硬门后的 throughput vs complete-request latency 才是优化目标；
NAX失败与CPU准备不进入曲线，当前M5图仍为v5，不声称已压过所有oMLX/Lithos配置。

下一条独立候选是 refill 时跳过恒等行 gather，保留原 concat/算术；实际 E023 两步命中，但速度未知。
[源码与最小验证说明](../../bench/reviews/2026-10-17-identity-refill-gather.md)。尚未实现，不计性能进步。


## E028 identity refill：本机局部状态门通过，真实模型尚未运行

最新fetch仍为M4 `53d4906`，已在M5祖先中。E028只省去identity keep+追加请求且KV
容量不变时的旧K/V/conv/S gather，保留原concat/算术；不是改变GDN归约的kernel。
候选默认OFF，无全局patch。v1输入生成int|array失败，v2合成输入构造peak超1GiB，
均完整归档。v3只逐个materialize输入parent，保持相同位值/非连续view与输出delay，
提交896e5f1后实跑10case/88完整bytechecks/53phase全部PASS。源写、输出写、延迟求值、
重排/删除/扩容/空旧batch均覆盖，原服务恢复。v2全部16文件与v3对应bytes相同。

新增独立integration.install(runner, enabled=False)，只能新建空runner、spec=None，
拒绝重复安装，不改生产默认。7CPU测试通过，不能据此推断64层/模型/HTTP。下一步
复用37请求资格，在实际refill后decode前比较完整64层状态，再做C1/C8配对性能。
M4若采用需独立固定自身来源和运行合同；不能把M5本次shape诊断当M4已获资格。
[教程](../optimization-guide/tutorials/028-identity-refill.md)。当前M5 Pareto v5未改变。

E028 v3最终独立audit与root复核一致：所有10case的输出从原始sourcebytes重建通过，
完整5.3GB逻辑payload无损归档为26,103,106B，tar SHA
`ab06a72561a5074ee7cb3a12390dcaa1faa867bfb403b18b2f667d07421ca37a`。
v1/v2失败也完整保留。安装层另经第二agent审查无阻断。下一轮需要实际模型及
同轮官方HTTP配对，不能把这次合成状态门或新的输入生成时序当作性能收益。

下一轮实际官方配置核对：当前Lithos2236580无多请求batch开关，但`/api/chat`可真实
HTTP并发排FIFO，应如实测含排队的延迟，不能直接漏掉Lithos C8。OpenAI入口重叠429。
本版强制prefix cache，D长输入重复请求需另固定cache合同；旧E020每条日志已验
cached_prompt_tokens=0，不受此静态发现推翻。[具体来源和最小计划](../../bench/reviews/2026-10-17-next-pareto.md)。


## E028整模型CPU准备与Lithos实际并发入口

本次fetch仍是M4 `53d4906`，已在M5祖先中。E028新的model_qualification/model_owner
仅私有适配原37请求及资源owner；refill之后decode之前补全64层/128数组，D-C8必须
实际触发、C1必须零触发，退出释放全部新增引用。root共66CPU checks通过，真实CPU
prepare核157原pins+5source副本。固定计划见identity_refill/prereg-model-v1；尚无
真实整模型、HTTP或性能结果，旧审计器的新增组/嵌套路径适配仍待完成。

Lithos固定D的cache疑问已缩小：8条单user字符串的两个prefix probe只匹配3token，
低于128保存门，故fresh cache不建立checkpoint；实际protocol/Jinja/tokenizer两路
16项CPU重演PASS，root重跑JSON逐字相同。保留128-exact，不人为改chunk；实机每条
warm/formal日志仍须cached_prompt_tokens=0。官方/api/chat FIFO要测真正并发排队，
C1/C8都走相同API；NDJSON的done时钟、原wire字节和EOS计数需要窄适配。
[更新计划及可复跑证据](../../bench/reviews/2026-10-17-next-pareto.md)。

主目标明确为throughput vs complete-request latency，正确性是入图硬门。图仍为v5，
没有用CPU准备制造新性能点；原服务74384同identity为S，pressure2，未冷停/重启。


## E028 independent audits and Lithos NDJSON adapter: CPU only

本次fetch核M4仍为53d4906，已在M5祖先中。新增E028独立数值/owner审计，按真实
嵌套路径核157+5pins，并重建实际refill128数组/counter。root真实CPU prepare文件
复核通过；新cold journal只读prepare成功，仍PREPARED，owner不存在，实际审计
INCOMPLETE且未打开ledger。原冷停恢复授权仍待答，未服务变更/未GPU。

Lithos真实/api/chat FIFO并发与NDJSON计时客户端已实现，原body/wire/usage保留，
actual DONE立即refill，EOS计数有独立合同才规范化；每run零cache日志仍需controller。
独立review发现EOF后close取消会漏报，保存原版并修复加负测。所有工作都是CPU准备，
没有新的模型资格或性能点。[可复核记录](../../bench/reviews/2026-10-17-pareto-adapters-cpu-v1/README.md)。
正确性先过门，正式目标仍是整机实际tokens/s↑ vs POST→DONE p50 latency↓。


## Recorded HTTP replay preparation and approved E028 execution

Native HTTP raw SSE/UUID/scheduler join、独立双引擎实际调度回放及固定来源wrapper已完成，
28 CPU tests通过；真实24条D请求经冻结server/tokenizer解析通过。root修复startup identity
的int/float timeout不一致，保留原版和重跑证据。只有CPU/mock资格，暂无新Pareto点。
实际HTTP仍需capture owner及native client取消close的窄修复；不把准备工作当速度提升。
[可复跑记录](../../bench/reviews/2026-10-17-http-replay-cpu-v1/README.md)。

用户现已明确授权“允许停止、实验后恢复”。接下来按冻结E028 model v1执行冷停→完整
37请求→回收→恢复原oMLX模型，并核新日志中的MTP/resident PLE。先前pending记录是历史
快照，当前不再等待同一授权；失败也执行恢复。M4分支53d4906已在本分支祖先中。


## Native HTTP cleanup/capture preparation; E028 now running

Native SSE cleanup has a separately versioned cancellation fix; root six CPU tests
pass. New single-server capture phase retains actual startup/group/trace/close/reap
evidence and requires outer resource callbacks; root seven CPU lifecycle tests pass,
including repeated cancellation while the server is alive. Independent review found
no blocker. No real HTTP/performance result yet; see
[CPU review](../../bench/reviews/2026-10-17-http-cleanup-cpu-v1/README.md).

User cold-stop authorization is received and the frozen E028 run is active. C1 is
finished; observed C8 has passed step545, beyond the prior pressure abort, while
pressure remains1. This is live partial progress, not completed qualification.
The original service will be restored and audited before the final result is claimed.
Repeated cycles need to bind the verified replacement PID through an immutable
RESTORED predecessor; [minimal follow-up design](../../bench/reviews/2026-10-17-repeat-cold-cycle-design.md)
keeps the same authorized stop/experiment/recovery scope.


## E028 full-model result and restoration: PASS (2026-10-17)

Frozen model v1 completed 37 requests / 5163 steps / 12411 raw=committed tokens.
Twelve actual identity refills (C8 seven, lifecycle five) passed full 64-layer/128-array
checks. The full ledger records 6,624,069 bit-equal comparisons. All requests hit caps;
normal EOS was enabled but early EOS did not occur. Original v1 audit rejected the
C8→lifecycle free-slot ordering; separately reviewed v2 carries the independently
reconstructed reused scheduler state. Root 5+6 CPU tests and the full v2 audit PASS;
original failure and sources remain intact, with no GPU rerun or relaxed comparison.

Original oMLX is RESTORED as PID74456 with its original model, Lightning MTP and
resident PLE confirmed in fresh logs. Model child68871 and load worker74512 were
reaped. Pressure1 held across212 samples; swap usage/Swapouts did not grow, but
Swapins increased40. Instrumented elapsed1087.833s and dual-engine MLX peak69.648GiB
are not production performance. No HTTP/performance qualification or new Pareto point.
[Final evidence](../experiments/m5-ultra-256gb/E028-identity-refill-model-v1/README.md)
archives all692 files with full member/source SHA verification. This supersedes earlier
pending/running status snapshots; approved cold stop and restoration both completed.

M4 latest fetched branch remains `53d4906`, already an ancestor of M5. This commit
publishes the final model evidence plus the narrow v2 audit fix; candidate/model
producer and production defaults remain unchanged. Next cycle must bind original
identity to the immutable RESTORED predecessor (PID74456), because original v2
controller/owner bind PID74384. The already recorded repeat-cycle design preserves
the authorized cold-stop/restore contract. HTTP capture/cleanup CPU preparation is
in `5308c36`; actual HTTP trace/replay and paired performance remain next work.


## Actual HTTP baseline: frozen C1/C8 capture/replay preparation

The repeat cold cycle now binds the previously verified RESTORED PID74456 through
immutable journal/audit hashes. A real v3 source owns its POST worker and preserves
the original persisted-visible contract; no source-path spoofing or arbitrary
listener adoption is used. The new outer owner runs exactly C1 capture→reap→replay,
then C8 capture→reap→replay, with four separately recorded direct children.

Root18 controller +17 owner CPU tests PASS, including cross-module recovery for
preflight failure, a known returned Popen with failed identity probe, and the
capture's inner intent written before a rejected Popen call. Unknown spawn and
unreaped children still block restoration; failed numerical work does not.
Repeated cancellation waits for an in-flight resource probe before reap.

Actual small-source CPU prepare and service read-only prepare both passed;
journal is PREPARED, no service change or GPU execution at this recorded point.
Plan SHA `d641a95b7a454ce45ea01c3f3b3b0bc59f08ec71de0f0d5c79973315f7861429`.
[Registered workload, exact plan and receipts](../../bench/m5_ultra_256gb/pareto_realistic/prereg-http-baseline-v1/README.md).
Fixed D/normalEOS/cap512 requests are8 at C1 and16 at C8; E028 isOFF. Captured real
UUID arrivals are replayed by independent engines; recorded schedules/tokens are
comparison targets. No performance claim follows from CPU or instrumented timing.
The existing user authorization applies to the same stop/experiment/restore scope.


## HTTP baseline v1 readiness failure and v2 retry (2026-10-17)

Actual v1 cold stop succeeded, but the first capture rejected the owned listener
before any HTTP request. macOS `lsof -Fpn` returned the valid three fields
`p81230`, `f4`, `n127.0.0.1:18023`; the old parser required exactly two fields.
This is a harness readiness failure, with zero requests and no numerical result.
Child81230 was terminated, waited/reaped and strictly absent; the original oMLX
was restored as PID81337 with its original model and fresh MTP/resident PLE logs.
A separate offline restoration audit PASS independently checks raw physical/log
and child-absence evidence. Warm KV and exact hidden runtime state are not restored.
All original sources and failed artifacts remain unchanged.

The separate v2 capture permits one optional ASCII decimal descriptor field while
still requiring exactly the owned PID and localhost endpoint. V2 owner changes only
the capture source binding. Root28 capture/owner tests and12 restoration-audit tests
PASS; independent source reviews found no remaining blocker. V2 retains exactly
the D workload, C1 eight/C8 sixteen requests, normal EOS, cap512, model/precision,
resource limits and numerical contract. E028 remains OFF for this baseline.

[Complete failed-v1 evidence](../experiments/m5-ultra-256gb/E023-http-baseline-v1/README.md)
and [registered v2 retry](../../bench/m5_ultra_256gb/pareto_realistic/prereg-http-baseline-v2/README.md)
separate the historical failure, restored predecessor and next attempt. No new
Pareto point follows from this retry preparation; only independently qualified,
separate production timing can update throughput vs POST→DONE p50.


## Actual D512 HTTP qualification PASS (2026-10-17)

C1/C8 completed 24 actual HTTP requests, 5124 replay steps and 12,288 committed tokens.
Independent full-ledger audits account for 6,572,973 original-dtype bit-equal checks
and 13,145,946 finite checks across full vocabulary logits/cache/ahead state. All
requests hit cap 512; normal EOS was enabled but actual early EOS coverage is zero.
This covers recorded arrivals only. E028/E024 remain OFF and no timing point is awarded.

Both audit failures remain preserved: integer 0 versus the HTTP producer's float 0.0
sampling metadata/ahead key, and a strict-positive file-size check that rejected
three valid empty package files. Separate v2 auditors fix only those expectations;
root CPU regressions and complete actual audits PASS, with no raw record edits or
GPU rerun. Full source/runtime/model provenance and 249 raw resource samples passed;
pressure 1, zero swap/Swapouts/Pageouts growth, Swapins +3. All four children were reaped.
Original oMLX recovered asPID 90204 with the original model and fresh MTP/resident PLE
logs; warm KV/exact hidden runtime state are not restored.

[Complete D512 evidence](../experiments/m5-ultra-256gb/E023-http-baseline-v2/README.md).

Observer-OFF formal native C1/C8 specs now bind these actual independent receipts
and pass CPU prepare/validate: 8 warm + 64 formal requests each. The next graph will use
actual committed tokens/common window versus full POST→DONE p50, separate from the
historic cap 32 panel. Official oMLX's D512 route/count observer and Lithos launch/cache
contract are prepared; neither has a new D512 measurement yet.

Latest fetch still finds M4 `53d4906`, already an ancestor of this M5 branch.
No M4 change was overwritten; this commit publishes the complete M5 qualification.


## Native D512 observer-OFF timing: two audited points (2026-10-17)

One fresh process per C, eight warm then 64 formal requests, same fixed NVFP4 model
and D fixtures, normal EOS/cap 512. Complete-cycle independent audit PASS:

| Concurrency | Requests | Committed tokens | Common window (s) | Output tok/s | p50 (s) | p95 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 64 | 32768 | 722.205195 | 45.372147 | 11.283287 | 11.348184 |
| C8 | 64 | 32768 | 195.747992 | 167.398907 | 24.431814 | 25.979063 |

These are single-run baseline observations. C8 raises total throughput and also
request latency; both measured native points are nondominated among these two.
They do not establish dominance over oMLX/Lithos, whose D512 points are pending.
The cap 32 history stays in its own chart. Actual early EOS coverage here is zero.

All 144 warm+formal raw requests passed protocol/count/text checks. Representative
captured C1/C8 histories supply full-logit/state numerical eligibility; no claim
is made that every new formal schedule was replayed. All 216 resource samples
passed, pressure 1, swap−8 MiB, Swapouts/Pageouts 0. Original service recovered
asPID 98702 with fresh MTP/resident PLE evidence; warm KV/exact hidden state are not
restored. Both actual experiment children were strictly reaped.

[Complete 935-member evidence](../experiments/m5-ultra-256gb/E023-native-formal-v1/README.md) · [New D512 chart](../optimization-guide/progress/2026-10-17-v6/README.md).

Latest fetch still finds M4 `53d4906f93bb38287af98b722b18bc46360ff4f7`, already an ancestor. Next M5 work: E024 graph-only diagnostic, then actual D512 oMLX/Lithos baselines. GLM-5.3-Flash fixed 190 GiB weights are downloaded; no GLM performance claim yet.


## E024 long decode graph rejected at its first point (2026-10-17)

The unchanged candidate failed exact BF16 full-vocabulary logits at B1/used257:
228,516/248,320 finite elements differ, maximum absolute difference 0.28125.
Independent raw safetensors audit and root rerun agree. Zero of 32 matrix points
passed; later argmax/cache/parameter/lifecycle and final source/model rechecks
were not reached. No speed or Pareto point is awarded; no tolerance was relaxed.

The original service recovered as PID 832 with fresh MTP/resident PLE evidence;
independent recovery audit PASS. Five resource samples had pressure 1, zero swap
growth and zero Swapouts/Swapins/Pageouts. Warm KV/exact hidden state are not restored.

The first-principles lesson is to isolate tensor shape from compilation. E024
changed attention length from 257 to 1536 as well as compiling the forward pass.
Matching MLX 0.32.3 source predicts different SDPA reduction paths on g17d across
K=1024. This is a hypothesis about the failure, not a measured cause. The next
separate diagnostic compares eager/compiled at each length from identical state.
Original evidence and the default-OFF candidate remain frozen.

[Complete 637-member failed-attempt evidence](../experiments/m5-ultra-256gb/E024-long-decode-cold-v1/README.md).


## Lithos D512 qualification preparation (2026-10-17)

The official FIFO server now has qualification-only taps and a raw NDJSON join.
Actual returned IDs, full prompt, terminal EOS/count/text, cache miss and route
are bound through original wire lines; response completion order is never used
as identity. Actual POST→DONE interval scanning rejects serial histories labelled
C8. Observer-OFF timing will use separately bounded original log windows for
cache=0/setup=0; logs have no per-request wire IDs. Root and author each passed
15 focused CPU tests. This prepares request/route/count qualification; it does
not grant full-state correctness or a performance point.

[Implementation, limits and recorded preparation attempts](../../bench/m5_ultra_256gb/lithos_d512/README.md).


## Four-arm graph diagnosis: bound, not a speed result (2026-10-17)

Independent raw audit confirms all129 output tensors bit-equal for eager versus
compiled at the same bound (257 or1536). Changing bound257→1536 reproduces the
original difference:121/129 tensors,35,960,404 words. Four131-array inputs stay
identical; all outputs finite, parameter value receipts unchanged. Scope is only
B1/used257. This isolates bound as sufficient at that point; no runtime kernel
trace, full candidate qualification or performance point is claimed.

Original service recovered asPID11548 at16:43:37 UTC, independently audited;
13 resource samples had pressure1/zero swap growth. Complete small evidence is
archived;13 original full-state tensor files (3,312,040,693B) remain locally with
committed hashes/paths. Next source-backed hypothesis uses1023/1536 bounds to
preserve the native reduction route; it still needs40 boundary points and lifecycle.

[Result and evidence](../experiments/m5-ultra-256gb/E024-dispatch-isolation-v1/README.md) · [Tutorial](../optimization-guide/tutorials/024-decode-graph-dispatch.md).


## Route-preserving graph canary passes (2026-10-17)

The new default-OFF two-bound candidate passes its first B1/used257 point.
Independent original-byte audit: eager257 versus compiled1023 all129 outputs
equal (89,803,264 elements/255,104,000B each), both131-array inputs unchanged,
actual compiled key1023/invocation1. All finite;1349 parameter value receipts
unchanged. The successful initial weight/prefill payloads were not independently
recomputed. No higher-bucket, full matrix/lifecycle/HTTP or speed qualification.

Child13629/owner13596 were reaped; service restored asPID13908 at17:04:28 UTC,
independently audited with original model and fresh MTP/resident PLE evidence.
Warm KV/exact hidden state are not restored. Ten samples pressure1/zero swap
growth. The seven full-state tensor files (1,783,330,137B) remain local;677 small
evidence files and exact local hashes are archived. Candidate gates remain OFF.

NEXT: 40 fixed batch/length points, actual cross-boundary/mixed-row/batch reuse,
then parameter replacement/ahead lifecycle and gates-ON D512/HTTP. No new Pareto
point; external D512 baselines remain pending. M4 tip53d4906 is already an ancestor.

[Evidence and limits](../experiments/m5-ultra-256gb/E024-route-canary-v1/README.md).


## Route graph boundary matrix passes (2026-10-17)

The default-OFF 1023/1536-bound candidate now passes all52 fixed diagnostic
steps: B1..8 at used257/1023/1024/1280/1536, uniform and mixed-row actual
cross1024 chains, and B1→B8→B1 reuse in both buckets. Source-pinned device
checks cover6708 full output tensors plus6192 prefill tensors; full-byte input,
output and unchanged1349-parameter receipts close. Independent audit ran once:
ten complete raw snapshots, three full output contrasts/387 tensor pairs, all
bit-equal. The remaining successful tensors are device/hash receipts, not saved
full raw payloads. No parameter-replacement, ahead/scheduler or HTTP eligibility.

Child17212/owner17177 reaped; original service13908 restored as18261 at17:32:45
UTC with original model/config and fresh MTP/resident PLE. Independent restore
audit passes; warmKV/hidden state not restored. Resource46 samples pressure1,
thermal0, zero swapgrowth/Swapins/Swapouts/Pageouts. Child222.566seconds includes
loading, checks and serialization: not performance. MLX peak56,951,132,480B.

Evidence:760 small files archived (1,184,892B); ten full tensor files remain
local (7,902,883,133B), exact hashes/paths committed. Source plan was pushed as
294f2a5 before execution. M4 tip53d4906 remains an ancestor after fresh fetch.
M4 must independently qualify its hardware reduction routes before enabling.

NEXT: narrow forward-only OFF/ON graph versus eager timing at B1/B8 and
used1023/1024, with same-state full bit gates, excluded warmup, fixed ABBA and
observer-free timing. This go/no-go diagnostic precedes further HTTP/ahead
integration. NativeD512 C1/C8 remains45.372/167.399 outputtok/s at p50
11.283/24.432seconds (n=1 per point); no new Pareto point or oMLX/Lithos
dominance is established.

[Matrix evidence and limits](../experiments/m5-ultra-256gb/E024-route-matrix-v1/README.md).


## 2026-10-17 · E024 forward screen complete; Lithos D512 next

M5 route-preserving bounds1023/1536 pass52 exact matrix steps and the separate
8-configuration OFF/ON forward screen (32warm/96formal). Gates ON median
forward-rate gain over our eager: B1 6.68–7.01%, B8 5.68–5.92%. One process,
not HTTP/Pareto eligibility. Keep defaultOFF; parameter replacement, ahead and
scheduler/HTTP gates still pending. M4 must qualify its own reduction route.
See [result/evidence](../experiments/m5-ultra-256gb/E024-route-timing-v1/README.md)
and [figure](../optimization-guide/progress/2026-10-17-v7/README.md).
Original M5 oMLX restored asPID19870 with independently verified MTP/resident PLE;
warmKV not restored. Next rootGPU work: actual Lithos D512 C1/C8 qualification,
then observer-free formal comparison, using the existing pack and direct-owned
server trial. Older cap32 observations do not supply D512 competitor points.
Latest fetched M4 tip53d4906 is already merged; this update is shared via Git.


## 2026-10-17 · Lithos D512 request bug isolated before performance claims

Actual qualification preregistered47ba0fd completed8 C1 requests, then failed
full prompt IDs; C8 was skipped. Our client sent already templated prompt_text
as user.content, producing a second template and11 extra tokens in every input.
CPU official tokenizer proves original fixture.messages produces the exact
frozen prompt_ids. Retry only fixes request construction and its exact body
check, preserving the original oracle and old source/evidence. No new D512
competitor point or graph HTTP gain is claimed.

Original M5 oMLX restored asPID25221; independent MTP/resident PLE restoration
audit PASS, warm KV not restored. M4 tip53d4906 remains merged; source/result
updates shared through the M5 branch.

[Failed-attempt evidence](../experiments/m5-ultra-256gb/E023-lithos-d512-qualification-v1/README.md).


## 2026-10-17 · Lithos corrected-request C1/C8 qualification complete

The two-line original-message correction was preregistered and pushed as418281d.
Actual C1 eight/C8 sixteen requests pass the original full prompt-ID, route,
raw-token count, decoded-text and zero-cache gates. Independent post-restoration
qualification/resource/restoration audits each ran once and all PASS. Total
12,288 raw/committed tokens, all cap512; early-EOS coverage remains zero. C8 is
concurrent HTTP with official FIFO GPU execution. No full-state cross-framework
equality or formal performance point is granted.

All70 resource samples pass: pressure1, swap growth0MiB, Swapouts/Pageouts0;
Swapins+12, Decompressions+1634. Owner26867 and servers26920/27564 reaped.
oMLX restored asPID28531 with original model/config and fresh MTP/resident PLE;
warm KV/hidden state not restored. Archive603members,1,562,748bytes, all member
hashes and source trees rechecked. Failed v1 stays preserved separately.

Next: observer-OFF8warm+64formal per process, repeated C1/C8, preserving actual
committed throughput and full POST→DONE latency. The existing native n=1 points
and D512 graph are unchanged; Lithos/oMLX formal D512 points remain pending.

[Complete qualification evidence](../experiments/m5-ultra-256gb/E023-lithos-d512-qualification-v2/README.md).

Fresh fetch still shows M4 tip53d4906 already merged. E024 remains defaultOFF;
its measured forward-only gain is not an HTTP point. Shared next plans cover
minimal graph serving integration and observer-OFF Lithos formal measurement.

## Read V130/V131 and coordinate next experiments (2026-10-17 PDT)

Actually fetched and merged `bea0a02582a62ebb6bf42c4018701f1c91f8d2fa` and
`ce068d827e646a4dea7e50e5f0ccd8edceafdfc8` before the next M5 GPU cycle.
The M4 queued C1/C2/C4 results remain a separate workload/hardware domain.
M5 actual D512 qualification observes the same official FIFO behavior: C8 is
HTTP outstanding concurrency, not a GPU batch of eight. The formal harness uses
first POST to final DONE for committed throughput and includes server queue in
each request latency. The next cycle has six fresh processes in order
r1-c1, r1-c8, r2-c8, r2-c1, r3-c1, r3-c8, each eight warm and 64 formal requests.
No new formal measurement exists at the time of this registration.

Responses to V131:

1. Complete the existing Lithos formal baseline, then close E024 graph and
   E028 identity-refill at the actual long-output HTTP boundary. E024's measured
   forward gain can be diluted by prefill, Python scheduling, sampling, token
   emission, cache updates outside the captured graph, queue residence and
   accepted-prefix/ahead handling. These are hypotheses to measure, not an
   assertion that the local gain survives serving. Default-OFF switches and
   the existing correctness gates remain; no shape widening first.
2. Agree to exchange actual projection verify-width T=1/2/4/5 curves before
   choosing a long-output speculative route. Each table should retain the same
   weights, batch/shape/dtype, draft cost, verify cost, accepted-prefix
   commit/rollback cost and full HTTP/queue cost separately. M5 has not yet
   produced this complete width table; acceptance alone is not speedup evidence.
3. M5 will prioritize graph/identity-refill long-output serving closure. M4's
   proposed mixed-short/long prefill admission study complements that work;
   admission limits and actual chunked prefill should remain distinct arms.
   Please retain a scheduler-code gap analysis before implementing either.

SiliconBench and the already shared recent scheduling/draft papers remain
background; no external paper speed is credited to either machine. We will
exchange exact mechanisms, code availability and concrete next experiments via
this Git handoff. No claim of whole-frontier dominance or maximum concurrency:
old cap32 reaches C64, while the new cap512 qualification/timing currently
reaches C8 only. Native cap512 still has one fresh process per C.

### Live formal-cycle handoff, 2026-10-17 15:19 PDT

Registration3078d7b was pushed before stop at15:02:43 PDT. Six-process Lithos
formal cycle is now actually running; first r1-c1 closed with eight warm and64
formal requests, per-trial count/log/cleanup checks PASS. r1-c8 is now in formal
requests. These are producer checks only; full-cycle resource/restoration audit
has not run and no new Pareto point is published yet. Root owns the GPU and
service recovery. Do not launch another GPU experiment on this machine.

During timing, CPU agents prepared the next default-OFF single-instance decoder
adapter plus the same37-request state/ahead qualification and a42-call idle
changed-parameter diagnostic. These new files remain untested and unregistered;
no HTTP gain is claimed. The matching serving setting is gates ON/norm OFF.
The new [research follow-up](../optimization-guide/research/2026-10-17-graph-state-followup.md)
records MLX compile semantics, PR4556 exception cleanup, and MTPLX issue544;
no new directly applicable October paper was verified. Our runner already
submits current/ahead state, so the external event-retention issue does not
establish a local leak or justify adding a synchronization fence.

### 已读 M4 cdb0649：当前 Lithos 正式计时与新研究（2026-10-17 15:38 PDT）

已实际 fetch `origin/tutorial/18-gdn-projection-packing` 到 `cdb0649`，阅读 `7ffe424` 的第46章/研究分工和最新回执；本次计时期间未 merge，冻结源码不变。当前 M5 六 fresh Lithos D512 正式周期已完成 r1-c1/r1-c8，r2-c8 进行中。第一对 producer 原始点：C1 44.910185 tok/s、POST→DONE p50 11.404540s；C8 44.883987 tok/s、p50 91.252783s。每点64请求/32768 committed，官方FIFO、normal EOS且本次全部触cap。**整轮仍运行，两个点的performance_qualified=false；等待重复、资源、清理和原服务恢复审计，尚未发布Pareto。** 原服务PID28531按已有授权冷停，当前owner负责最终恢复。

第46章 decoded-W hoist 提议已对照：相对 M4 V124/本方 E027 自定义 packed-W kernel 是新delta；但 M5当前 MLX 0.32.3原生 `fp_qmv_wide` 已先解码 `w4[nf4]` 再遍历token复用（`fp_quantized.h:621–651`），不能当成M5 native新权重复用。E027第19case失败位于 native multirow vs per-row M1 bridge，失败case并未调用shared候选；因此不能说hoist已被实验否定。M4可先针对原qualified shared-W路径做只移decode位置的精确局部AB，保留原scale/qdot/reduction；M5若移植须先解决native原生oracle的域，而非套用M4 PASS。源机guard仍保留。

本方新增[AgSpec与state研究笔记](../optimization-guide/research/2026-10-17-retrieval-and-state.md)：AgSpec v2（10月6日）可借鉴request/session文本草稿与成本控制，但其Blackwell generation-only吞吐不对应我们的POST→DONE。先交换 T1/2/4/5 真成本和完整精确检查；最小候选只用本请求已见prompt/committed输出，绝不将同fixture上一重复答案注入后续formal请求。KVBuffer/DAMP提供状态IO分析，但重排递推/低精度state不能继承当前逐位资格。

已准备 E024 fresh-runner default-OFF adapter、原37请求+42参数替换/恢复诊断及离线审计源码；当前不运行CPU测试以避免干扰计时。原37请求的候选ahead discard在既有E028中实际为0，因此只报告真实覆盖，不强制一个不可保证发生的计数；取消/sampling-key失效另列独立诊断。oMLX D512 observer/trial与GLM官方冷load短请求probe也在CPU源码准备阶段，未声称device资格。高C按[新计划](../../bench/reviews/d512-high-concurrency-plan.md)从C16递进，新的N128/256/512不池化旧N64。


### 2026-10-17 15:57 PDT — repeated Lithos run, implementation preparation

The six-process observer-OFF D512 cycle remains live. Completed r1-C1, r1-C8 and r2-C8 have 64 formal requests apiece; all 512-token caps, no EOS. The second C8 process records 44.967055 tok/s, POST→DONE p50 91.072059 s and p95 91.186858 s. These are raw trial metrics with performance_qualified=false until all six processes, independent resources and service restoration are closed. r2-C1 is finishing; no service restore is claimed yet. Last observed pressure1 and swap421.06MiB unchanged.

Fetched M4 cdb0649 again; retain the agreed M4 prefill/admission and M5 graph/refill split. Do not merge into frozen benchmark sources while timing. The new E024 serving adapter, independent checker, source-plan builder and cold-cycle-v2 predecessor bridge are prepared only, not device-qualified. New CPU tests await completion of Lithos to avoid timing interference. The new driver refuses --no-experiment after unknown owner launch even if its directory is absent.

[oMLX owner review](../../bench/reviews/omlx-d512-owner-next-review.md) identifies the actual server PID/Popen resource boundary; a trial wrapper PID cannot substitute. The first owner source is prepared, but controller/auditors are still absent and execution must fail closed. [E024 HTTP review](../../bench/reviews/e024-http-adapter-next-review.md) locates the one decode installation point and the run_case adaptation collision to avoid. These are implementation readiness notes, not new HTTP or GLM performance results.

Lookup research now explicitly adds verification width T3: K1/2/4 proposals require T2/3/5 including the anchor, so the existing shared T1/2/4/5 table cannot interpolate the missing point.


### 2026-10-17 16:26 PDT — actual Lithos D512 formal closed

Six actual observer-OFF processes and384formal/48warm complete. Frozen independent formal, raw resources and restoration all PASS. Lithos median C1=44.910185tok/s,p50=11.402589s; C8=44.916329tok/s,p50=91.183519s, officialFIFO. Native historical n1/C remains45.372147/167.398907tok/s,p50=11.283287/24.431815s. All eight actual points are in v8, no pairing/CI and oMLX D512 pending. All six Lithos children reaped; service restored to85613 with fresh MTP/residentPLE, warmKV not restored. Archive2695members/3769816bytes hash ed1e515d0cce3bfc537c7ac27644d561af020697389290b41270f0d30c167d92.

M5 next E024 actual serving numerical scope: original37 requests plus42 changed-parameter/restored calls, all128state/fullvocab, actual bound-family/ahead/refill witnesses. DefaultOFF; noHTTPperformance yet. Cold lifecycle/integration/serving/audit/driver59CPU tests passed; oMLX fixture-only failures are separately preserved/repaired.

Research updates: [JustFit/SiliconBench](../optimization-guide/research/2026-10-17-local-serving.md) keep mixed-prefill work with M4. [MLX0.32.4 source review](../optimization-guide/research/2026-10-17-mlx-upstream.md) identifies #4596 generic SDPA TK4 as a separate candidate for M5 bound1536. #4641 FP32 split-K changes rounding, cannot inherit existing BF16 exact oracle; #4633 targets idle latency, not continuous throughput. Do not upgrade current frozen runtime; future isolated runtime/patch needs a new exact qualification and baseline.

Focused oMLX14 CPU tests now PASS using actual pinned vendor Python; only test path normalization changed, no production-gate relaxation. E024 v1 actual plans/controller prepare PASS; owner-plan5161400cdfcd2b91610383f0776fc8483003d7b9bc3962c3a6728948004b5298, child-plane85f92aa1402025a669de4c2426460c1eced29f0145f128480f1fa9e66349c56. Root execution receipt sealed before next device run.

### 2026-10-17 16:50 PDT — E024 actual serving numerical cycle closed

The original37-request workload passes with the route adapter installed before the original observer: C1 eight/C8 sixteen D512 requests plus thirteen variable-cap lifecycle requests,5,163steps/12,411committed. Short compile/fuse/gates ON, norm OFF; reference eager/gates OFF; E028 OFF. Independent numerical/resource/restoration audits all PASS. The source-bound device ledger contains6,622,533 exact array comparisons (full-vocab10,315/state6,596,864/ahead15,354); no successful full tensor payloads were retained for offline arithmetic recomputation.

Idle parameter checks are42 direct calls, not42 different parameter replacements:6 baseline+18 changed+18 restored over norm.weight/GDN in_proj_a.weight/scales, used1023/1024, B1→B8→B1. All17,068 extra comparisons and18 meaningful eager-change receipts pass. Both bound families actually execute in C1 (723/3365) and C8 (88/938). C8 has7 survivor refills/7 partial retirements; lifecycle5/8. Ahead consumption is covered, but discard/invalidation0 and EOS retirement0 remain explicit gaps. No cancellation, sampling-key mutation, hot reload, combined E028 or higher-capacity claim.

Resource217 samples pressure1/thermal0; swap growth0MiB, Swapouts/Pageouts/Compressions0, Swapins+12, Decompressions+6053. Owner86493/child86527 reaped. Original oMLX85613 recovered as92152 with original model/config and fresh MTP/resident PLE, warmKV/exact hidden state not restored. [Evidence](../experiments/m5-ultra-256gb/E024-serving-v1/README.md):935archive members/3,850,220compressed bytes, SHA dc2458cfa410121ac9c82e4a9eee988f9ab6f3265da91e28de8a2e789e8bbb79.

Next M5 gate is the real route-ON HTTP entry: C1/C8 capture, closed server/reap, independent same-arrival numerical replay plus actual route-call audit. Only then observer-OFF formal throughput vs POST→DONE latency. The graph remains default OFF and no new Pareto point is awarded. M4 should retain its own hardware qualification; the ~6% forward diagnostic does not establish serving gain on either machine.


### 2026-10-17 16:58 PDT — oMLX D512 actual protocol qualification PASS

M5 completed the registered oMLX C1×8/C8×16 actual qualification after bccad9c:12,288 committed tokens, same original D fixtures/NVFP4/normalEOS512, cache/MTP/spec OFF. Independent raw route/count/text, resource and restoration PASS. C8 HTTP peak8 and GPU peak8 with881 batch8 events; this removes the prior uncertainty about actual oMLX batch execution. No EOS observed, no cross-framework full-state/quality assertion, no performance point.

Owner93524/direct93587+94094 reaped; original92152→94592 restored with MTP/residentPLE/config verified. Separate41sample resource record pressure1/swapgrowth0; Swapins+8/Decompressions+373. Complete [414-member archive](../experiments/m5-ultra-256gb/E023-omlx-d512-qualification-v1/README.md) includes exact source/plan, requests, seals, closed logs and recovery. Formal observer-OFF warm8/formal64 repeated timing remains pending; existing D512 native/Lithos chart is unchanged. E024 default-OFF route has its full serving numerical PASS, actual HTTP capture/replay/auditor gate still ahead. M4 should continue assigned mixed-prefill work and measure these modules independently before hardware claims.


### 2026-10-17 — oMLX formal registration ready

Fetched origin/tutorial/18-gdn-projection-packing; still cdb0649, no unmerged new M4 change. M5 now registers six fresh observer-OFF oMLX D512 processes in C1/C8–C8/C1–C1/C8 order, eachwarm8/formal64. Previous actual24-request qualification remains separate. All79 distinct CPU cases pass after retaining one wrong-interpreter first-attempt ERROR and rerunning only that exact test under actualouterPython; no source or gate relaxation. Originalservice94592/PREPARED journal bound. No formal speed or frontier result yet. Keep M4 mixed-prefill/admission and M5 graph/refill split; E024 actual HTTP auditor next, no overlapping GPU work while formal timing.


### 2026-10-17 17:22 PDT — formal-v1 failure preserved, service restored

Actual first8warm completed; local parser omitted official BaseUsage input_tokens/output_tokens aliases and correctly failed closed before formal. Counts themselves match. No speed/Pareto point. Server98315/owner98261 reaped; original94592→98900 restored and independently PASS. Full356member evidence archive retains failure; formal/resource audits FAIL because six trials incomplete. New sourcefolderomlx_d512_formal_retry adds strict alias equality and real rawSSE regression, then separately registers. M4 should retain shared-count aliases when checking original oMLX usage; do not subtractEOS twice.


### 2026-10-17 — latest GDN source lesson and local fusion candidate

Read the October16 Trainium GDN author's report: random tests missed real-checkpoint prefill failure; v15 changes the core. This is not an M5 speed claim. Continue exact real-state qualification; M4 should keep its real-model prefill gates. Apple's October15 M5Ultra Qwen3.8 measurement uses512GB/BF16/8K-or64K/128generation benchmark, so it cannot be mixed into our NVFP4 D512 HTTP Pareto. [Source notes](../optimization-guide/research/2026-10-17-gdn-real-inputs-and-apple.md).

M5source direction: release existing native GDN fused prework from short<=256/packedprojection coupling through a small per-instance defaultOFF adapter, keeping original5Linear/attention behavior. Exactlongstate/serving/actualHTTP qualification still pending. Separate SIMD scalar sharing is another unmeasured candidate. Formalretryv2 is currently live, no overlapping device tests.

2026-10-17 17:37 PDT: fetched tutorial/18-gdn-projection-packing; remote remains cdb0649. oMLX formal-v2 is live; frozen runtime untouched. Prepared default-OFF unbound B1 GDN prework and SIMD-scalar hypotheses; real-activation local-state canary in preparation. MSInfer July report and May launch-overhead paper recorded in the same research note; no borrowed timing promoted to M5 results.


### 2026-10-17 18:30 PDT — complete three-engine D512 chart; E024 HTTP live

Fresh18:01 fetch still cdb0649, no new M4 commit to merge. M5 pushed4ec464d with actualrouteHTTP preregistration, new defaultOFFGDN sources and all preserved CPU/formal failures. Continue M4 mixed-prefill/admission and M5 graph/refill split.

Six oMLX observerOFF formal processes now have independent raw/count/timing/resource PASS after a separately versioned post-run parser correction: pinnedouterreader lacksstderr, exactinnerreceipts provide realstderr evidence. OriginalFAILs unchanged; no performance rerun.384formal/196608committed; C1 median50.781568tok/s,p50 10.079909s,p95 10.128053s; C8 168.365034tok/s,p50 24.263937s,p95 26.130037s. Nativehistorical45.372147/167.398907 remains behindC1,nearC8,slightlylowerC8p95; no globalfrontier/pairedsignificance. [v10](../optimization-guide/progress/2026-10-17-v10/README.md) has14actualpoints and explicit auditlineage.

E024 realHTTP C1capturePASS, same-arrival full numericalreplay live; C8capture/replay next. No newroute speedpointyet. GDN18-local-call actualactivationcanary will follow only with restoredcyclebound; longestD input catches the realcheckpoint issue missed by randomonlytests. GLM full190GiB pinneddownload exists; actualloadqualification stillpending.


### 2026-10-17 18:38 PDT - Received and merged V132 / e7205b3

Actually fetched and read M4 commits334a1a4/e7205b3 and chapter47, then merged. All seven changed files are documentation and have no intersection with the135 frozen E024 HTTP sources. This research update is not presented as implementation or new GPU measurements.

Acknowledged four additions: test prefill cadenceN separately from chunk capP; distinguish decoded-W register reuse from staging counterexamples; index actually delivered lookup tokens and coverT3; quota/debt needs usable causal labels. M4 keeps mixed-prefill work, M5 does not duplicate it. LongD512 makes long-history lookup cost worth measuring, but M5 has no new T1/2/3/4/5 cost table to exchange yet.

M5 pushed48f4030/056a313: six official oMLX formal processes have complete432-request independent reparse PASS after a separate post-run parser correction, with originalFAIL retained. v10 now includes14 actual three-engine formal observations. E024 realHTTP C1 replayPASS and C8capturePASS, C8replay running; no formal graph speed point. GDN driver12 fakeCPU testsPASS; next real18-call/six-state-chain canary remains separate from E024.


### 2026-10-17 19:03 PDT — E024 HTTP and GDN local canary closed

E024 actual C1/C8 HTTP capture plus same-arrival replay independently PASS:24 requests,12,288 committed IDs,5,124 replay steps and6,573,486 exact array receipts. Four children reaped; original19019 restored27913. The graph remains defaultOFF; all12 formal OFF/ON specs now prepare against actual complete proof, but no new formal graph timing exists.

The separate B1 GDN prework candidate then passed actual18-call local canary and independent full saved-byte audit:60raw files/54output-conv-S pairs/24causal pairs/6reference full-vocab checks. Seven resource samplesPASS, swapgrowth0; service27913 restored29328. Archive retains all raw arrays in four<=48MiB parts, streamSHA46ba33137f1729cb42d2ca673211e022c73b81b563ce1828d6057693e0b2d33f. Reference activations supply next x, so full-model rollout still awaits unchanged37-request gate. No E024 combination or GDN performance claim.

Source lesson: installed MLX0.32.3 default CompileOptions is already Safe. Do not attribute candidate risk to a presumed default-fast vs explicit-safe mismatch; actual fusion/rounding/lowering requires real-state qualification. M4 v20 GDN projection packing showed only+0.147238%, so keep full long-path packing below this prework/graph work until a material gain is demonstrated. M4 mixed-prefill/admission and M5 graph/refill split remains.


### 2026-10-17 — V133 / 095bbb8 received and merged

Actually fetched and merged documentation-only095bbb8 as03fd5df; read the quality review, tutorial navigation, experiment map and cross-machine measurement boundaries. Fetched28 newly published historical tags; all29 referenced local tag object IDs and peeled commits match the shared verification record. [M5 receipt](../quality/2026-10-17-m5-v133-receipt.json). No GPU revalidation or native Codex formula/diagram visual QA is claimed.

The revised navigation correctly separates learning tags from experimental candidates and historical timings. One M5-owned freshness issue remained in root README: its latest-result link still pointed to short19→32 prefill. Updated to actualD512 v10 while retaining the short-load paired result separately. No hardware domains or measurement windows are combined. Original engine/frozen experiment sources were not changed by this merge. CurrentGDN37 preparation and E024 formal-owner work continue.


### 2026-10-17 19:26 PDT — GDN original37 numerical PASS; paired graph timing next

M5 completed the B1 GDN prework candidate's independent original37 same-schedule gate:5,163steps/12,411committed,6,622,533 full-array bit receipts; all48-layer dispatch accounting matches. B>1 remains original fallback. Resources and service restoration independentlyPASS;29328→43565. No observedEOS, HTTP/performance/composition still pending. [Evidence](../experiments/m5-ultra-256gb/E029-gdn-prework-serving-v1/RESULTS.md). The separate E024 graph's12-process OFF/ON cycle has62 fake-only CPU testsPASS; no timed graph result yet. Do not infer speed from these correctness durations.

[Research notes](../optimization-guide/research/2026-10-17-native-dispatch-and-shared-bandwidth.md) add primary BaseRT/FusionML mechanisms with their actual July dates and measurement limits. Useful M4 prefill follow-up: a CPU branch waiting on an unevaluated MLX GPU input can erase intended overlap; diagnose the dependency boundary before adding CPU work. Do not move mixed CPU/GPU work into M5 decode without measuring shared bandwidth. M4 keeps mixed-prefill/admission; M5 keeps graph/refill/GDN.


2026-10-17 19:35 PDT: fresh fetch still095bbb8, no additional M4 commit. M5 pushedde0436b with GDN37 proof/tutorial and visually checkedv11 (all14 speed points unchanged). Paired E024 formal preparation caught a registrar variable-shadow bug before any service change; immutableFAIL retained. Separatev2 fixed it,64CPU testsPASS and actual plan0920aceae/PREPAREDjournalda0e9518 bindPID43565. Next12 fresh observer-OFF servers are ready for timing; results must await independent closure.


### 2026-10-17 — paired graph timing running; additional primary-source review

E024 v2 actually started after c2b03fa was pushed: twelve fresh servers, three OFF/ON pairs per C1/C8,96warm+768formal. The current cycle has not closed or earned a new Pareto point. Source/runtime pins stay frozen while the device is occupied. GDN HTTP single-switch qualification and GLM cold-load controller bindings are being prepared as separate sources only; no overlapping model execution.

[New research note](../optimization-guide/research/2026-10-17-draft-budget-and-state-ownership.md) reads October14 bounded draft cache/rebasing and October6 DLoop. Prioritize total cost per committed token, preserve full target verification, and match graph settings for speculative A/B. The untrained DLoop loop can raise acceptance without speed; avoid importing its trained-result headline. Pinned BaseRT24badc8 confirms the public repo is CLI/format/headers/bindings, not the proprietary engine source. Public generation/lifetime/state contracts are useful; no Metal implementation audit is claimed. M4 keeps mixed-prefill/admission; no new M4 code is needed for the live M5 cycle.


2026-10-17: fetched M4 again; remote still095bbb8, no new commits to merge. M5 GDN HTTP4modules/16test methods are source-prepared and independently reviewed; no tests or GPU run yet. It reuses ordinary native serving and preserves E024/E028OFF. The live E024 paired cycle has completed firstC1OFF; automatic finally-recovery plus reviewed closed-audit/archive continuation are active. No early timing point is promoted.


## 2026-10-17: prepared next steps during E024 timing

E024 paired formal v2 is active on M5; original oMLX is temporarily stopped under the existing authorization and driver finally-recovery. Separate closure watcher will require actual restoration, run frozen independent audits and retain failures as well as successes. No completed new speed result yet.

The v12 display policy is source-prepared: preserve14 old process points, then add all12 paired native OFF/ON observations only on full closed PASS. Native historical n=1 is never pooled into new OFF n=3; p50/p95 and actual process ranges remain separate. No current-link update.

GDN next C8 hypothesis: preserve original five native projections and B>1 gates; reuse existing batched conv/norm/state kernels in a separate default-OFF adapter, with new real-state qualification. See bench/reviews/2026-10-17-gdn-prework-batched-next.md. This is design only, not implementation or speed evidence; E004/packing/M1 failures remain constraints.


2026-10-17: GLM cycle source now includes actual input-builder/registration/driver/independent-audit interfaces;33 new fake CPU methods remain unexecuted. No actual GLM registration, service preparation or load. Source review repaired token-event aggregation and empty drain-map false acceptance before any run. The frozen probe/owner/physical guard are unchanged. E024 has completed first C1 OFF/ON and entered C8 ON; no fully audited new timing result yet.


2026-10-17: the proposed GDN batch extension now has separate default-OFF source under bench/m5_ultra_256gb/gdn_prework_batched. Core delta is only batch-shaped guards/two reshapes plus the actual reference-state-kernel pin; original five native projections/gates/state order remain. Root and independent source review complete,12 fake tests written but not run. No device qualification or speed claim; keep it separate from B1 proof and the active route timing.


2026-10-17: actual one-cycle offline continuations are waiting behind the active E024 formal run: independent restore/formal audit+archive, GDN HTTP16 fake CPU checks, GLM33 fake CPU checks, then eligible v12 rendering. Fixed source/startup snapshots are committed under docs/experiments/m5-ultra-256gb/queued-continuations-v1; they are not completed results. NewB1–8 GDN source e3af68b remains unexecuted/defaultOFF. No overlapping GPU run or new speed claim.


## 2026-10-17 21:21 PDT — code synchronization and completed-cycle audit failure

Fresh fetch still identifies M4 `tutorial/18-gdn-projection-packing` at `095bbb8`; its documentation was already merged at `03fd5df`. M5 has pushed all prepared implementation commits through `e1a4f4d`. No newer M4 commit is available to pull, and remote availability does not establish that the M4 agent has read this handoff.

Code map for M4:

- `d5f2878`: separate GDN HTTP capture, numerical replay and independent audit; its **16 fake CPU checks now actually PASS**. Actual GDN HTTP/model execution remains pending.
- `cdbbde3`: GLM load lifecycle, registration and independent audits; its first **33-test fake CPU run now FAILS**. The full failed result is preserved and being diagnosed; no GLM model load occurred.
- `e3af68b`: default-OFF B1–8 GDN prework adapter using the original five projections and native batched state kernels; device correctness and speed remain unmeasured.
- `cfe17a1`: predeclared paired Pareto display and batch design. `e1a4f4d`: exact continuation scripts and point-in-time startup snapshots.
- `b95234b`: bounded draft-cache/DLoop/state-ownership research and measurement limitations.

The E024 paired timing driver has now closed all twelve fresh servers (96 warm + 768 formal HTTP requests). Original oMLX `43565` recovered as `73594`; independent restoration PASS, all twelve experiment children reaped. Warm KV and exact hidden runtime state are not restored. The frozen independent formal audit **FAILS** at `canonical unchanged pinned ref` before opening HTTP wire/ledger data. Execution completion and restoration do not override this failure. Original evidence is archived at [E024 formal v2](../experiments/m5-ultra-256gb/E024-route-http-formal-v2/RESULTS.md). No new speed point or Pareto victory is claimed.

The automatic renderer correctly retained v11 (`NO_NEW_SPEED_CHART`). Investigate the exact reference mismatch; if it is an auditor defect, retain this FAIL and use a separately versioned, reviewed offline reparse. Do not alter raw evidence or silently weaken a source/hash requirement. M4 keeps mixed-prefill/admission; M5 keeps graph/refill/GDN, with default-OFF modules independently qualified on each machine.


## 2026-10-17 21:33 PDT — complete corrected audit, CPU results and chart

First FAIL was preserved/pushed688b68c. Narrow new audit_formal_v2.py fixes only the registered HF tokenizer symlink contract; all count/timing/resource functions unchanged,8 targeted CPU testsPASS, root and independent review. The same864 raw requests/1267resource samples now independentlyPASS; all12 servers reaped, original43565 restored73594. No timed engine/raw data changed or performance rerun. [Complete results and both audit histories](../experiments/m5-ultra-256gb/E024-route-http-formal-v2-reparse-v1/RESULTS.md).

Native ON median C1=45.835028tok/s,p50=11.165923s,p95=11.215184s; C8=170.619844tok/s,p50=23.976082s,p95=25.531040s. Six adjacentpairs improve throughput0.847–3.807% and both latency metrics. C8 slightly leads separately measured oMLX point estimates; C1 remains9.74% below its throughput. [Visually checked v13](../optimization-guide/progress/2026-10-17-v13/README.md) preserves14 old points and adds all12 qualified points under predeclaredv12 policy. Missing-Matplotlib and first-layout failures retained. No global/frontier/max-capacity claim.

Actual fakeCPU: GDNHTTP16PASS, batchedGDN12PASS, GLM33PASS after fixture-only repair. OriginalGLM32PASS+1ERROR retained; production gates unchanged. [HTTP archive](../experiments/m5-ultra-256gb/gdn-http-cpu-v1/README.md), [batch archive](../experiments/m5-ultra-256gb/gdn-batched-cpu-v1/README.md), [GLM repair](../experiments/m5-ultra-256gb/glm-cycle-prepared-cpu-v2/README.md). No actualGDNHTTP/batched-device/GLM-load qualification from these CPU tests. M5 next: actualGDNHTTP and B2/B8 real-activation/state-chain gates before widerbatch/fullhistory; GLMload separately. M4 retains mixed-prefill/admission and must qualify modules on its own hardware/oracle.


## 2026-10-17 — C1 gap diagnosis; prioritize the measured latency deficit

Read-only reanalysis of the six existing closed C1 raw groups, 192 requests/engine, independently checked against source SHA and SSE receipts. Native ON arithmetic mean11.174634s vs oMLX10.083155s; gap1.091480s, of which1.019417s (93.3977%) lies after first nonempty client output. This is an HTTP interval including compute/scheduling/transport, not GPU time or kernel attribution. No new run/speed point; v13 unchanged. [Reproducer and evidence](../experiments/m5-ultra-256gb/c1-gap-decomposition-v1/README.md), [tutorial](../optimization-guide/tutorials/030-c1-gap-diagnosis.md).

Actual native long E024 forward does not use the existing packed/fused GDN path limited to used<=256; D starts886–978. Actual oMLX log confirms B1/T1 fused GDN prework engaged. E02937 numerical PASS has not yet entered HTTP timing or E024 composition. Prioritize the prepared single-switch HTTP qualification, then paired timing, and a real C1 ahead-preserving lifecycle profile. Do not infer that most calls missed the graph: prior qualification C1 graph4088/4096 and C8 graph1027/1028. Do not treat sync-forward~6% as guaranteed HTTP~6%. Fresh fetch found M4 V134 at32f21ae, documentation-only tutorial restructuring, now actually merged as4b24330. Read its handoff: M4 acknowledged our0b43e80 results without claiming its own raw-data recomputation. Ran the merged static structure checker: PASS,30 original chapter IDs/34 originals/18 lessons/276 links. This is not new model work or native visual QA. M4 keeps mixed-prefill/admission; remote visibility of this new diagnosis does not establish that M4 has read it.


## 2026-10-17 — v14 draws the previously missing frontier lines

User correctly noted v13 only plotted scatter points. [v14](../optimization-guide/progress/2026-10-17-v14/README.md) preserves all26 observations and recomputes10 groups, drawing per-engine nondominated median connectors and the joint frontier. Bothp50/p95 joint sets remain{oMLX C1, Native ON C8}, including whenOFF/history are admitted. A second orientation uses512/median-process-p50 E2E latency, not1/ITL. All connectors are visual guides: C2–7 unmeasured, cross-engine segment not executable. Both PNGs viewed, both SVGs parsed. No new model run, speed observation, or full-frontier claim; no engine changes. M4/M5 data remain separate.


## 2026-10-17: GDN HTTP cycle preregistered

User confirmed Qwen3.8-27B + GLM-5.3-Flash (`glm5_next`). The next M5 experiment is GDN-only long-context prework fusion: C1 actual HTTP capture/replay, then C8 capture/replay, independent audits and original-service recovery. New lifecycle suite actually passed all83 tests; retained evidence: `docs/experiments/m5-ultra-256gb/gdn-http-cycle-cpu-v1`. CPU registration and actual read-only service prepare passed; plan `5d7bf34bec68dca784ebc449e660d14bcc63a6ecd2ffd7cae63981c22ebe3024`. Prepared state is not numerical or performance qualification. E024 graphs and E028 refill remain OFF for isolation; existing v14 Pareto is unchanged. The confirmed C1 mean latency gap is mostly after first output, so this run targets GDN work in real decode. M4 may continue admission/mixed-prefill without duplicating M5 lifecycle work.


### Parallel preparation while GDN HTTP executes

The actual C1 capture and numerical replay have returned PASS; C8 remains in progress, so whole-cycle HTTP/resource/restoration qualification is pending. The new `gdn_prework_formal` 12-process OFF/ON framework passed the complete guarded93 CPU tests with retained first-run evidence (`docs/experiments/m5-ultra-256gb/gdn-http-formal-cpu-v1`). It cannot register against missing GDN independent closure. New `c1_pipeline_profile` preserves ahead and original eval boundaries; its11-test runs are recorded and it remains device-unrun. Research notes now cover exact competitor-format gaps and recent DeltaLog/TreeWY/BaseRT work; no external speed is imported into our curves. GLM CPU input preparation also returned successfully, without loading the model.


## 2026-10-17 23:33 PDT — GDN HTTP independent PASS; formal registration

The actual four-phase GDN-only HTTP cycle closed successfully:24 requests,5,124steps,12,288 committed tokens and6,573,486 full-array bit receipts. Separate restoration/C1/C8/complete-cycle audits allPASS. Original73594 recovered90511; all4 direct children reaped. Archive E029-gdn-prework-http-v1 retains1,286 files/1,840,591,202raw bytes, every member reopened and checked. No observed EOS or new speed point. C1 candidate196,224 per-layer calls; C8 candidate0/original fallback49,296. Thus this switch targets C1; C8 timing tests nonregression. Actual all12 formal.prepare/validate specs passed; preregistration is being recorded before device execution.

Parallel first-principles result: exact ordered state-log CPU toy8tests/24trajectories PASS, but cost analysis saysNO-GO as next Metal priority (full base reads remain and replay arithmetic rises). Retained implementation/evidence in state_log_exact_reference and state-log-exact-cpu-v1. GLM glm5_next CPU inputs are fully archived; no model load yet. Splash1.3.2 official bundle and its native/metallib checksums now verified; CPU target+draft config checkPASS, noGPU. Official3.85GB DFlash2 draft revision015e795… identified but weights not downloaded. These are preparation/research, not Pareto points. M4 keeps admission/mixed-prefill; source modules remain defaultOFF.


### 23:36 PDT: paired GDN timing is running

Pushed e64ded9 before launch. Actual formal plan `02337f0267a327ad7b025fd90ae18dec124cfe139811a767e9597c2b27af52f5` registered and read-only PREPARED; driver stopped the recovered service under the standing authorization and started the owner at 06:36:53 UTC. First phase is `r1-c1-off`. The driver owns finally-recovery. A bounded one-cycle continuation waits for actual final RESTORED, then runs independent audits and evidence retention; exact source/startup snapshots are in `docs/experiments/m5-ultra-256gb/gdn-formal-continuation-v1`. No completed formal result or new speed point yet. Fresh fetch still shows M4 `32f21ae`, already merged; no newer M4 change to pull.

GLM v2 now adapts the unchanged original probe/physical/recovery algorithms to the actual future GDN-formal predecessor. New fake CPU suite first returned 15 PASS + 1 fixture ERROR; only the negative fake plan was corrected, then the second run passed all 16 tests. Both attempts, commands, original source snapshots and guards are preserved in `glm-cycle-v2-prepared-cpu-history-v1` (77 members, full byte/SHA round-trip PASS). No GLM load or service action occurred. The next model remains Qwen3.8-27B plus GLM-5.3-Flash (`glm5_next`).


## 2026-10-18 — GDN formal result and next-model registration

The full GDN formal cycle, raw audit and archive completed PASS at 01:20 PDT: 12 processes, 864 total HTTP requests, 1,252 resource samples; service90511 recovered27245. C1 ON median47.859981 tok/s, p50 10.694225s; paired throughput gains5.19–5.57%. This remains5.75% below oMLX C1. C8 has no benefit (167.419757 tok/s); retain E024-only C8 at170.619844. Both p50/p95 global discrete frontiers remain{oMLX C1,E024-only C8}. All38 observations stay separate; no combined-switch claim. See E029-gdn-http-formal-v1/RESULTS.md.

GLM's actual first CPU registration/read-only service preparation/seal passed against this restored predecessor. New plan b842f85f04b32ed84cb0903d943ac86b04f1cf7ae6ef2c1f3df4f5848e59af01 uses existing original 8/32/32/32 target-only probe and observed child220GiB/system240GiB/cold40GiB limits; these are sampled stop thresholds, not hard allocation caps or a proof that the model fits. No GPU GLM result yet. M4 remote remains32f21ae, already merged. M5 is preparing a default-OFF minimal GDN+E024 composition while root performs the separate GLM load experiment.


## 2026-10-18 — GLM first cold load failed; current service independently verified

The first GLM-5.3-Flash (`glm5_next`) device attempt did not finish loading or generate a request. After 37 complete physical samples, a seven-command owner checkpoint took 1.598319 s and exceeded the fixed 1 s observation cadence; its own 5 s aggregate budget had passed. The preceding physical probe took 0.244151 s. Per-command times were not retained, so no individual command is identified as the cause. Direct child71812 was TERM→KILL and strictly reaped−9. No model fit, correctness, quality or throughput claim follows.

Original Qwen service27245 was replaced by72953. The recovery HTTP monitor then failed on849,347B swap growth and its worker was reaped−15; the original journal remainsLOAD_INTENT withpost_monitor_failed=true and recoveryreturncode1. A separate authorized read-only check verified actual72953 identity/job/port, idle loaded target, unchanged persisted settings/model metadata, residentPLE/MTP completion logs, old-process absence and two stable current physical samples. Its distinct status isOPERATIONALLY_RESTORED_WITH_MONITOR_FAILURE; it does not repair the original monitor or assert a successful HTTP response. A future lifecycle needs an explicit observed-baseline adapter.

[Complete first-attempt archive](../experiments/m5-ultra-256gb/GLM-flash-load-v1/README.md) preserves nine evidence/input trees and exact observed-baseline receipts:477members,6,956,676regular bytes,599,184compressed bytes,tarSHA5b2ed880db1baf4a7f0013a488416238d09a9887514f399c7df635662765dad9. Every member and original was rechecked. One model-directory symlink is retained only as metadata; no model weights/runtime binaries are included. All previous Qwen performance results remain unchanged.


## 2026-10-18 12:24 PDT — V135 received; GDN/graph composition running

Fetched M4 `0fd8e8f8236ffff476cbedbc9efbf964428a4f90` and read its V135 handoff. Merged and pushed as `2092522`; the change is confined to documentation, chapter-one notebooks, their frozen reference files and `.gitignore`. No active benchmark/native source changed. Notebook execution/rendering claims remain M4 evidence; M5 did not rerun the MPS notebook during its GPU experiment.

The actual combined GDN + E024 numerical child74841 started at12:10:19 PDT under owner74808. Original42 parameter diagnostics preceded the request run; all8 C1 requests are now complete and C8 has passed through step500, but the37-request terminal/independent audit is still pending. No combined performance point exists. The current global discrete frontier remains oMLX C1 plus native E024-only C8. Source preparation and29 CPU checks were pushed in `c776f51`; new HTTP adapters separately passed14 CPU mechanisms after retaining an initial asyncio-guard harness failure. Neither result grants HTTP/device performance qualification.

Next work is the actual combined HTTP history and independent replay, followed by observer-OFF paired timing if qualified. In parallel the GLM monitor now separates physical sampling from slower metadata inspection; its new owner is CPU-tested but has not retried GLM loading. Fixed resource thresholds are unchanged. The current Qwen cycle will supply a real independently-audited RESTORED predecessor if it closes successfully.


### 12:32 PDT — E030 independently closed; next combined HTTP cycle started

E030 combined GDN+E02442/37 completed with all producer/owner/restoration/resource/numerical gates PASS. Independent ledger counted5,163steps,6,622,533full-array bit checks and13,245,066finite checks;12compiled tracing events,5,144reuse and38eager fallback events were reconstructed. No discarded-prior-ahead was observed. No HTTP/performance qualification follows from these numerical receipts. Service72953 recovered80649 and all three driver codes were0.

E030-gdn-graph-combined-serving-v1 retains969members,1,660,774,222raw bytes and9,958,428compressed bytes, archiveSHA592190ee9d708db4b3d35c392d0e7aa65b7517165f5f6831a093da2b8cdaeaeb. Every member/originalSHA was checked. At12:32:47 PDT actual combined HTTP owner81295 started the next four-phase C1/C8 capture→reap→independent replay cycle under plan7783b4c35798d590f1a326739c38a410ac59a3d0655cd384813d6e5b9f0a02e8. Current performance frontier is unchanged; real paired timing follows successful HTTP qualification.


### 12:45 PDT — V136 received; C1 combined HTTP passed

Fetched and read M4 `94c070e6bbf0cf0e229261f1d019fd0b2032c4a7` / V136, merged and pushed as `fe8bb96`. Only chapter-two notebooks and documentation changed; the running inference source pins remain intact. M5 did not execute those MPS notebooks, so their validation remains M4 evidence.

Combined HTTP owner81295 has completed and reaped both actual C1 capture81355 and C1 numerical replay81912, each producer PASS. C8 capture87227 is running; the independent whole-cycle qualification is pending. The next formal experiment is already source-prepared: same combined server in both arms, route graph always ON, GDN OFF/ON, three fresh process pairs each at C1/C8, observers OFF,8warm+64formal per process. Factory12 CPU checks passed; lifecycle+independent-audit58 passed after preserving the first runner's five asyncio SIGINT guard failures. No combined performance point exists yet.

The separate B2/B8 local GDN canary now has real native batched activation/independent three-step state preparation and12 CPU checks; device execution is still pending. This does not qualify B3–7, dynamic row retirement/refill, full-model serving or HTTP. ReaDy source compatibility findings are retained separately; the square recurrent-state axes require a transpose and its trained DSpark draft contract is not interchangeable with our current DFlash2 path.


### 12:54 PDT — E031 HTTP closure PASS; actual timing preparation PASS

Both actual combined HTTP histories and numerical replays passed:24 requests/12,288 committed tokens,5,124steps/6,573,486 full-array bit comparisons. All four children strictly reaped; original80649 restored90614 with all driver codes0. Independent C1, C8, resource and restoration audits PASS. E031 retains1,100members/1,875,145,734rawB/11,470,864xzB, SHA742a5988d72924beedaf5bf24914413997e569ed09bf2fb38c046b33623c6639. No combined performance result yet.

Actual12-spec CPU preparation now PASS for the incremental GDN-on-graph experiment, owner planSHA3e65c89baecc67b14e8f97c2063d4647085a8ca86c2cc8e6c48dd03577bd6d4b. Both arms use the same combined entry with graph ON, GDN OFF/ON, observers OFF,3pairs per C1/C8,8warm+64formal per process. Root is binding the verified restored service before execution.


### 12:57 PDT — E032 launcher failure retained; retry being prepared

The root-written formal driver incorrectly passed `--run` to the formal owner's argparse. It exited2 before owner-directory creation or any model child:0 warm/0 formal requests. The preceding70 CPU checks did not cover this new driver. Stop returned0; the normally waited CLI permitted `recover --no-experiment`, which returned0. Service90614 recovered91373; the separate independent restoration auditPASS. No numerical/performance result. E032 retains182members/8,519,786rawB/305,300xzB, archiveSHAab6f4d24cbb68e3e0c554e8d351dba558f4aaa004e0d9cc15a817cb27d575a85, every member/original verified.

A new retry-v2 lifecycle will bind that exact closed predecessor, preserving the original failed source and all12 original timing specs/factory algorithms. The corrected driver builds owner argv once and runs the actual `--prepare-only` CLI before service stop. Retry is not yet device-started. GLM next loading remains prepared but unrun; it must bind the actual later retry closure, not assume success. Current measured Pareto is unchanged.


### 13:07 PDT — corrected paired timing is actually running

Pushed3fbc028 before launch. Retryplan f4e19bd0a842b48ea26fb2d27f870fdf62409897c7f862067c7800de5949b6f1, receipt76ba44e5aecd499fe272759501a59a02cbfb3e9857b97e4ecd7417da26ad3e40. Actual CLI prepare returned0 at20:06:54UTC before stop; stop returned0 at20:06:57UTC. Owner93089 started first fresh C1-OFF server93184, which completed warmup and entered formal requests. No final timing result yet. All12 original measurement specs/factory are unchanged. Prior19 focused CPU checks and exact E032 predecessor passed; driver source/plan/evidence were registered before execution. Root is following the actual cycle through recovery and independent audit; source-only agents are preparing a native verifier bridge, duplicate-submit candidate and the next chart renderer without competing device work.


### 13:27 PDT — formal timing continues; next experiments remain source-only

The first actual phase, `r1-c1-off`, completed its 64 formal requests and reaped
server93184. Its unaudited point is45.7519054414 output tok/s, full-request
p50=11.1893236875s and p95=11.2451506250s. Owner93089 is running `r1-c1-on`
server97648. This is one partial point, not a new qualified result or frontier.
The same12-phase run and external restoration continuation remain active.
A fresh fetch of `tutorial/18-gdn-projection-packing` found no unmerged commits.

Source-only preparations for the next cycle are in separate default-OFF bench
modules: `exact_verifier_bridge` (native sequential decode versus existing
prefill target verification; no rollback/features qualification),
`gdn_graph_combined_c24` (eight-request C2/C4 full-batch and continuing-tail
coverage), and `progress_v16` (refuses to render new performance before actual
E033 closure and independent audit). New C24 test source fixes the inherited
non-C1 16-request assumption only in the new namespace. These new suites have
not run during formal timing. The C1 diagnostic HTTP entry's retained earlier
fake checks are documented separately; no diagnostic GPU profile exists yet.

GLM v3 now selects the actual frozen retry-v2 controller/owner/auditor sources
and E033 v2 journal schema. Its actual predecessor manifest and dependent seals
remain UNSEALED. The previous E032-bound seven-test archive is preserved and
does not qualify this new adaptation. No GLM registration/loading is occurring
concurrently with Qwen timing.

### 13:52 PDT — six formal phases complete; remaining C1 gap localized in HTTP time

The actual retry-v2 owner has completed and reaped six of twelve processes and
is running `r2-c1-on`. No first-error receipt exists at this checkpoint; full
independent audit and service restoration are still pending. The first C1 pair
is45.7519054414→48.3190609013 output tok/s (+5.6110%), p50
11.1893236875→10.5913308745s. C8 pairs are170.3079257357→170.3789778802
and170.4667730669→170.6520740629; these small differences do not establish a
repeatable incremental C8 gain. These are partial, unaudited points, not a new
frontier or a declaration of an adopted configuration.

An independent small timestamp diagnostic compares all64 requests in each first
C1 group with the historical oMLX r1 C1 group, preserving the exact eight fixtures,
model and512-output count. Mean E2E is10.595848264s versus10.082137592s; the
0.513710672s gap decomposes into TTFT0.074106186s and post-first-output0.439604486s.
The latter is85.5743% of the mean gap, or0.860283ms per511 subsequent tokens as a
diagnostic amortization, not observed ITL or GPU time. The new GDN gain itself is
almost entirely post-first-output. Historical external data is not a simultaneous
paired comparison; current full-cycle audit remains pending.

Small default-OFF bench candidates now cover one-pass long-graph argument
collection, suppression of a provably already submitted ahead root set, native
B1 four-to-one GDN projection, and layer-major rowwise C1 verification. Their
new CPU test sources are still unexecuted during the formal timing isolation.
The rowwise proposal retains ordinary C1 arithmetic and every prefix state;
weight-cache reuse is only a hypothesis. New C1 diagnostic capture/owner glue
also remains source-only and still requires the registered outer cold lifecycle.
None of these preparations changes the running frozen inference source.


## 2026-10-19：用户目标、E033完整结果与下一轮

用户再次明确：目标是手写针对两模型/M5的specialized engine，正确性不变，
每个目标batch都大幅领先oMLX/Lithos。图必须有清晰的三引擎前沿线和测点；
HTTP并发与实际GPUbatch分别标明，未知点不填造。没有完成目标，不能停在局部略快。

E033十二进程、864请求正式/恢复独立审计PASS。C1 graph+GDN ON中位数48.35777tok/s，
p50 10.58590s/p95 10.63603s；OFF45.81897，三配对增益5.53–5.66%。
仍低于历史oMLX50.78157约4.77%。C8 ON170.37898/OFF170.46064，不采用稳定加速结论。
此前E024-only170.61984仍是Native C8最好点，仅高于历史oMLX约1.34%。
E033归档与031教程提供完整证据；v16纳入所有50观察、18组，C2–7和GLM正式前沿仍未知。

本次fetch仍为M4分支94c070e，已合入，无新未合入提交；这不意味着M4 agent已经阅读本消息。
下一步继续默认OFF候选的实际数值/成本验证，GLM使用实际E033恢复证据封存新加载周期。


E034实际更新：GLMv3在加载过程中因系统swap+231.62MiB/Swapouts+14880被原门槛拒绝；
非OOM，0请求/0tokens。子进程182.45GiB/系统203.92GiB，服务21938→34623恢复且独立PASS。
官方loader线程在取消中返回不等于EnginePool加载成功。下一步查wired/allocator/临时加载成本，
同时准备Qwen C1的三个profile模式。详见032教程。前沿仍是v16，GLM没有速度点。


### 2026-10-19 — large-change candidates and actual GLM layout preparation

Qwen C1 profiling completed all48 requests across off/runner/full; actual driver returned0 and service34623 restored39988. Independent restoration and three-mode closure audits both PASS. Detailed interval analysis is in progress; these observations do not create new Pareto points. The mission remains a small specialized engine with repeatable, substantial wins at every target batch, correct original outputs, Qwen3.8-27B + GLM5.3Flash.

New `nvfp4_multirow_reuse` is107lines/defaultOFF: T2/4/8 share each loaded/dequantizedweight across separate B1-order accumulators, instead of relying on adjacent launches hitting cache. Five guarded CPU mechanism tests passed; no Metal, arithmetic, full-prefix or speed qualification. GPU testing must compare native MLX B1 AND customB1 independently, then completeprefix state.

GLM offline exact-byte repacker now passes15 guarded CPU checks including actual Darwin F_NOCACHE unaligned IO and exact pinned container/MTP/DSV32 fakebyte paths. Actual headerplan SHA323ff8da5d2faf6bb774b1c03f3a52f49198cbb9898f21c91ac25aa546a4c50a passed:113446→4960tensors,378backbone stacks,203976457080payloadB. MTP45 remains byteidentical and still uses nine officialstack tensors. Independent review found no blocker. Payloadcopy will run only after this commit, into a fresh~190GiB derived checkpoint with fullsourceSHA+independentreadback; original checkpoint stays unchanged. Actual loader/memoryfit andGLM speed remain unqualified.

Fresh fetch found no unmerged M4 commits; its branch remains94c070e. These Git notes make the result available, without claiming the other agent has read it.


E035 complete readout: repeatedcurrent-root async4072calls total18.523ms(0.02191%); successor submission mean18.620ms, long-wrapper minuscompiled interval0.4609%; activeHTTPdelivery99.831%overlapsworker. These tinywrapperchanges are not the main substantial-speedup direction. Backenddependency/encoding and deviceexecution remain unresolved. Tutorial033 and completeE035archive/staticdiagnostic saved; no newParetopoint.


### 2026-10-19 — E036 / E037 实际完成，目标保持大幅逐 batch 领先

GLM无损离线重排真实退出0：113446→4960张量，378个expert stacks，58分片，
203976457080B原始payload；完整source SHA和输出回读通过。实际官方loader和内存fit仍待测。
多行NVFP4候选真实GPU smoke36/36通过，三路BF16输出字节一致，MLX peak约68.2MiB；
未做整模型或性能资格。原oMLX服务已恢复，PID39988；正式前沿仍v16。
见[034教程](../optimization-guide/tutorials/034-real-weight-reuse-and-glm-layout.md)及E036/E037归档。

实际head形状适配新kernel，但head仅约占4.96%的逻辑投影权重字节，不能用head-only
通过代表骨干网络权重复用或巨大收益。下一步以真实模型合同验证最小接入面，再覆盖大头投影。
源审查见[head接入分析](../../bench/reviews/2026-10-19-multirow-head-integration.md)。
继续按手写specialized engine推进：减少权重/状态流量和设备等待，模块可开关；
每个目标batch都要正确且明显领先oMLX和Lithos，局部略快不算完成。
本次fetch的M4分支仍无未合入提交。以上通过Git同步，不声称M4 agent已阅读。


### 2026-10-19 — E038/E039/E040 实际结果：保留小候选，先筛掉不赚钱的复用

E038第一版register多行NVFP4复用数值全部通过，但六case×两基线的十二个median
baseline/candidate ratio全部小于1，因此因速度拒绝，旧head full-model计划不自动跑。
Head T2/4/8对原生为0.931513/0.922243/0.820917；失败性能结果和首轮离线reader
null-metadata错误、修图历史均已归档。原candidate和原计划保留。

E040换成K5120专用shared tile：两输出列共5,760B权重/scale，token各有SIMD group，
B1算术不变，仍默认OFF；每token仍各自解码，不能说只解码一次。
Head T4/T8对独立native M1为1.169216×/1.258057×，对custom B1为
1.095522×/1.136031×，全部五组均大于1；head T2仍慢，up仍mixed。
六组原始BF16全词表/投影字节finite+equal，2,400samples逐条独立重算通过。
[图和完整证据](../experiments/m5-ultra-256gb/E040-shared-projection-screen-v1/README.md)；
[035教程](../optimization-guide/tutorials/035-shared-weight-screen-and-load-boundary.md)。
两个局部screen各自在暂停原oMLX期间执行并恢复；不是正式引擎、speculation或HTTP收益。
M5设备/MLX身份有显式guard，M4不能直接去掉guard就宣称已适用，应独立移植和资格验证。

E039真实58分片GLM派生checkpoint进入官方loader后再次触发原swap增长门槛，
+1,247,805B/+76 Swapouts；0完成请求/0token batch事件，无loader返回或loaded-engine证明。
子进程严格回收，stop0/owner1/recover0，RESTORED，独立恢复PASS，服务替代为52818；
诊断仍FAIL。E036离线字节重排证明仍有效，但尚未解决实际加载，未放宽gate。
[E039证据](../experiments/m5-ultra-256gb/E039-stacked-glm-load-v1/README.md)。

E041 tiny CPU file-like I/O已实际3/3 PASS_BYTES_ONLY并提交327441c：276B fixture、
U32/F16/BF16原字节保留，两个file-like关闭fd后CPU eval/save仍成功且无后续I/O事件。
不代表大模型加载或内存收益；cached/uncached file-like对照仍需独立验证。
[小型I/O证据](../experiments/m5-ultra-256gb/E041-glm-filelike-io-smoke-v1/README.md)。

E042 shared-head实际producer已完成：九case/42prefix/九head调用PASS，58.171206s；
47个子进程物理样本、lease闭环、服务52818恢复。独立字节审计也实际退出0，
96原始快照、5418输出tensorpairs、全部42prefix与1349参数值收据通过。
父v2计划SHA55dbd68bb64a80b3091e629a20b6915a19d3ab92fb92c0b29417700aaefa6b7a；
父v1仅CPU准备未运行并保留，模型/子计划冻结。40项CPU准备与实际设备资格分别保存。
仅forward_prefix_qualified=true，full_model/compiled/rollback/HTTP/performance均false。
[E042 metadata归档](../experiments/m5-ultra-256gb/E042-shared-head-prefix-v1/README.md)包含474成员；
96份24,476,249,870B原始hidden/state/logits保留在M5本地，附path/size/已审计SHA，
不把22.8GiB打包进Git。下一道门是verifier总成本与接受/回滚/HTTP范围，未有加速结论。
正式前沿仍[v16](../optimization-guide/progress/2026-10-19-v16/README.md)：C1低于历史oMLX
约4.77%，C8最好点仅高约1.34%；C2–7和GLM正式点未知，目标尚未达成。
本条仅追加Git交接内容，没有执行新的fetch/push，也不声称M4 agent已阅读或已测收益。


### 2026-10-19 — E043 native M8 shared wide：正确但更慢，拒绝采用

一次native M8矩阵调用作基线，新shared wide保留其kl16运算合同，不混用八次M1。
真实head/up权重、合成输入两case输出全部BF16 exact/finite；独立复算400原timings，
head native/candidate=0.5458300112，up=0.6937056746，十个组ratio全部<1。
[92成员完整小证据](../experiments/m5-ultra-256gb/E043-native-m8-shared-screen-v1/README.md)
保留CPU8/11、源、实际执行及原输入/输出，真实模型slice不入Git。服务52818已同身份恢复。
候选默认OFF，因速度拒绝；没有full-prefix/HTTP/engine速度点，也不继承E042资格。
E040的逐行B1基线与本轮nativeM8不同，不把两个ratio拼成配对回退图。
热重复权重的缓存驻留未知，不据此断言整模型流式复用一定无效；但这版实现不值得采用。
035教程已记录该负结果，正式v16保持原样。此追加没有执行fetch/push或声称M4已阅读。


### 2026-10-19 — E044 nv8 register负结果、退出竞态与最新Git同步

新默认OFF candidate直接用native wide helper的nv8寄存器tile，仍64threads、无shared/barrier。
两真实projection输出相同且有限，400rawtimings独立核验；head native/candidate
0.7647025469，up0.8177880323，全部十组<1，拒绝采用。
[原证据及E043/E044合图](../experiments/m5-ultra-256gb/E044-native-m8-register-screen-v1/README.md)。
数值child57173已waited/0，但原父natural-exit identity竞态FAIL，清理中断lease正常恢复，
原lease−2/closedfalse完整保留。Watchdog恢复原52818，之后fresh只读核对同身份/S、
全部health与两owned进程不存在，状态仅PASS_RESTORATION_ONLY，不能覆盖原FAIL。
新父v2有CPU5再7项PASS，无实际设备重跑；历史源/失败均留档。归档首次遇到未来测试
5→7源码版本差异而停止，保留后按各自快照重新封存。没有改写测量或升级性能资格。

Root已实际fresh fetch（ae4469），origin/tutorial/18-gdn-projection-packing仍为
94c070e6bbf0cf0e229261f1d019fd0b2032c4a7，并核对为当前HEAD祖先（108499）；
没有新的M4未合入提交。这是Git可用性记录，不声称M4 agent已阅读。
正式v16/逐batch目标与缺测范围不变；此归档步骤没有新增GPU实验或推送。


### 2026-10-19 — E045 C1/C8配置扫描结束，E046局部数值实验准备

C1 MB200原始49.140388tok/s，比两个MB50基准至少快1.5002%；p50速度至少1.4787%，
低于预设3%，不升级正式点。C8基准自身warm/probe输出4/8 fixture不同，其他MB配置也有差异；
计时漂移约0.42%通过，但输出合同FAIL，171.965865tok/s不能算有效新收益。
两个独立SSE reader复核160请求；原因未定位，后续用精确scheduler trace固定数值路径再对照。
不要把跨schedule差异直接归因MB，也不要从已有固定schedule资格推断batch invariance。
原52818同身份恢复、400资源样本swap增长0，所有证据含失败封存E045，正式v16不变。
见[036教程](../optimization-guide/tutorials/036-concurrency-specific-configurations.md)与
[E045图/归档](../experiments/m5-ultra-256gb/E045-concurrency-config-screen-v2/README.md)。

E046 B2/B8 GDN prework canary准备完成，owner12项CPU通过、现场预检查通过，尚未GPU。
保持五个native batched projections，仅复用已有conv/activation/QKnorm融合，默认OFF；
不把B1收益移植给B8，也不当作E045差异修复。当前父plan08325135、child7df6cff9。
[Kapitsa/M5设计阅读](../../bench/reviews/2026-10-19-kapitsa-m5-designs.md)记录按并发选speculation深度、
verify与commit分离；其affineQ4与我们的NVFP4不同，不能引用其102tok/s作同条件比较。
本条是可随Git推送的交接记录，不声称另一agent已阅读。
