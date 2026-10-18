# 优化教程：六篇，十八个小步骤

[返回01—15基础教程](../README.md) · [原30章逐章对应表](coverage.md) · [完整实验档案](../../experiments/README.md)

基础篇已经搭起模型、cache、scheduler和服务测量。接下来每节只解决一个主要问题：先看原实现，再看替换点，最后确认数学、状态和实际结果。**六篇是六个主题，不是把30章内容挤进六章。** 优化部分分成18节；完整学习路线是15个基础章，再读这18节。

原16—45的30个实验编号全部保留，其中一些编号原本就有讲解、阶段报告等多个文件。短章整理核心机制和前后对照，复杂推导、完整代码、全部试验和失败日志从每节末尾进入原记录。因此只读短章得到的是主线；要研究某次实现，仍须展开对应原文。46—47保留为后续研究，不算已完成优化。

## 六篇分别回答什么

| 篇 | 小节顺序 | 读完能回答的问题 |
| --- | --- | --- |
| A：先测清楚 | [A1 时间边界](a1-timing-boundaries.md) → [A2 公平比较](a2-controlled-comparison.md) | 慢在哪？这次加速真由代码改动造成吗？ |
| B：融合准备工作 | [B1 SiLU与拆头](b1-postconv.md) → [B2 卷积与tail](b2-conv-fusion.md) → [B3 投影与门](b3-gate-fusion.md) | 哪些中间操作能一起做，哪些舍入边界必须保留？ |
| C：复用计算与存储 | [C1 Q/K范数](c1-norm-reuse.md) → [C2 输入与权重](c2-weight-reuse.md) → [C3 状态buffer](c3-state-buffers.md) | 什么能共享？省下读写后，谁持有正确状态？ |
| D：减少每步主机工作 | [D1 命令复用](d1-command-reuse.md) → [D2 设备反馈](d2-device-feedback.md) → [D3 服务所有权](d3-service-ownership.md) | 怎样连续生成，并在EOS、取消和新请求之间安全复用？ |
| E：让投机路径正确且值得用 | [E1 前缀事务](e1-spec-transactions.md) → [E2 逐行M1](e2-verify-m1.md) → [E3 草稿加载](e3-draft-loading.md) → [E4 完整成本](e4-spec-cost.md) | 接受的token与提交的状态如何对应？正确之后为何仍可能更慢？ |
| F：回到用户感受到的性能 | [F1 HTTP计时](f1-http-measurement.md) → [F2 Pareto](f2-pareto-configurations.md) → [F3 排队与batch](f3-queue-vs-batch.md) | 和oMLX、Lithos比较时，两根轴和每个点到底代表什么？ |

按 A1→F3 阅读即可。B3先把准备Q/K/V和scale的函数当作已有接口，C1再解释其中范数怎样复用；D篇使用C3建立的状态所有权。E篇还依赖基础第10章的投机概念。若只是想核实当前比较结果，F2/F3分别直链同轮图与补齐Lithos并发后的图。

## 每节怎样读

1. 找到 **baseline → 本节改动**，先说清改的是哪层：数学操作、内存、提交还是HTTP测量。
2. 阅读关键节选或明确标注的伪代码；源码链接打开完整实现，不能把伪代码当运行入口。
3. 检查不变量：有限精度的求和顺序、状态更新、所有权和计数是否保持。
4. 看结果的范围：局部数值、整模输出、生命周期和完整请求性能分别证明什么。
5. 回答理解题；需要继续推导时打开末尾的原章和固定代码。

读前可查[术语表](../../experiments/README.md#3-读后半部分前先认识这些词)：C是客户端并发，B是模型batch，T是本次计算位置数，CAP是物理缓存容量；ICB是可复用的Metal命令缓冲。实验中用到的“资格”指指定范围内的正确性检查，不是所有输入的证明。

## 代码版本仍按原来的演进保存

这18节重排的是教学顺序，没有产生18个新的引擎版本。基础01—15继续使用原tag；优化节直接链接原tag、冻结source目录及before/after diff。B、C、D的排列也不表示所有候选最后都一起启用：未采用、默认关闭和只在某路径启用的状态分别写明。

例如B1查看旧融合实验，C1查看范数复用实验：

```bash
git fetch origin --tags
git diff tutorial-15-v1 tutorial-16-experiment-code -- tutorial_code/miniengine
git diff tutorial-15-v1 tutorial-17-static-reading-v1 -- tutorial_code/miniengine
```

这些命令只显示各自相对基础的变化，第二条diff不是“接着上一条自动升级”。复现须按原章的固定版本、模型与条件运行；具体worktree方法见[基础目录](../README.md#用固定版本阅读避免后面代码混进前面)。本次目录和短章单独提交文档commit；后续新增优化继续独立提交代码、正确性结果、教程与性能证据，不移动旧tag。

## 内容保留与后续研究

[对应表](coverage.md)逐项列出原16—45的主要知识点、短章去向和仍需展开的细节；[机器可查映射](coverage.json)同时保存原文SHA256。[结构检查脚本](check_structure.py)可离线检查30个编号、18节、原文未变和新文档链接，不加载模型。

[46 论文到实验](../../experiments/46-next-ideas-from-papers.md)与[47 近期论文精读](../../experiments/47-recent-paper-reading.md)另列研究。论文机制、待做方案和已测结果继续分开；本次重写没有新性能点，也没有改变原前沿。
