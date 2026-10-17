# E015：M5 官方 external MTP 数值与路径资格

**PASS，范围仅为 instrumented EnginePool API。没有 HTTP 性能结论。**

M5 Ultra 256 GiB；oMLX 0.7.1.dev1、MLX 0.32.3、mlx-vlm 0.7.7、mlx-lm 0.31.4.dev132+g94cdcae13。目标 `Qwen3.8-27B-nvfp4` revision `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`，external MTP revision `b9b764dada894b8c416f89cbc8cf99bc7e20b831`。

同一个 19-token 原始 prompt，greedy、32 个输出、正常 EOS、跨请求 prefix cache 关闭。普通和 MTP 各一 fresh process，每进程顺序处理 3 请求。六条原始 token 流逐项一致，共比较 192 IDs。普通官方控制是 oracle；本结果不能外推到其他 prompt 或长输出。

| 每请求的真实观察 | 官方普通 | 官方 external MTP |
|---|---:|---:|
| 请求数 | 3 | 3 |
| 输出 raw IDs | 32 | 32 |
| MTP verify 次数 | 0 | 16 |
| 提案 token | 0 | 31 |
| 接受提案 | 0 | 15 |

每条 MTP 请求经历真实 load、engine/scheduler attach、负 uid 路由。前 15 次 verify 宽度为 3，最后一次因输出预算宽度为 2；没有为满足固定次数而改 block。raw events、`draft_lens` / `accept_lens` 与实际 proxy 累计计数链逐项一致。所有原始轨迹均保留。

运行前的交叉审查发现一个观察器缺陷：oMLX `_MTPResetBindingProxy` 的计数赋值停留在 proxy 上，underlying drafter 的列表由引用共享，数值 counter 却始终为 0。修复后记录实际 round 参数的对象身份与每次 before/after，独立比较器还检查换 proxy、counter 断链、错误 prompt、错误 raw ID 和 verify 宽度。此缺陷在首次 GPU 资格前修复，无失败 GPU 试跑被隐藏。

10 份资源样本全部 AC、thermal 0、pressure 1。swap 使用量恒为 268.81 MiB，Swapouts 增量为 0。两个模型 child 均 exit 0、wait/reap，并再次确认为 absent；暂停的外部 omlx-server PID 74384 身份一致且恢复到 S。进程检查覆盖已知可执行程序与 exact owner，未知 runtime/GUI GPU 使用未被证明排除。

审计重新哈希 2,301 个官方安装文件，两个 post-load loaded-source snapshot 分别含 381 / 384 模块，与对应 distribution hash 一致。归档包含 380 个相关已加载官方 Python 源文件和本次 harness，native binary 与模型权重保留 hash，未复制内容。两臂运行前均完整读取模型权重 SHA；CPU 后审没有再次重读权重。

- [summary.json](summary.json)：合同、版本、逐请求计数及资源概要。
- [audit.json](audit.json)：独立比较、源码与清理审计范围。
- [cpu-checks.json](cpu-checks.json)：8 项 corruption tests 的完整输出。
- [manifest.json](manifest.json)：441 个归档成员，全部大小与 SHA 验证通过。
- [evidence.tar.gz](evidence.tar.gz)：全部原始结果、events、owner/lease 记录及冻结源码，2,152,208 bytes。

归档 SHA-256：`fa9c1b32e5dff53d481229e976a6542882b126222a8f0436cfffc35541413df5`。

下一步是使用未安装观察器的原始官方 CLI 做 fresh HTTP 配对计时；本次观察器每步同步落盘，任何 wall time 都不进入排名。
