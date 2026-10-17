# E023 · Lithos D512 六进程正式实测

2026-10-17 15:02–16:26 PDT，M5 Ultra 256 GiB。**正式计数/吞吐延迟、资源、原服务恢复独立审计全部 PASS。** 预注册 `3078d7b` 在停服务前已 push；原始失败资格 v1 和通过资格 v2 均保留。

固定 Qwen3.8-27B NVFP4 revision `5ff8ef1`、原八个 D coding fixtures（886–978 prompt tokens）、greedy、normal EOS、cap512。六个 fresh 官方 Lithos 进程按 C1/C8/C8/C1/C1/C8 运行，每进程 8 warm + 64 formal。实际 48 warm、384 formal，共 196608 formal committed tokens；此次全部到 cap，early-EOS 覆盖仍为零。observer OFF，官方 FIFO；C8 是同时等待的 HTTP 请求数，不是 GPU batch8。

| 进程 | committed tok/s | POST→DONE p50 (s) | p95 (s) |
| --- | ---: | ---: | ---: |
| r1-c1 | 44.910185 | 11.404540 | 11.521853 |
| r1-c8 | 44.883987 | 91.252783 | 91.313418 |
| r2-c8 | 44.967055 | 91.072059 | 91.186858 |
| r2-c1 | 44.947629 | 11.397076 | 11.517165 |
| r3-c1 | 44.901685 | 11.402589 | 11.531824 |
| r3-c8 | 44.916329 | 91.183519 | 91.324838 |

三个进程指标中位数：C1 **44.910185 tok/s、p50 11.402589 s**；C8 **44.916329 tok/s、p50 91.183519 s**。吞吐分母为第一条 formal POST 至最后一条 DONE；延迟含服务端排队，不含等待客户端并发槽。所有实际 Generation 条目 cached prompt=0，formal setup=0；日志窗口和条目数已独立核对，但日志本身没有 HTTP request ID，不能声称逐 ID 绑定。

已有原生同 D512 点：C1 45.372147 tok/s、p50 11.283287 s；C8 167.398907 tok/s、p50 24.431815 s。对比 Lithos 进程指标中位数，原生 C8 吞吐约 3.727×，p50 约 0.268×；C1 吞吐差约 1.03%。原生每档 n=1，Lithos n=3，来自不同轮次，未配对、无置信区间。新图保留八个实际点，不制造中位数采样点或插值前沿。**oMLX D512 仍待实际资格与正式实测；不是全网最快或最大并发证明。**

## 资源、恢复与证据范围

999 个资源样本通过原始 pressure1 / AC / thermal / sustained-swap 门；swap growth=0 MiB、Swapouts=0、Pageouts=0，Swapins=41、Decompressions=28556。六个实际 server 全部 wait/reap、严格 PID/port absence。未知 runtime 或 GUI GPU 活动并未被完整排除。

oMLX 从 PID28531 恢复为 **PID85613**；原模型、持久配置及新的 MTP/resident-PLE 日志与可见状态通过独立恢复审计。恢复 POST worker 已退出，19 个物理样本中系统 footprint 最大 175299117056 bytes。未恢复暖 KV，也不声称精确内部 runtime signature 恢复。

正式资格依赖此前真实 route/prompt-ID/raw-count/text/cache 检查；本次 observer-OFF wire 本身不含全部原始 token ID。没有跨框架 full-state 或模型输出质量等价证明，这些不能由性能审计替代。

## 可复核归档

`evidence.tar.xz`: 2695 members，3769816 bytes，SHA256 `ed1e515d0cce3bfc537c7ac27644d561af020697389290b41270f0d30c167d92`。全部 member hash、原始文件和目录成员复核 PASS。保存完整五个 evidence trees、预注册、冻结审计与实际依赖源码；模型/pack/venv payload 不复制。官方 source inventory 中已知 native `.so` 只保留原 inventory 和 owner pre/post 校验回执，打包器不另行复制或重hash该二进制。

[独立正式审计](independent-formal.json) · [独立恢复](restoration-audit.json) · [恢复 journal](restored-journal.json) · [汇总](summary.json) · [逐文件清单](manifest.json) · [归档验证](archive-verification.json) · [归档说明](ARCHIVE.md) · [Pareto 图与原始坐标](../../../optimization-guide/progress/2026-10-17-v8/README.md)

下一步保持默认 OFF 的 E024 route-preserving 图在真实 ModelRunner 上检查原37请求、42次参数替换/恢复与全部状态。只有通过后才测 HTTP 收益；既有局部前向加速不加进本图。
