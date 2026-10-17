V128 独立保存结果审计：PASS，可在下述范围使用。
只读取冻结源码和原始保存结果；未运行 GPU、服务、live ps/socket 或修改 W/Git。
从全部原始 POST/DONE 时钟重算 126 组：每组 request p50→每 arm 三组中位→每配置三个 fresh arm 中位；Y 每组总 output/(last DONE−first POST)，再取两级中位。
21 fresh arms、270 HTTP/8640 outputs；其中正式 135 HTTP/4320 outputs。另 3 次启动 ordinary warm/96 outputs 仅有源码断言及运行计数，不能称已保存 raw IDs。

| 配置 | 完整请求 p50 ms | 总 output tok/s |
|---|---:|---:|
| native:C1 | 1226.301292 | 26.094729 |
| omlx:C1 | 1274.119292 | 25.115388 |
| lithos:C1 | 1461.269166 | 21.898772 |
| native:C2 | 1470.631771 | 43.511700 |
| omlx:C2 | 1570.562146 | 40.154138 |
| native:C4 | 2011.992146 | 63.378995 |
| omlx:C4 | 2306.974000 | 54.379935 |

实测联合非支配点恰为 native:C1/C2/C4；所有测试的官方配置均受支配。12 个同 C/同轮次配对均改善两轴（保留全部原始有符号值）。原 suite.full_frontier_win=false 是预审保守标记，原文件未修改。
C1 是 CAP64/direct/深度2 静态入口，每 arm 186 completed/delivered、6 reset、bank96/90、48 S/83 direct；C2/C4 是原 accepted25/CAP256 静态入口，实际 formal B=2/4、warm hook 已移除，非运行中迁移。
126 native HTTP 有 raw32：C1 等 canonical，C2/C4 等预声明同 C 历史合法 stream；144 官方 HTTP 只有 visible32/usage，不能声称 raw IDs 等价。normalized events 不是 wire bytes。
19 controller+234 runtime refs 当前字节重验；官方实际 preimport tree/解释器/文件数与冻结 registry 相符（不是复验当前 2688 个 oMLX 文件）。1349 参数 metadata 和 idle 清理核验通过；本轮未新增全模型 tensor/参数 byte-readonly 资格。
359 离散资源样本均 AC/thermal0/pressure1/knownforeign0，均在请求时钟外；swap 5038.06→5030.06 MiB，Swapouts 不变。20 次固定 60s 卸载后冷却均在时钟外，不是自适应重试。
18 Lithos usage cached 缺失保留 null；另 18 条引擎 metrics 的 cached_prompt_tokens=0，来源分开。
全部 21 server 退出/端口回收和 owner/driver rc0 wait-reap已核；root 新外部 release 精确覆盖 23 PID/21 port。审查者只读保存证据，没有另做 live absence。
结论仅限本轮 M4/Qwen3.8 NVFP4/19→32/静态配置集；不是全工作负载、统计显著性、连续无热瞬变、自动采用或新 kernel 资格。
复算：python3 /private/tmp/minivllm-engine-optimization-20261014/v128-actual-independent-review-build1/audit.py
