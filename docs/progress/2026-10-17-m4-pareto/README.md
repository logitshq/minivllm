# M4 throughput–latency Pareto（旧 native15 轮次）

[PNG](m4-throughput-latency-pareto.png) / [SVG](m4-throughput-latency-pareto.svg)。只使用已冻结历史reviewed_rows；未重审原raw或重跑模型。

正确性是入图先决约束，不作为坐标。目标为降低完整请求latency、提高总输出throughput。联合非支配点只有 **oMLX C1（1259.769708ms,25.401468 token/s）** 与 **native15 C4（3291.085916ms,38.886488 token/s）**。native15 C2被自家C1支配，保留空心点并明确标注。

每家已测前沿：native15 C1/C4；oMLX C1/C4；Lithos仅C1。7个配置中位点和全部21fresh-arm小点保留。粗虚线为联合前沿；细线为各家已测前沿；连接只标出实测端点关系，不代表可行插值或额外实测配置。Lithos C2/C4因已记录429边界N/A，无零值点。

latency来自真实client POST start→SSE DONE：每measure group request p50→每arm3group中位→每配置3fresh-arm中位。throughput为组B×32/(最晚DONE−最早POSTstart)，按同样层次取中位。Qwen3.8-27B NVFP4 g16/b4，M4Max48GiB，19→32，旧轮次thermal0/1混合且有重复波动；不声称统计显著性、最新候选胜出或完整frontier覆盖。未过整模资格的最新spec候选不入图；未混M5/GLM数字。

输入input-reviewed-rows.json为前一冻结快照原字节副本；pareto.json保留每点联合/本家支配者和非支配判定。generate_figure.py重算并assert联合/各家前沿与每点支配关系，使用Matplotlib3.11.2/Agg，中文SVG嵌字形路径。正确性资格沿用历史campaign验证，本图不新增跨引擎逐字节等价证明。

CPU重绘：

    env PYTHONPATH=/private/tmp/minivllm-engine-optimization-20261014/progress-figure-dependencies-build1 MPLCONFIGDIR=/private/tmp/native15-status-mplcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/homebrew/bin/python3 -B generate_figure.py

实际视觉QA：最终PNG已查看，标题/轴/图例/前沿/7点标注/右表/脚注可读，无裁切；修正了旧脚注和C1标签压线。SVG同figure导出且XML可解析。无GPU/权重/新请求/W/Git操作。
