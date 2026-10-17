# M4 三引擎 native15 历史进度快照（2026-10-17）

主图：[PNG](native15-throughput-latency.png) / [SVG](native15-throughput-latency.svg)。仅Qwen3.8-27B NVFP4 g16/b4、M4 Max48GiB、19输入→32输出，真实HTTP客户端时钟。这里是旧完整21-arm轮次，非最新本方候选或完整Pareto前沿；未混入M5或GLM数字。

| 引擎 | 并发 | 完整请求延迟 p50 ms | 总输出 token/s | 旧 first-text 速率 token/s |
|---|---:|---:|---:|---:|
| native | C1 | 1273.094500 | 25.135605 | 27.951541 |
| native | C2 | 2799.683687 | 22.628166 | 13.487015 |
| native | C4 | 3291.085916 | 38.886488 | 12.200735 |
| omlx | C1 | 1259.769708 | 25.401468 | 29.646924 |
| omlx | C2 | 3115.389792 | 20.303770 | 13.156233 |
| omlx | C4 | 3640.560083 | 34.445485 | 11.623697 |
| lithos | C1 | 1470.179625 | 21.766048 | 29.488619 |

Lithos C2/C4为已记录429并发边界，N/A；本轮没有这些实测点，不补零、不连成可行曲线。

主x由实际保存的start_perf_counter_s和done_perf_counter_s重算：每个measure group先取请求duration的p50，再对每arm三个group取中位，再对每配置三个fresh arm取中位，乘1000为ms。y每组为实际B×32输出token/(最晚SSE DONE−最早POST start)，同样group→arm→配置中位。包含排队、prefill、HTTP；清理发生于DONE之后，不计入时长。旧x按首个非空文本块剩余重建token数/到DONE时长重算并单独保存，绝非请求latency倒数。

全部21arm小点和7配置大点保留，不构造置信区间或显著性结论。每arm为3warm+3measure groups；126groups/270requests/8640tokens，计时部分63groups/135requests/4320tokens。每请求19/32 usage、normalEOS下finish_reason=length、无prefixcache均核对。官方可见文本/原生ID的旧资格差异不在此图重判；白名单不导出文本或IDs。

源：m4-final-native15-official-attempt2/output的21个validated及suite-complete，共22份JSON递归检查无凭据/argv/settings/raw文本后逐字节复制到aggregate-summary。逐arm报告按validated保存的SHA严格验证，只提取离线重算必需的270条白名单时间/计数/标签，不复制原报告。source-map.json保留原路径、SHA和角色。client.py:80–112/142/164定义时钟和旧坐标；本图没有新请求或推理。

旧资源样本357个（非独立重复）：thermal0 136、thermal1 221；AC与pressure1各357。图保留混合thermal与重复波动限制，未用资源分组挑点或推断因果。suite内旧GPU_released=false是当时聚合写出时状态，不是当前GPU占用判断。

文件：reviewed_rows.json与arm_rows.csv/configuration_medians.csv为已审数据；group_rows.json保留126组；request_timing_whitelist.json/.csv保留270条允许字段。all_measured_request_p50_ms_appendix_only只作附录，不作主x。verification.json记录计数/精确复制/时钟重放；QUALITY.md记录实际视觉检查。

重算（CPU stdlib）：

    /Users/ryanli/code/minivllm/.venv/bin/python -B collect_rows.py

绘图复用已存在的Python3.14 CPU依赖与中文字体，无包安装：

    env PYTHONPATH=/private/tmp/minivllm-engine-optimization-20261014/progress-figure-dependencies-build1 MPLCONFIGDIR=/private/tmp/native15-status-mplcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/homebrew/bin/python3 -B generate_figure.py

Matplotlib3.11.2/Agg；字体STHeiti Medium，SVG嵌字形路径。未读权重、运行MLX/GPU、改W或Git。
