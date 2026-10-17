# 优化进度与图

主目标是**在通过正确性检查的配置集合中，优化throughput vs完整请求latency的Pareto前沿**。正确性是入图条件，不是优化轴；局部数值PASS不能算性能进步。主图必须显示实际吞吐、请求起点到DONE的延迟及不被支配的配置，保留被支配点供比较。

每个有实际结果的检查点保存：教程的前后代码对比、代码版本及SHA、全部正负结果、当前进度台账和适用的图。通过源码审查、局部数值、整模数值及端到端性能是不同阶段，不用一个“完成百分比”代替。

已有的日期快照保持原样；同日后续结果另存检查点，Git commit保留演进。每次显著数值／性能结果之后commit/push、fetch M5交接，并更新本方交接。性能图只使用同机、同权重、同负载与相符资源条件的结果；M4/M5、局部/整模、不同电源和不同轮次分开。

- [M4 throughput–latency Pareto主图](2026-10-17-m4-pareto/README.md)：正确性为先决条件；联合前沿、各家前沿与被支配配置均标明。
- [M4此前完整三家对照的延迟—吞吐图](2026-10-17-native15/README.md)：旧轮次重新核算真实HTTP完整请求时长，保留21个arm、22份聚合原件与全部白名单时钟；不是新候选跑分。
- [2026-10-17 图与台账](2026-10-17/README.md)：oMLX none/MTP三对HTTP计时，以及同输入Linear局部数值图。PNG/SVG、已审数据和生成脚本均保存。
- [当前模型与三家对照范围](MODEL-PLAN.md)：Qwen3.8与GLM-5.3-Flash。
- [M4→M5交接](../handoffs/m4-max-to-m5-ultra.md)：已实际推送、接收的源码事实与下一步。
- [39：广播M1教程](../experiments/39-verify-broadcast-m1.md)与[固定代码快照](../code-versions/39-broadcast-m1/README.md)。

[40：整模通过后再测端到端性能](../experiments/40-m1-verify-and-bounded-loader.md)记录 V117 严格整模通过、V120 两次加载资源中止与 V122 小缓存诊断。普通路径单组测到 1250.37 ms / 25.59 tok/s，投机路径尚无请求性能；不混入旧前沿，完整 Pareto 胜出尚未成立。

native15快照README保留作者在源机运行的命令。其`collect_rows.py`依赖未公开的完整arm报告与原目录布局，离机不能把这条命令当作完整采证重放。公开时间戳白名单和聚合表足够复算图中延迟与吞吐；重新绘图先复制快照到新目录，并准备Matplotlib及适用中文字体。`generate_figure.py`从脚本所在目录读数据、写图，当前字体路径为macOS STHeiti；不要直接在冻结快照中覆盖原图。

[41：TCP_NODELAY HTTP 对照](../experiments/41-native-http-nodelay-pilot.md)新增首对负结果：OFF 为 1227.177 ms / 26.0761 tok/s，ON 为 1230.205 ms / 26.0119 tok/s。按冻结计划停止，未采用；两臂同机 AC、thermal 0、pressure 1。该内部单因素筛查单独保存，不替换上面的三家对照前沿。
