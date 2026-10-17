# V130 queued Lithos: portable saved-result data

离线复算：`python3 -B verify.py`。仅需Python标准库；不导入MLX、不启动服务、不访问原权重或私有结果路径。`prepare_publication.py`是从原私有保存文件提取本包的源码，重做提取需要那些原文件；普通读者只需运行verify。

- `requests-normalized.json`：134条，资格8 + 性能126。性能再分warm63 / 正式63；只有正式63请求/2016tokens用于排名。
- `groups-normalized.json`：54个性能组（27warm、27正式）+4资格组。`lithos-perf-data.json`仅包含9个性能arm和C1/C2/C4各3fresh的中位数。
- 主延迟：客户端pre-POST→收到官方NDJSON `done:true`，含服务内排队，cleanup除外。每组request p50→每arm3正式组中位数→每C3fresh中位数。总吞吐以组内32×C输出/(最晚terminal−最早prePOST)，同层次聚合。server eval_duration/total_duration只作原始数值记录，不代替客户端时钟。
- C是并发HTTP请求数。官方Ollama adapter通过原FifoLock票序串行generation；没有自定义队列、新vendor源码、GPU batch2/4或HTTP网络到达公平性证明。
- 资格的OpenAI C1与Ollama C1/2/4全部visible32/19prompt/32output/length匹配旧OpenAI可见流。官方内部raw IDs未暴露，`internal_raw_ids=null`保持。资格首次OpenAI冷启动约11.8秒不是性能点，不能拿来排名或计算协议加速。
- 所有134条usage/count与134条原engine Generation的cache0/prefixNone证据范围分开：NDJSON缺少cache字段不补0。9个性能arm实际各有6×C个Generation记录、cached_prompt_tokens0、prefix_sourceNone。
- 每arm receipt里的 `no_new_performance_point=true` 复用了资格worker元数据；它不否认driver已保存3warm+3measure。原值以 `inherited_no_new_performance_point_flag` 保留并解释；正式统计来自本轮driver的实际group时钟。
- 本轮实际固定电源是AC，9arms均如此。原CPU准备README里的Battery示例没有被当作实际电源记录。175个资源白名单投影=资格11 +性能arm135 +全局campaign27 +root释放后2；离散样本均AC/thermal0/pressure1，保留swap与所有数字vm counters。原guard核Swapouts/swap-used不增加，未因此宣称Pageouts为零或连续无瞬变。8个60秒卸载后间隔在请求时钟外。
- `releases-selected.json` 是root保存释放文件的白名单，性能11PID/9port、资格3PID/1port，原始SHA可追溯；没有新live probe。owner脚本无环境/实际argv导出。

该Lithos切片与V128是**两次独立campaign**，不是native/oMLX的同轮配对。图层可清楚标注后描述性并列，但不能pool成同一campaign、按配对百分比推论、或把旧OpenAI C1和新Ollama C2/C4拼成同协议曲线。本包本身不计算跨campaign胜负/全局最优。

`source-v130/`保留实际client/driver/qualifier/protocol计划原字节快照。`source-audit/`是既有CPU来源证据摘录；`source-identities.json`链接218原始pin及关键vendor文件SHA，已重验，不重复vendor树。`originals.json`逐个保留派生数据私有原文件的SHA/长度；实际raw argv、环境、headers、鉴权值、进程清单和输出文本不导出。

本包verify是作者的可移植保存行复算，不冒称独立raw审计或GPU重算。V20的独立审查报告由root另行纳入；后补图/独审文件不在此数据包manifest覆盖范围，分别按各自COPY-LIST核对。`manifest.json`及包根`COPY-LIST.json`完整列出本包payload。
