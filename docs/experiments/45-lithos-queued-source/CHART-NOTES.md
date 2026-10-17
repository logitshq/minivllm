# 九配置吞吐—延迟图

`plot-data.json` 的 native/oMLX 六配置及18个fresh arm逐字保留第44章V128的对应数值；Lithos三配置及9个fresh arm从V130原始27个正式group时钟重算，并与本章安全`lithos-perf-data.json`完全相同。旧V128 Lithos OpenAI C1没有混入。

- 两个独立AC campaign，同M4 Max40GPU/48GiB、Qwen3.8-27B NVFP4、19→32greedy；不是新同轮配对或完整三家27arm campaign。
- X为每组request POST→终止p50，再取3formal group和3fresh arm两级中位；Y是组输出/(last terminal−first POST)，同层级取中位。V128终止为SSE DONE，V130为NDJSON done:true；时钟都含排队、排除cleanup。
- 9个配置大点、27个fresh小点，坐标不jitter；C1放大框重复展示三个配置以保持左侧可读。小点不是置信区间。
- 只对9个中位点作描述性dominance计算。黑线连接native C1/C2/C4；各家虚线连接已测C1/C2/C4，不证明线段中间配置可行，没有跨campaign配对改善/统计显著性/全局采用声明。
- native/OM/Lithos配色沿用第44章。圆/方/三角表示C1/C2/C4；填充大点表示当前九点中非支配，空心大点仍保留受支配配置。
- 19→32为实际短负载。官方raw IDs不可见，只有usage/可见文本重分词门，图不代表新增数学资格。V130官方FIFO仍单generation，不是GPU batch2/4。

重画（CPU，需要Matplotlib）：

```sh
python3 -B plot.py
```

本机制作环境是Python3.14.7/Matplotlib3.11.2，Agg后端，DejaVu Sans。PNG与SVG使用同一源。`plot-source-map.json`保留旧44数据、V130原suite/27formal groups、安全V130数据的精确SHA与原始路径；只记录身份，没有复制launch/env/argv/headers/token值或私有wire内容。

安全组/请求复算：`python3 -B verify.py`（数据作者提供，stdlib离线）。图层文件由`chart-manifest.json`独立列SHA，避免改变数据包的`manifest.json`。旧44原图及全部结果不修改。

实际view_image检查：完整图与C1细节的九个配置标签、两轴单位、模型名、campaign标注、脚注均可读，无裁切；SVG由同一renderer生成。根线程也已查看PNG并要求补模型名，最终副标题已显示Qwen3.8-27B。
