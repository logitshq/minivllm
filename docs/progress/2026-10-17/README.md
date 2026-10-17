# M4 Max 优化进展图 · 2026-10-17

这组静态图保存两个已经发生的结果：官方none/MTP的三对C1计时，以及同497个输入的Linear局部数值检查。不是完整Pareto前沿，也不把局部exact当作整模修复或速度提升。

- [进展台账](progress.md) / [机器可读台账](progress.json)
- [C1配对图 PNG](c1-none-vs-mtp-paired.png) / [SVG](c1-none-vs-mtp-paired.svg)
- [Linear局部数值图 PNG](local-linear-exactness.png) / [SVG](local-linear-exactness.svg)
- [已审数据、定义及receipt SHA](reviewed_rows.json)

C1每个点是一台fresh服务的3次测量中位数，保留3个pair的全部6个arm。图中的x是基于首个非空文本块重编码的客户端交互速率估计；MTP首块可能含多个token，不能将其误标为普通逐token decode吞吐。y是整请求输出token/HTTP时间窗。V111共36请求，其中18条计时、576个计时输出token；不池化请求、不构造置信区间。AC/thermal0/pressure1，但保存记录有swap_used瞬变和Pageouts增长，不能称零分页。

局部图计数单位是同一个实际first-verify block中497个Linear输出对照，不是497次整模运行。V113重现V112输入及旧输出hash，然后broadcast-M1对serial-M1全部逐字节一致。完整序列/缓存资格、实际PSO和性能仍属后续工作。V114因资源压力停止，A96通过而B未到达；V115在本快照仅CPU就绪。

## 重绘

已提供PNG和SVG；SVG使用字形路径以便离机显示中文。`generate_figures.py`只读取同目录的已审JSON，使用Matplotlib Agg在CPU绘图，无MLX/GPU依赖。

```sh
python3 generate_figures.py --rows reviewed_rows.json --out . --font /path/to/chinese-font.ttc
```

本次渲染使用Matplotlib3.11.2和系统STHeiti字体；具体信息见 `render-metadata.json`。绘图依赖未复制进此目录。可用任何兼容Python/Matplotlib环境重绘；不是原引擎运行环境。

`collect_rows.py --source-root /path/to/private-evidence` 可在持有原件的源机重新提取白名单聚合行并核验约束。它不复制argv、settings、原始流文本、参数数组或tensor。离机可直接读取 `reviewed_rows.json` 和已导出图；本快照不包含全部113/114/115运行原件，receipt SHA用于追溯，不声称离机重做数值资格。

建议仓库位置为 `docs/progress/2026-10-17/`。复制时使用 `publication-files.json` 白名单；不要复制本地字体cache目录。此目录只保存本次进展，不创建定时或监控任务。
