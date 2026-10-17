# Chapter 44 evidence

本目录只包含 V128 新完整 campaign 的公开安全投影、精确源码快照、离线复算与图。没有模型权重、二进制、实际启动 argv/environment、鉴权值、原始进程清单或输出文本。`requests-normalized.json` 保存全部 270 个请求的白名单时钟、usage、raw-ID/visible-ID；它不是 POST/SSE wire capture。

## Reproduce offline

```sh
python3 -B verify.py
python3 -B plot.py
```

`verify.py` 只用 Python 标准库重算 63 个正式组、21 个 arm、7 个配置与 12 对百分比、Pareto 支配关系，并核对所有 manifest payload。`plot.py` 需要 Matplotlib；本次使用 Matplotlib 3.11.2/DejaVu Sans/Agg，详情见 `render-metadata.json`。无引擎/GPU依赖。源码及渲染脚本没有运行模型。

## Aggregation

- latency = `1000 * median_3fresh(median_3formal_groups(median_requests(DONE - POST)))`；POST/DONE 是原客户端 `perf_counter` 记录。
- throughput = `median_3fresh(median_3formal_groups(sum(output tokens) / (last DONE - earliest POST)))`。
- config-median ratio 与 median-of-signed-pair-percentages 分别保存；负值未过滤，虽然本轮 12 对主指标均为正。
- native 126 HTTP / 4032 tokens；official 144 HTTP / 4608 tokens；合计 270 / 8640。正式 135 / 4320。另有 native C1 启动 ordinary warm 3 次 / 96 tokens，在 HTTP campaign clocks 外；总执行 273 / 8736。
- C1 三家 NOL/OLN/LNO；C2、C4 native/oMLX 各 AB/BA/AB；21 fresh，不含旧125/旧C4。20 个固定 60s unloaded gaps，逐条保存于 `cooling/`。

## Source and qualification boundaries

`source-v128/` 是本轮 driver/owner/protocol/plan/pins 的原字节快照；`source-v126/` 是不可变静态入口、逐字节 V125 C1 委托服务与原 native15 batching 核心。`source-identities.json` 保留 234 个 runtime 与 19 个 controller source SHA，全部从原文件核对；旧数学/runtime 依赖不重复打包，二进制和 runtime-config 值不导出。`originals.json` 记录每个被读取的实际证据文件的 SHA/长度。

V126 冻结的 profiles 中 `HOLD/GPU_UNRUN/qualification_pending` 是其准备时的原字节标签，未追写旧文件。资格结论以随后 actual126 receipt/release/独审，以及本轮实际 warm/formal 路由证据为准。V128 suite 中 `full_frontier_win=false` 同样是未经独审自动宣告胜出的保守占位；真正的 7 点支配计算保存在 `plot-data.json` 和独审记录，不修改原 suite。

C1 继承实际103双模式资格，CAP64/direct FIFO；warm93与final186、6reset、ICB[96,90]、0discard/0failed/idle分别核对。C2/C4 继承 accepted25 数学及新126实际入口资格：同C合法 raw stream、真实 B>1、warm hook 拆除后的 reentry、1349参数对象不变。正式段不挂 GDN 路由观察器，结束后检查原有 step trace 及 idle；不能套用 C1 的186 pipeline 计数。参考 IDs 只作输出断言，没有喂给后续解码。

## Null/cache/resource semantics

raw IDs 只在 native 暴露；official 的 `output_ids_when_available=null` 保持 null，visible roundtrip32不能改称其内部 raw32。Lithos usage 不暴露 cached_tokens 的记录保存 `null/not_exposed_in_usage`；另有 engine Generation 的 cached_prompt_tokens0 证据，不以客户端 `.get(...,0)` 冒充实际 usage 字段。旧 first-text interactivity 若缺失仍保留 null，不填0；主 latency/throughput 独立重算。

359 条资源是 357 arm + 2 owner、位于请求组以外的保存采样。公开投影仅保留电源/thermal/pressure、Swapouts、swap-used 和数量等白名单，不复制原始系统命令、进程 argv 或环境。通过的是采样处 AC/thermal0/pressure1 与 Swapouts/swap-used guard；不因此宣称 Pageouts 为零、整个请求期间连续监测或某个物理因果。allocator cache 与 prefix/KV cache 不是同一设置；各原服务设置保留，不假称三家 allocator 一致。

owner 是 3600s wait-only，有 TERM grace/kill bound；root 另做外部释放核验，23 PID / 21 ports 严格 absent。`owner-selected.json`/`external-release-selected.json` 是白名单投影，原始 SHA 可追溯；没有虚构独立 lease 文件。

## Publication provenance

本包作者同时是 V128 harness 作者，离线复算不冒称独审。独立 actual 审查由另一个 agent 完成，原字节报告将随最终封包纳入 `independent-review/`。公开复制清单在包根 `COPY-LIST.json`；复制后仍需 root 核对索引内每个文件字节，而不只核本地目录。
