# 030：为什么还没全线超过——从 C1 请求时窗定位差距

2026-10-17。本次只复算已经闭合的真实 HTTP 记录并审读执行路径，不运行新模型，不新增速度点。模型是同一 Qwen3.8-27B NVFP4 快照；结论不覆盖未完成资格的 GLM。

## 当前结论

[v13](../progress/2026-10-17-v13/README.md) 的同并发点估计显示：C8 Native ON 吞吐比 oMLX 高1.34%，p50低1.19%；C1吞吐仍低9.74%，p50高10.77%。已测的C1/C8上均领先Lithos；当前联合前沿包含oMLX C1和Native ON C8，不能称为全线超越，也不是全网最快证据。

代码体积不是运行时成本。每个 token 的主要成本由实际执行的投影、状态读写、归约、kernel提交和不能重叠的等待决定。通用框架中有价值的专用快路径仍然需要学习和验证。

## 先定位缺口，再归因

[可复现原始时间戳分解](../../experiments/m5-ultra-256gb/c1-gap-decomposition-v1/README.md)分别使用两边3×64条正式请求。下表是请求**算术均值**，不能替换Pareto图的进程中位数/p50。

| 客户端区间 | Native ON | oMLX | 差值 |
|---|---:|---:|---:|
| 总请求 | 11.175 s | 10.083 s | +1.091 s |
| 首个非空输出前 | 0.636 s | 0.564 s | +0.072 s |
| 首个非空输出后到DONE | 10.539 s | 9.519 s | +1.019 s |

93.4%的均值差发生在首输出后。这个区间包括后续计算、调度、求值等待、文本解码和流式交付；它不是GPU profile，也不能把93.4%归给GDN。两边流式合并粒度不同，更不能把非空SSE事件数直接当作生成token数。优先调查真实生成流水及交付，继续只做prefill优化不足以解决主要差距。

## 已确认的执行路径差异

原有[CompiledDecode](../../../tutorial_code/miniengine/engine/compiled_decode.py)把packed/fused路径限制在`used <= 256`。本次D提示长度886–978，当前[E024长图](../../../bench/m5_ultra_256gb/route_decode_graph/candidate.py)调用普通模型，不传`_decode_projections`；[原长GDN](../../../tutorial_code/miniengine/model/gdn.py)仍包含独立卷积、q/k归一化等预处理步骤。因此`--fuse-decode on`不等于此次长上下文已采用该融合路径。

oMLX实际正式进程日志记录`Qwen B1/T1 fused GDN prework engaged`。这证明一项具体快路径已启用，但日志没有给出其独立节省的毫秒数。不能据此承诺它解释全部差距。

[E029](029-unbound-gdn-prework.md)保留native五个投影、BF16卷积舍入及原状态归约，把预处理融合为可独立开关的模块。它已通过实际37请求全状态数值门，尚未通过真实HTTP资格/性能测试，也尚未与E024组合验证；因此尚未进入当前Pareto。

## 为什么局部约6%没有变成HTTP约6%

[E024局部计时](../../experiments/m5-ultra-256gb/E024-route-timing-v1/README.md)测B1/B8和used1023/1024，每次forward后求值全部输出状态并同步。HTTP已有ahead流水，真实资格覆盖used887–1489；两者测量边界、长度分布和已有重叠不同。局部比例不能直接移到HTTP。

真实HTTP资格C1共4096个调度step、4088次实际图调用，C8共1028个step、1027次调用；ahead使两者并非逐step一一对应。这不支持“多数长decode没命中图”的说法。正式计时观察器关闭，不能冒充有正式逐调用profile。当前也没有可归因本次D512的CPU/GPU耗时占比。

## 下一项可证伪实验

1. 在真实C1完整512-token请求中，保持现有ahead、不新增同步，测已有调用边界、图域、ahead消费和已有求值等待；先确认测量自身开销。这定位CPU/等待区间，仍不等于GPU kernel独占耗时。
2. 完成E029单开关HTTP捕获及同历史exact回放，再按既有正式框架进行OFF/ON配对D512计时。E024/E028保持OFF以隔离假设。若C1无实质收益，保留负结果并检查归一化、状态流量与attention路径，不把正确性PASS当性能PASS。
3. 单项确有收益后，再验证E024+E029的组合正确性和性能。两项独立通过不能直接相加。C1单序列attention实际长度/无mask是后续候选；既往改变SDPA长度曾改变数值，不能跳过验证。

本轮的改进是把优先级收紧到已观察到的C1缺口，并保存可复算证据；没有宣称新优化已经更快。
