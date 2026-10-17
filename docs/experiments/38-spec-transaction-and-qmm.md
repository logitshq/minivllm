# 38：先保证投机验证能提交正确的历史

这一轮解决两个问题：我们的 verify 能否提交正确的缓存前缀；官方 oMLX 打开外部 MTP 后，在本机是否更快。实际结果是：缓存事务的小规模检查通过，整模旧 token 序列检查失败；官方 MTP 数值资格通过，但这个短请求的 HTTP 配对测试更慢。没有新的完整 Pareto 胜出，也没有修改默认引擎。

模型仍是 Qwen3.8-27B NVFP4，实际架构为 Qwen3.5；M4 Max 40 GPU cores / 48 GiB，MLX 0.32.3。前章见 [37](37-simd4-async-and-spec-handoff.md)，精确源码与公开运行证据见[本章归档](38-spec-transaction-and-qmm-source/README.md)。

公开证据包含187项精确源码身份、417份原字节运行文本及32份凭据字段脱敏JSON view。脱敏只移除localhost benchmark临时API key；PID、时钟、统计及其他字段不变。原SHA保留为private-original身份，离机模式验证公开view，源机 `--originals` 额外验证私有原件及字段变换。

## 1. verify 一次算完，不代表可以随便留下最终缓存

普通解码每次计算一个已知 token，得到下一 token。投机解码先提出几个草稿，再让目标模型一次验证多个位置。若只接受前两个位置，就必须留下“刚好算过这两个位置”的状态。

Attention 的 K/V 有 token 轴，可以保留相应前缀。GDN 的递归状态 S 会被每个 token 更新；只保存块末尾 S，无法直接切回块中间。先前的 V107 因为缺少同一次计算产生的中间状态而保持 HOLD。

V108 保留批量权重投影，让块内 GDN 逐 token 调用原 V44 helper，并保留每次实际返回的 S、conv tail 和 producer 所有者。下面只是契约示意，精确实现以归档代码为准：

```python
# 之前：块末状态不足以提交任意中间前缀。
posterior = verify(block)

# V108：计算时保存真实历史，再按交付边界提交。
posterior, transaction = verify_with_literal_history(block)
transaction.commit(keep)
```

`keep` 受输出预算、拒绝位置和 EOS 共同约束。最后交付的 token 成为下一次尚未计算的 anchor，不能因为已经交付，就假装它已进入缓存。keep=0 恢复原状态；attention 未用的物理尾部也必须恢复或清零。

T=1/3/5/8、全部 keep=0..T 共 21 个 GPU 小例通过。异常案例在 producer 之前触发，部分 producer 执行后的异常回滚没有获得 GPU 覆盖。整模检查中，普通 A 的三请求共 96 个旧 IDs 通过，但投机 B.base 在 index 4 由旧 60277 变成 100606，硬门失败。

这个 B 的完整 32 IDs 恰与官方 oMLX 输出一致，仍不满足本轮预先固定的旧序列要求。失败保留；B 失败后的参数只读、最终清理及 provenance 检查没有执行，不能写成全部通过。进程的回收与外部释放另有记录。

## 2. 同一输入，批量乘法与逐行乘法也可能不同

V112 先重现 V108 的实际首次 verify 输入和 128 个起始缓存签名。原 verify 完成以后，才建立替代诊断图。每个 hook 都返回原输出，因此后续算子继续接收原轨迹的输入；诊断比较的是各算子对**同一输入**的两种计算方式。

```python
batch = linear(x)  # x: [T, K]
rows = mx.concatenate([linear(x[i:i+1]) for i in range(T)], axis=0)
# 严格比较 batch 与 rows；不把 rows 喂回原 verify。
```

978 个捕获操作形成 1,138 项对照。497 个 Linear 中 146 个逐字节相同、351 个不同；161 个 RMSNorm、48 个 GatedRMSNorm，以及所检查的 gates、SiLU、RoPE、两种前缀 SDPA 对照全部相同。

最早差异是第 0 层 GDN 的 out_proj：输入 [5, 6144]，输出 25,600 个 BF16 元素中 5 个不同，最大绝对差约 4.768e-7。这把下一步集中到量化矩阵乘的形状与归约选择；尚未观测实际 PSO，也没有证明仅修 Linear 就能通过整模旧 IDs。

下一候选 V113 拟把 x 表达成 `[T, 1, K]`，并给 packed weight 增加广播批次轴，检查能否保留逐行 M=1 的数学。M5 的 E010 用类似广播保留每条序列的 M=T；两者目的与计算形状不同。此处仅记录计划，V113 不在本章实际结果中。

## 3. 官方 MTP 确实参与了计算，但本负载更慢

V109 先验证原官方 CLI 的 none/MTP 两个 fresh 服务，各 3 请求。外部 MTP 约 239 MB；target 本身没有这套 MTP 权重。六条 raw32 均与 fresh 普通官方控制逐值相同；每个 MTP 请求实际 17 次 verify、34 个提案、15 个接受、32 个交付。加载、attach、路由和提案计数有交叉证据，未把 soft fallback 当成通过。观察器版本的时间不排名。

V111 随后移除观察 hook，直接运行原官方 CLI。固定三对顺序 none→MTP、MTP→none、none→MTP；每个 fresh 服务 3 warm + 3 measure，每个 fresh 服务卸载回收后、下一服务启动前固定冷却 20 秒，共五次。总计 36 条 HTTP 请求、1,152 个输出 token，其中正式计时 18 条、576 token。

| C1 配置，19 输入／32 输出 | x：首文本后速度，tok/s | y：整请求吞吐，tok/s | TTFT，秒 | 整请求，秒 |
|---|---:|---:|---:|---:|
| oMLX balanced，none | 28.987 | 24.765 | 0.219 | 1.292 |
| oMLX balanced，外部 MTP，默认 block=3 | 18.651 | 17.233 | 0.193 | 1.857 |

表为三次 arm 中位数；逐对有符号变化的中位数是 x −34.85%、y −29.59%，与“中位数之比”是不同统计量。三对均更慢，全部保留。MTP 的 TTFT 较短，整请求较长；普通解码仍是这两个已测配置中吞吐更好的一个，不声称全局最佳配置。

计时接口未提供 raw IDs，保存的是完整可见文本重新编码的 32 IDs；原始 raw32 与参与证据来自同配置 V109 资格。本次日志也记录 MTP 路由与统计，但固定分母不是新的逐次实际提案观测。

115 个资源原始样本均为 AC、thermal=0、pressure=1，Swapouts 不增。一次 swap_used 短暂增加约 32.07 MiB，Pageouts 总计增长约 3,064 页；通过原先的资源门，不等于零分页，也不能用它解释某一对的速度。六个服务与外层 driver/owner 均已回收，并保存端口/进程消失证据。

## 4. 两台机器交换什么

已实际 fetch M5 分支到 `490bb25`，阅读其 E010 广播实现与默认关闭的 E012 GPU 草稿选路模块。M5 已合入我们的 `4862215`，并在交接中明确保留 V108 FAIL 与 V109 的数值边界。

M4 继续定位 verify 数学并验证事务；M5 的 E012 减少草稿选路中的主机同步，已有其本机三进程模块资格。组合前仍需 M4 数值与整请求资格。持续往返记录放在 [M4→M5 交接](../handoffs/m4-max-to-m5-ultra.md)，不合并两机性能数字。
