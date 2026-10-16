# 28：把卷积、归一化复用和 state 保留接到真实 ICB

第27章把主机逐步编码换成真实双 bank Metal ICB，V38 的 B-only 探索为36.188 ms/步。本篇继续记录三份独立冻结源码：V39 只加入已验证的真卷积融合，V40 组合紧凑 q/k 归一化复用与 `held[32]`，V44 把两者接起来。三版分别通过27B完整数值资格；V44 本次 B-only 中位为 **34.699 ms/步、28.820 decode token/s**。

这些是独立进程的绝对计时，尚无配对 A/B、HTTP、C2/C4 或 Pareto 获胜结论。实验候选默认关闭，当前主引擎没有切换。runner应用、HTTP电源切换STOP及后续同步实验另卷保存。

## 三版实际执行的内容

| 版本 | GDN路径 | 实际原生实现 |
| --- | --- | --- |
| V39 | V23真卷积 + BF16 SiLU + raw tail；原GDN递推 | 已资格的V38 ICB，原字节不变 |
| V40 | 原concat/product/sum卷积；紧凑q/k预处理、norm复用、held32 | V40 native build2，仅扩大准确的GDN函数/绑定资格 |
| V44 | V39卷积输出已激活值，接已激活norm预处理和V40 held32 | V40 native build2，原字节不变 |

每版候选都来自原accepted25参考包，参考manifest为 `31fa0539…cb41c`；V39保留其冻结父包的其他文件。完整文件身份以[逐版本manifest与provenance](28-native-gdn-composition-source/provenance.json)为准，不把版本名称当作源码等价证明。

V39的卷积仍保留每tap BF16 product、原Y4归约分支和native BF16 SiLU语义；新tail由旧tail后两项加本次raw projection组成，不能把激活值反馈进tail。第21章的局部micro没有替代这里的完整模型资格。

## V40省去重复的norm，并在递推中保留32项state

原递推按48个value head及多个value tile运行，q/k来自16个key head。预处理先保存紧凑q/k/v，对每个key head按原顺序串行累加128项norm，再保存两个scale。递推按 `h / 3` 读取对应q/k及scale，保留原 `SPLIT=4`、`VALUE_TILE=32`、每段升序32项及最终升序4段归约。没有改成新的SIMD求和树。

[V40状态差异](28-native-gdn-composition-source/prototypes/v40-gdn-normreuse-cache-build1/state.diff)把原来的两次源码读取：

```metal
kv += ks * kh[d] * S[base + d * Dv + dv];
// 完成四段归约、算出delta之后：
float s = ex * S[base + d * Dv + dv] + ks * kh[d] * delta;
```

改成第一次访问时保存，之后使用保存值：

```metal
held[j] = S[base + d * Dv + dv];
kv += ks * kh[d] * held[j];
// 完成原顺序归约之后：
float s = ex * held[j] + ks * kh[d] * delta;
```

`math_mode="safe"`、norm表达式、scale、递推算式与barrier顺序保留。`held[32]`是源码层面的值复用；本轮没有测寄存器分配或spill，不能保证编译器一定省掉一次物理内存读取。

## V44必须避免第二次SiLU

V40的入口消费卷积后、激活前的BF16 `u`，预处理内部执行typed native SiLU。V39真卷积核已经输出激活值，因此直接接V40会再做一次SiLU。V44只改预处理的输入读取：[原始差异](28-native-gdn-composition-source/prototypes/v44-gdn-activated-normreuse-build1/prework.diff)把 `v18_bf16_activation(u[...]).activated` 换成 `activated[...]`，state核的body、名字和求和顺序不变。

实际组合调用如下，完整来源见[V44 GDN](28-native-gdn-composition-source/candidates/candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine/model/gdn.py)：

```python
activated, new_tail = conv_silu_raw_tail(qkv, st.conv, w)
o, S = gated_delta_step_activated_norm_reuse_metal(activated, g, beta, st.S)
st.conv, st.S = new_tail, S
```

gates、最终gated RMSnorm和out projection仍按原路径执行。候选的 `set_conv_norm_reuse_cache(True)` 只对普通B1 decode启用，其他batch走原实现；whole-model functional compile未启用。

## 先保留V40第一次真实绑定失败，再收修复

V40 native build1把新state PSO的 `S_new` index假定为8。第一次真实tiny capture显示 `S` 为只读index6、`S_new` 为写index9，eligible为0，wrapper在任何replay之前失败退出。原始报告与stderr都在证据包中，没有重标为PASS。

[actual-index-evidence.json](28-native-gdn-composition-source/prototypes/v40-native-icb-build2/actual-index-evidence.json)保留实际反射记录。build2只把该精确函数白名单的输出index从8改成9；仍检查同次dispatch、参数名、access、整allocation及无其他别名。不能单靠Python输入数组数量推断Metal绑定位置，自动生成签名还可能使用shape元数据。

修复后tiny2的3请求、15步、75项严格检查通过。V40/V44整模捕获均实际证明48个S pair eligible、83条其他反馈链、双bank ICB；没有丢弃原CB/fence/event边界或GPU反馈依赖。

## 数值与计时证据

V40和V44各自用9个历史真实卷积后bank及6个synthetic bank完成30项严格FP32 o/S比较，外部输入保持只读。历史bank本身不是新preconv捕获；后续完整模型才执行真实projection、tail和卷积组合。

V39/V40/V44各加载一次模型。A临时绑定原accepted25 `_decode`，三个请求为原19-ID prompt、相同prompt reset、真实20-ID changed prompt。每版96个生成IDs、93步 ×132项 logits/128缓存/三个反馈标量严格一致；每个A输出也与保存的实际V38 A哈希、shape、dtype、字节数核对。原seed、捕获输出和请求seed均只读。B只在每请求开始上传131个范围，之后token/state由GPU反馈；完整参考比较不会回填B。

性能脚本各启动一个新进程，3次warm后测3次，每请求重新prefill/reset/上传seed。decode窗口包含31次native执行、GPU等待和每步4字节prediction观察；132个最终输出的hash、资源探针和写文件都在窗口外。三版各6份最终132输出均匹配对应资格A。

| B-only独立运行 | 每步中位 | 31 / decode窗口 | 32 / prefill至完成窗口 |
| --- | ---: | ---: | ---: |
| V38，第27章参考 | 36.188 ms | 27.633 tok/s | 24.504 tok/s |
| V39，真卷积融合 | 35.608 ms | 28.083 tok/s | 24.932 tok/s |
| V40，norm复用 + held32 | 35.354 ms | 28.285 tok/s | 25.086 tok/s |
| V44，组合 | 34.699 ms | 28.820 tok/s | 25.477 tok/s |

这些差值没有平衡配对控制，不作为已确立的A/B收益。V44实际ICB每bank1625条dispatch、30个重建PSO、1046个barrier；V38参考为2009条dispatch。减少命令数量也不能直接等同于速度收益。

V44的93个测量步，graph编码中位0.065500 ms，全部主机编码0.237709 ms，native host total 34.681792 ms，wait 34.441750 ms，CB GPU envelope 34.121000 ms。GPU envelope与wait重叠；graph是全部编码的子区间，各中位数不能相加。这里只复算保存的时间戳，不宣称单kernel独占耗时。

所有成功与失败轮次均保留root owner、原始资源与外部释放证据。V44完整资格53份、screen19份保存资源为AC/thermal0/pressure1，minfree46%/47%，swap MB与Swapouts无增长；Pageouts分别增长288/264。进程已wait/reap并严格证明PID不存在；数值通过不代表整机没有paging。

[复现说明](28-native-gdn-composition-source/README.md)、[源码身份索引](28-native-gdn-composition-source/source-map.json)、[依赖解析](28-native-gdn-composition-source/dependency-map.json)、[逐版本provenance](28-native-gdn-composition-source/provenance.json)、[原始文本索引](28-native-gdn-composition-source/evidence-index.json)和[证据包](28-native-gdn-composition-source/evidence.tar.gz)保存本卷。二进制、权重及raw tensor未收入；旧章节的同字节来源通过hash引用，不复制旧大证据包。
