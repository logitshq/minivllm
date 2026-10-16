# 27：少搬 state，再把真实 Metal 命令留在 ICB 里

第26章已证明 GPU 能把新 state、argmax token、下一位置和长度接回下一步，但每步还要重新编码约2000次 dispatch，并搬运全部反馈。本篇归档截至 V38 的三条实验：V36 让 GDN 的 S 在两块固定区域间轮换；V37 独立尝试每个原 buffer 对应一个克隆；V38 在 V36 上真正建立 Metal ICB。

三版都通过对应小探针及27B三请求数值资格。当前这些 B-only 探索中最快的是 V38：**36.188 ms/步、27.633 token/s**。尚无新的 HTTP、配对 A/B、服务生命周期或 Pareto 获胜结论，主引擎未替换。

## 先省掉48份 S 的往返搬运

V36 保留原 GDN 算式，把每层 FP32 S 的两个完整 allocation 作为 A/B 两个 bank。第一步读 A、写 B；第二步读 B、写 A。它没有把同一块输入原地覆盖。

资格条件来自真实绑定：每块 S 为3,145,728字节、offset0、独占整个 allocation；输入只有原 GDN kernel 的 `S` 只读参数使用，输出是同次 dispatch 的 `S_new`，且冻结 kernel 源码证明该参数只写。其他绑定或范围别名出现时，仍走原 staging 路径。

实际48层全部 eligible。logits、conv tail、KV 和三个标量仍按原顺序先 staging、再反馈；S 不再参加这两段 copy。按 copy 数据长度计，每步 GPU blit 从341,939,224字节降到39,949,336字节。这个数字不能直接当成 GPU 带宽或速度收益。

[原生实现](27-native-icb-pingpong-source/prototypes/v36-native-pingpong-build1/replay.mm)的 `v36_register_pingpong` 保存实际绑定证明；`resultBytes` 按本步真正写入的 bank 检查结果。每次请求 reset 回 bank0，原始 seed 与参考数组保持只读。

## 保留 V37 这条没有带来收益的分支

V37 从 V34 原生版独立分叉，不包含 V36 的 S 轮换，也不包含 ICB。它把505个 mutable MTLBuffer 分别映射到505个固定克隆，保留原 offset、allocation 内别名和 resourceOptions；克隆数据长度合计366,067,053字节。原 CB/fence/event 的完整 scope barrier 仍保留，只合并没有 dispatch 隔开的相邻 full-buffer barrier。

最初的疑问是：公共 arena 会不会把不同原资源的 resource-specific barrier 扩大成同一个大 buffer 的同步。源码中确实有这种映射可能，但**实际捕获的 resource-specific barrier 为0，映射列表为空**，因此这个猜想没有在本次捕获中发生。V37 最终编码1239个 scope barrier，合并177个相邻 full-scope barrier。

数值资格通过，B-only 中位时间40.240 ms/步；这次独立探索没有看到预期收益，保留负结果，不将它混入后面的 V38。

## V38 才真正使用 Metal ICB

V38 的图仍是原 MLX 算出的 kernel 序列，新增的是一次编码、后续复用的实际 `MTLIndirectCommandBuffer`。两套 ICB 分别固定 S 的偶/奇 bank 绑定，每套2009条 dispatch 命令。每步只声明资源并执行当前 bank：

```objective-c
[e useResource:S().arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];
// 完整实现还声明全部只读权重、常量 bank 与所选 ICB。
[e executeCommandsInBuffer:I().banks[PP().bank]
                 withRange:NSMakeRange(0,I().dispatches)];
```

完整代码见 [V38 replay.mm](27-native-icb-pingpong-source/prototypes/v38-native-icb-build1/replay.mm)。它保留实际解析后的 MTLFunction 和完整 pipeline descriptor/options；实际33个原 PSO 都不支持 ICB，因此只把 descriptor 的 `supportIndirectCommandBuffers` 改为 YES 后重建，再检查实际支持和原 threadgroup 尺寸。数值通过来自下面的真实运行，不能只凭“同一函数”推断重建 PSO 必然逐字节一致。

ICB 没有 `setBytes`，所以全部原 payload 被保存到256字节对齐的固定只读常量 bank。每个原同步标记作用到下一条 dispatch 的 `setBarrier`，每套实际1238个 barrier；尾部依赖仍由 graphDone fence 保证。全部1349个只读参数 buffer 加常量 bank 声明为 Read，mutable arena 为 Read|Write，当前 ICB 为 Read。compute→staging→feedback 的自有 fences、完成等待及错误检查保留。

这与 `mx.compile` 是不同层次的复用，也不是逐步伪装成 ICB 的普通编码循环。

## 数值资格与这次计时各证明什么

每版先通过三请求、15步的小探针，再各自完成27B的原19-ID prompt、重置同 prompt、真实20-ID changed prompt。每次生成32个 ID，共93步、12,276项 logits/128缓存/三个标量的严格比较；原 seed、捕获输出和请求 seed 保持只读。B 每个请求只上传一次131个范围，随后 token/state 由 GPU 反馈。资格测试的 CPU 会用 Cmemcmp 读取完整输出来验证，并取4字节 prediction 检查 token；这些参考值不会回填 B。

V38 整模记录实际93次成功 ICB 执行，两个 bank 为[48,45]；数值脚本保留真实参考 A 的哈希及 Cmemcmp 结果，没有保存全部 B 输出 raw blob。独立 CPU 审计核对保存证据，没有重新运行 GPU。

性能脚本为每版启动一个新进程、加载一次模型，执行3次 warm 和3次测量请求。每次重新 prefill/reset/上传 seed；decode 窗口为之后31步，包含 native 调用、GPU等待及4B预测观察。132个最终输出的哈希、资源探针和文件写入都在窗口之外。

| B-only 版本 | 每步时间中位数 | 31 / decode窗口 | 32 / prefill至完成窗口 |
| --- | ---: | ---: | ---: |
| V35 原生逐步编码（第26章参照） | 40.087 ms | 24.946 tok/s | 22.534 tok/s |
| V36：48层 S 轮换 | 39.283 ms | 25.456 tok/s | 22.833 tok/s |
| V37：独立 buffer 克隆 | 40.240 ms | 24.851 tok/s | 22.486 tok/s |
| V38：S 轮换 + 实际 ICB | 36.188 ms | 27.633 tok/s | 24.504 tok/s |

这些是不同运行的绝对速度记录，没有平衡配对控制，不能把表中差值当作已确立的 A/B 或服务收益。

## 为什么编码下降了，GPU 时间却仍约35毫秒

V36/V38 均在请求结束后导出 native 原始时间戳。93个测量步的中位数为：

| 区间 | V36 | V38 |
| --- | ---: | ---: |
| graph 编码 | 3.195 ms | 0.070 ms |
| 全部主机编码（含 feedback） | 3.418 ms | 0.247 ms |
| command-buffer GPU envelope | 35.318 ms | 35.586 ms |
| 主机等待完成 | 35.918 ms | 35.869 ms |
| native 主机总区间 | 39.219 ms | 36.128 ms |

GPU envelope 与主机 wait 大量重叠，不能相加或视作独占 kernel 时长。graph 是全部编码的子区间；各列中位数也不必相加等于总中位数。逐行的 encode=graph+feedback、host total=encode+commit+wait 及 GPU 起止时间均经复算。

ICB 在这些记录中明显减少了主机编码时段，GPU envelope 仍约35.6 ms；尚不能据此宣布模型已经追上任何官方服务前沿。

所有轮次保留原资源与外部 PID 释放证据。CPU复核的 full/screen 样本为 AC、thermal0、pressure1，swap MB 与 Swapouts 未增加；Pageouts及压缩仍有变化，没有把“数值通过”描述成整机无 paging。各进程已回收，原数据不覆盖。

[源码与复现边界](27-native-icb-pingpong-source/README.md)、[逐版本 provenance](27-native-icb-pingpong-source/provenance.json)、[源码索引](27-native-icb-pingpong-source/source-map.json)、[文本证据索引](27-native-icb-pingpong-source/evidence-index.json)和[证据包](27-native-icb-pingpong-source/evidence.tar.gz)保存这三条分支。V39及后续实验另行记录。
