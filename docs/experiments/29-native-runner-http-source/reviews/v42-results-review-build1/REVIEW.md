V42 结果：SKIP_NO_REDUCTION；不运行实际性能 screen，不替换 native40。

实际 fullqual 收据 SHA `0f4df333…07bd8032`：3 请求、93 步、12276 输出严格相等，另 12276 个 A hash 与已保存实际 V38 基线一致。[收据](../v23-root-control-build1/fullmodel-v42native-attempt1/child-output/receipt.json)

独立 CPU 审计重算全部 1769 dispatch 的 R/W 集合及 1318 boundary origin；原屏障位置1142 → planned1143，removed0、deferred0，唯一新增为 ordinal0 首屏障。[完整计数和 witness](analysis.json)

- 原始 origin：1059 memoryBarrierWithScope（全部 exact Buffers）、84 encoder_end、85 CB_commit、84 fence_update、5 fence_wait、1 event_signal；其余 origin 为0。259 forced 条目折叠成83 dispatch位置。
- 1059 个 scope 都在原 next dispatch 遇到同 original-allocation RAW；770仅RAW，289另有WAR+WAW。没有 scope 被 forced 边界强行保留。因此当前 frontier 规则下没有可延后 scope。
- 这不是 shared arena ID 造成的假冲突：规划与独立审计用1859个互不重叠 original MTLBuffer ID，271个 mutable allocation参与 scope 冲突，zero-copy参数无冲突。见 [allocation验证](../v42-native-dependency-icb-build1/replay.mm:261)、[active reflection构集合](../v42-native-dependency-icb-build1/replay.mm:273)。
- 现有证据只证明 allocation 级依赖；并未记录每个 shader 参数实际读/写字节 extent，无法断言这些冲突都是字节级重叠，也不能把它们叫作可删除假依赖。未实现 byte-range 方案。
- 逐对独立证明覆盖27363个冲突 dispatch对（RAW19353/WAR17088/WAW8779 allocation edges），两 bank 同构；不是只相信 native 的 proof 布尔值。[审计程序](analyze.py)

Lithos实际 `barriers.py:38–74` 同样使用 whole-base R/W。其61–64行还有“把屏障前移到紧邻独立 op”的规则，V42未实现；本次177个 scope 的 next dispatch 与紧邻 prev 无 allocation冲突，仅是该规则的静态线索，不是已资格方案或收益证据。其余882有相邻冲突。[固定donor源码](/private/tmp/minivllm-remaining-20261014/lithos-source/monolith/compiler/barriers.py:38)

MLX0.32.3实际安装头文件也保留prev/next输入输出resource集合和maybeInsertBarrier接口；这与本次观察一致，但头文件不能证明所有调度都最优。[安装头文件](/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/device.h:133)

screen已绑定实际PASS收据并标记SKIP_NO_REDUCTION；6请求循环AST仍等于V40，仅为将来明确要求复现保留。当前没有性能点，也没有采用结论。[screen](../v42-native-decode-screen-build1/screen.py)
