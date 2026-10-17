# V127：真实同输入共享 W-M1 局部资格

状态：CPU_READY，GPU UNRUN。独立 V124 保存结果复核通过（40 synthetic cases/80 matrices），但本次真实九几何尚未执行。无模型源改动、无引擎集成、无性能或采用结论。

见 [DESIGN.md](DESIGN.md) 和 [protocol.json](protocol.json)。使用 candidate116 原字节 target-only、true CAP64、一次 ordinary19→32 startup warm 后重建固定原 V113 block；不调用/加载 drafter，保留其 bounded loader 原文件。proposal fixture 来自原 V108 **B** 实际提案，SHA 和来源在 prerequisites/protocol 内；本次不声称新 drafting。

原 T5 target 图完成并移除 capture hooks 后，逐497真实 Linear（含 lm_head）验证全部旧输入和 batch 输出哈希，再逐 native-row-M1 对照共享 W kernel。T2/T4仅是这些相同 T5 X 的独立 prefix，明确不是新的真实 verify 轨迹。总1491完整矩阵严格 shape/dtype/finite/每字节比较，所有9×3 shared route 计数必须命中；old113 fallback 路由被记录且不能冒充 shared 覆盖。实际 GPU descriptor shape/stride 在调用前用只读小 probe 观测，不冒称内部 flags。

原1349参数对象身份/字节、497输入/捕获输出/W/scales与prefix只读，keep0全部128缓存严格恢复。无 weight copy/repack/rebind，无候选结果向原轨迹或未来 proposal 回流。所有一次性替代输出在下一 case 前释放。运行保留34GiB allocator limit/64MiB cache、50/50/20/0，逐阶段 raw resources + pressure/paging/AC/thermal0或1 guard。root外层还须负责 nominal0 preflight、owned child timeout/reap/strict absence；本包不越权创建 owner/服务器。

CPU inert（不导入 MLX/NumPy）：

```sh
python3 -B /private/tmp/minivllm-engine-optimization-20261014/v127-real-input-shared-m1-qual-build1/check_cpu.py
python3 -B /private/tmp/minivllm-engine-optimization-20261014/v127-real-input-shared-m1-qual-build1/qualify.py --inert
```

root owned child 的 GPU argv（新绝对输出目录；不由作者运行）：

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v127-real-input-shared-m1-qual-build1/qualify.py --execute-gpu --out /private/tmp/minivllm-engine-optimization-20261014/v23-root-control-build1/numeric-v127-attempt1/child-output
```

child 环境须只有 MLX_MAX_OPS_PER_BUFFER=50、MLX_MAX_MB_PER_BUFFER=50、MLX_BFS_MAX_WIDTH=20、MLX_METAL_FAST_SYNCH=0 这四个选中控制变量，无 DYLD/PYTHONPATH 等。owner clean-parent 负责注入这些值（不要把它们写入要求 clean-parent 的 owner 自身启动环境）。建议900秒硬总期限，resource raw 定期外层监控与强制清理沿现有 root owner，不调整 guard。

本包 source-pins 固定来源，包括 V124 shader/helper/license、candidate11630个 Python 文件、V113 fixture/gold、V123 warm 函数和实际资格小件。源码中的 MIT 许可保留在冻结 V124源包及 candidate 原件，不复制或改写 shader。独立旧113、117、123结果只给来源/一致性约束，不能转移为本次新 kernel 的 propagated/fullmodel 资格。
