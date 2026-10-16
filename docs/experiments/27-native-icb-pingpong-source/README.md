# V36 S-pingpong / V37 独立克隆 / V38 实际 ICB

这里只归档截至 V38 的源码与九轮真实运行。每版都有 tiny、27B三请求 fullqual、3warm+3measure B-only screen；全部实际通过数值/最终哈希检查。V37 的无收益结果完整保留。当前最快的36.188ms/步、27.633tok/s只属 V38 B-only，没有公平配对A/B、HTTP/Pareto、服务资格或采用结论。

| 版本 | 父源码 | 原生源码目录 | 特点 |
| --- | --- | --- | --- |
| V36 | 第26章 V34 native1 | `prototypes/v36-native-pingpong-build1/` | 48个S allocation对轮换，83条staged反馈，仍逐步编码 |
| V37 | 第26章 V34 native1 | `prototypes/v37-native-resource-clones-build1/` | 每个原mutable buffer独立克隆；保留scope同步，安全相邻合并；无S轮换/ICB |
| V38 | qualified V36 | `prototypes/v38-native-icb-build1/` | 两套固定ICB、保留原函数/descriptor、常量bank及资源声明 |

`source-map.json` 是逐文件的原绝对路径、归档路径、字节数、SHA及角色；复制文件保持原字节，未重写脚本。`provenance.json` 将每个实际 run 的状态、收据SHA和owner命令绑定到版本。`evidence-index.json` 列出压缩包每个原始文本成员的路径/字节数/SHA；包内含所有九轮的收据、native报告、原始资源、stdout/stderr、owned wait/cleanup与external release。

`dependency-map.json` 逐项解析实际 worker/screen 的source pins：本包源码、本包tar成员、此前精确归档或外部MLX/原生二进制。继承的Shared/Resources/小探针等只按其实际版本使用，不能拿后来候选代替。`reference/accepted25/` 保存实际父模型和kernel的25个源码；native S_new stores-only资格依赖这里的冻结kernel，不依赖当前仓库工作树。

`reviews/` 保存原独立CPU报告及可复算脚本。V38有47-pin整模、51-pin screen审计；V36/V37 screen有49/46 pins。复核的是保存的A哈希/Cmemcmp记录、最终B哈希、计数和时间戳，没有独立GPU重算全部原始输出。源PSO重建只改变ICB支持flag的事实与真实数值通过分别记录。

## 如何定位和复现

1. 从`provenance.json`选 tiny/full/screen，读取对应root owner及worker/screen源码；版本不能交叉替换。
2. 原环境是M4 Max、MLX与mlx-metal0.32.3、原Python/NumPy及同一27B checkpoint。运行记录/每份source-pins保存实际路径与SHA，模型、张量raw、`.dylib`、`.so`及`.metallib`都不入包。
3. 原生编译命令形如：`clang++ -std=c++17 -fno-objc-arc -Wno-deprecated-declarations -dynamiclib -framework Foundation -framework Metal replay.mm -o libv38replay.dylib`；36/37使用自己的源码和库名。CPU-READY及收据保留实际已运行库的SHA，重编译不冒称生成同一二进制。
4. 四项控制为`MLX_MAX_OPS_PER_BUFFER=50`、`MLX_MAX_MB_PER_BUFFER=50`、`MLX_BFS_MAX_WIDTH=20`、`MLX_METAL_FAST_SYNCH=0`，allocator cache2GiB；每个子进程只注入自己版本的库。root owner串行执行，保留超时、精确PID身份和释放条件。
5. 先tiny，再完整base/reset/changed三请求strict资格；对应B-only screen只能绑定该版实际PASS资格收据。各screen的`--inert`仅核源码/固定协议，不导入MLX、不构成GPU资格。

旧绝对路径、既有运行输出目录及版本pins故意保留。这个档案不是可直接解包运行的安装包；另机复现需在新实验目录适配依赖和路径、重新记录源码/库身份与新收据，不能覆盖或沿用旧结果。`prepare.py`与diff属于原CPU准备历史，实际运行源码以`replay.mm`和收据所绑定SHA为准。

## 已验证边界

三版全模型各93步/12,276项严格比较，B每请求一次131范围上传；48S轮换只用于V36/38。每版screen在一个新进程中6次fresh请求，3warm+3measure；最终132项哈希在计时区外检查。V36/V38额外保留逐步native计时原始导出。

V37实际resource-specific barrier=0，不支持原先关于这类barrier被arena扩大的原因猜想。V38主机编码时段下降，CB GPU envelope仍约35.6ms；GPU envelope与host wait重叠，不能相加或当独占算子占比。原始Pageouts/压缩变化和所有释放证据一并保留。
