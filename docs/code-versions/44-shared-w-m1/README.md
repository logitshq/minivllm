# 44：共享原始权重的多行 M1 探针

**当前结论仅是 M4 Max 上的局部数值 PASS。没有共享-W整模/性能/采用资格，M5没有自动准入。** [中文补记](LOCAL-NUMERIC.zh.md)解释它与已正确资格的broadcast-M1基线的关系。

| 材料 | 范围 |
| --- | --- |
| `kernel124/` | 原字节helper、MLX MIT许可/header、qdot与共享packed-word kernel、CPU索引检查；历史UNRUN措辞保持不改 |
| `synthetic124/` | 原字节40case/80matrix synthetic harness；actual结果见saved-review124 |
| `real127/`、`support/` | 原字节497真实T5同输入捕获/1491矩阵资格、router、GPU descriptor probe、fixture、ordinary warm来源 |
| `author-recheck127/` | harness作者的保存结果复核，**不是独立源码作者审查** |
| `evidence/` | 原127 receipt明确白名单投影的gzip、42资源字段和root保存清理的投影；原件SHA保留，不冒充exact原件 |
| `future-plan/` | 仅未来prepared/kernel缓存计划，没有接入；V129不包含/未执行 |

实际1491比较=497原T5+497派生T2+497派生T4，九种N/K×三T，全部shared route/no fallback；含lm_head。497旧输入/batch/native-M1 SHA复现、2485 actual GPU shape/stride descriptors（非内部MLX flags）、497 RO记录及1349参数前后哈希一致。literal histories先同边界eval，keep0全128由执行断言确认；未保存每个post-keep0或完整literal-history只读快照。完整矩阵raw也未归档，离线验证不会重算GPU tensor。

21worker+20owner+1post=42保存样本均AC/thermal0/pressure1，Swapouts17519593、swap5038.06MB不变；不据此声称连续无瞬态或唯一GPU使用。MLX allocator峰值16,952,614,436B，**不是RSS/物理footprint**；cache配置64MiB，观测最大69,883,654B，并非硬用量上限。root tool82729 exit0，owner50896/worker50911 wait/reap并严格absent；这里复用保存的实时证明，作者未liveps/GPU。

## 编译与调用合同

见 [compile-geometry.json](compile-geometry.json)：grid `(64,N//8,1)`、threadgroup `(64,1,1)`、2个SIMD组、每SIMD四输出行；模板BF16/T2,4,5/K/N；原U32 W与U8 scales、NVFP4 g16/b4/no bias，实际canonical GPU strides必需，N不完整tile或未知shape拒绝。保持冻结header/表达式；不传math-policy override。source/header与`LICENSE-MLX.txt`原样保留。

`kernel124/shared_m1.py`是注入mx的standalone helper，不import MLX或修改engine。`prepare(...)`强持有原W/scales，`project(...)`接实际X stride；原127故意每call prepare，只作数值探针。未来缓存方案未实现，不能把这个调用路径拿去宣称生产性能。

## 离线验证

以下只用Python标准库，**不import MLX、不加载模型、不运行GPU/服务器、不写归档**：

```sh
python3 docs/code-versions/44-shared-w-m1/verify_sources.py
python3 docs/code-versions/44-shared-w-m1/verify_sources.py --repo-root .
```

第一个核本包全部manifest payload、54个exact source/audit映射、选字段证据算术/范围；第二个额外核39条已公开依赖（38 exact文件+1 lossless gzip117原receipt）。源码引擎在[第39章版本](../39-broadcast-m1/README.md)，原资源probe/fixture在[第27章归档](../../experiments/27-native-icb-pingpong-source/README.md)，117实际原receipt在[第40章材料](../../experiments/40-source/README.md)。

77个原127 pins全部列入dependency-map：33个bundled、39个repository引用、**5个历史actual/release只留source-machine身份**。这些5原件、HuggingFace checkpoint、MLX/编译器及原P目录布局没有打包。冻结historical CLI含原绝对路径与gate；不能直接从此目录执行并声称portable GPU replay。M5应重建所需输入/metadata，保留M4原件，建立自己严格本地资格；不要伪造原receipt或删除device guard来获得“通过”。原始31个candidate116源文件引用现有公开版本，不复制权重。

源码目录中的CPU-READY/UNRUN和future-plan是冻结时点的历史材料；当前实际结论以本README/选字段证据为准。源码内预声明CLI不等于复制了actual launch/env；公开包不携带raw launch、raw process argv/env或凭据。数值之外的完整C1/C2/C4端到端结果属于root另行资格/测量，本包不作排名。
