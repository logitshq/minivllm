# 29. 把原生 decode 接进调度器，再测真实请求

上一章把卷积和 GDN 的两个优化组合起来。接下来要回答一个不同的问题：循环里跑得动的原生图，能不能接回我们的 inference engine，连续服务多个请求？

本章完成了请求生命周期的数值验证，但真实 HTTP 对比没有完成。C1 实验中途从交流电切到电池，整轮停止；不能据此宣称超过 oMLX 的 Pareto 曲线。

## 1. 我们要替换哪段代码

原来的 `ModelRunner.run()` 用 MLX 调用模型，读取预测 token，并由普通 runner 管理缓存。之前的实验直接在独立脚本里调用原生 replay，还没有经过 scheduler 的排队、完成和清理逻辑。

本章新增默认关闭的适配器，只在显式安装时替换当前 runner 实例的两个方法：

```python
adapter = NativeRunner(engine, library, library_sha256)
engine.runner.run = lambda out: adapter.run(out)
engine.runner.drop = lambda request_id: adapter.drop(request_id)
```

实际代码使用绑定实例方法，见 [native_runner.py](29-native-runner-http-source/prototypes/v41-native-runner-build1/native_runner.py)。原来的 `run` 和 `drop` 都保存在适配器中。

这里限定 scheduler 的容量为 **1**。原因是原生 decode 更新的是自己持有的活跃状态，普通 runner 的 MLX 缓存仍是进入原生路径时的种子。若中途把请求并进其他 batch，普通 runner 会读到旧状态。因此，进入原生 decode 的请求一直使用它，直到完成或被删除。

## 2. 一条请求经过哪些步骤

```text
scheduler 选中请求
       │
       ▼
普通 runner 做 prefill，产出第一个 token
       │
       ▼
检查是否适合原生路径
       │
       ├─ 不适合：整条请求继续普通 runner
       │
       ▼
从真实 prefill 缓存取 131 个输入，reset 原生状态，上传一次
       │
       ▼
原生图：token → logits → argmax → 下一个 token
       │              状态和 token 在 GPU 上反馈
       ▼
CPU 每步只读取 4 字节的预测 ID
       │
       ▼
达到停止条件 → drop → 清理请求和普通 runner 的旧缓存
```

适用条件包括 B1、greedy、无投机解码、缓存容量 256，以及 prompt 长度加最大生成长度不超过这个容量。非零温度等请求整条回退；prefill 就结束的请求也不进入原生 decode。

**改写前**，每步由 MLX 执行模型，普通 runner 持有实时状态。**改写后**，prefill 仍用普通 runner，适合的 decode 请求由原生图持有实时状态；CPU 不为下一步上传参考答案或整份状态。

原生图只构建一次，持有参数、捕获输入和捕获输出的所有者，避免底层内存被释放。新请求从它自己的 prefill 缓存重新设种子，不能沿用上一条请求的结束状态。

## 3. 我们实际验证了什么

同一适配器 V41 分别搭配三个已通过上一章数值验证的计算路径：

| 生命周期实验 | 计算路径 | 原生库 | 结果 |
| --- | --- | --- | --- |
| V41 | 卷积融合 V39 | V38 ICB | PASS |
| V43 | norm reuse / held32 V40 | V40 build2 | PASS |
| V46 | 卷积融合 + activated norm reuse V44 | V40 build2 | PASS |

每次只加载一份模型。先把三条请求一起排队，验证 C1 按顺序接手：原 prompt、同 prompt 重置、改变 prefill 的 prompt。

每条输出 32 个 token，其中第一个来自 prefill，后面 31 个来自原生 decode。三条合计 **93 步 × 每步 132 个输出 = 12,276 次检查**，包括 logits、所有层的缓存、预测 ID 和位置。每个输出都核对形状、dtype、字节数和实际参考输出的 SHA；捕获输入和捕获输出的原始 MLX 数组保持不变。

随后还验证了：

- 非零温度、`top_k=1`、只生成两个 token 的整条请求回退。
- 只生成一个 token 的 prefill 完成路径。
- 回退之后再进入原生 decode，输出 32 个 token，并核对最后一步的全部 132 个输出。
- 每条请求结束后，scheduler、请求表、staging、普通 batch 和 ahead 状态均清理。

因此，每个生命周期实验一共是 **12,408 次输出检查、124 个原生 decode 步、4 次重置**。最后一条请求只核对了最后一步的全部输出，不能写成四条请求都逐步检查了全部 132 个输出。

具体执行身份、实际收据 SHA 和进程退出记录保存在 [provenance.json](29-native-runner-http-source/provenance.json)；[evidence-index.json](29-native-runner-http-source/evidence-index.json) 给出压缩归档内每份证据的身份。

## 4. 另一项尝试：按依赖减少屏障

Lithos 的 barrier planner 提醒我们，不能只数 kernel，还要检查哪些 kernel 被屏障串行化了。V42 因此尝试用捕获到的 buffer 读写关系规划屏障。

但“相邻两个算子不冲突”不代表可以移除它们之间的屏障。更早的算子也可能还在运行，必须覆盖整个开放区间的 RAW、WAR、WAW 依赖。

V42 通过了完整模型数值验证，实际图上却没有减少屏障：原来 1,142 个，规划后 1,143 个；增加的是起点的一个屏障，移除和延后的数量都是零。因此停止它的性能预筛，没有采用，也没有把数值 PASS 写成提速。

V47 是使用该库的适配器草稿，未运行。它保留在归档中，以便区分“写过代码”和“真实验证过”。

## 5. 为什么还必须测 HTTP

原生循环的 token/s 不包含全部请求开销。用户实际等待的还包括 prefill、scheduler、HTTP 和流式响应。

V45 把 V46 的计算路径和原适配器接进实验服务，在相同模型、prompt、greedy 32 token、无缓存复用和无投机的条件下比较我们、oMLX 和 Lithos。oMLX 使用此前测过的最佳配置 `balanced / burst=0.03`；这表示“已测配置中最佳”，并不声称穷尽所有配置。

预先固定的三轮顺序是 NOL、OLN、LNO。每个服务进程独立加载，先做 3 条 warm 请求，再测 3 条请求。客户端记录定义与此前正式对比一致。

第一轮完整 AC 数据仅作描述：

| 服务 | 流式输出速度 x（token/s） | 整条请求吞吐 y（token/s） |
| --- | ---: | ---: |
| 我们的 V46 路径 | 28.347 | 25.548 |
| oMLX | 29.744 | 25.479 |
| Lithos | 29.524 | 21.795 |

这里 x 从首个文本片段到 DONE 计时，并扣除首片段中的 token；y 从 POST 到 DONE 计时。它们是客户端观测值，不能与独立原生循环的 token/s 混画。

**这轮还没有赢：我们的 x 落后 oMLX。**

九个服务进程中只完成四个。第五个进程运行 Lithos 时，在测量组之后资源记录报告电源从 AC 切到 Battery，测试停止。虽然已经保存部分请求数据，该进程仍不算完成，整轮也没有最终聚合结果。

所有已启动服务均已等待退出；父进程、driver、五个服务进程和五个端口另行检查为不存在。主流程因最后一次电源检查再次失败，没有生成完整 suite 的清理文件；归档保留了独立的进程与端口释放证据，不补写一个虚假的完整测试 PASS。

## 6. 本章结束时的代码状态

我们已经把原生 decode 接进 C1 请求生命周期，验证了排队、重置、回退和清理。原生路径仍默认关闭；本章没有提升主引擎的默认实现，也没有新的完整 Pareto 曲线。

下一步是减少 decode 中可测的开销，并重新完成同机对比。提前排入下一步、减少状态反馈的复制，都需要各自的数值验证和真实性能测量。

[源码映射](29-native-runner-http-source/source-map.json) 保留每个实际源文件的路径与 SHA；[依赖映射](29-native-runner-http-source/dependency-map.json) 标明引用前章的相同字节和外部运行环境。归档不含权重、张量输入或编译后的原生库，不能当成已迁移好的运行环境。
