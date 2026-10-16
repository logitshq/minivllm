# 30. 提前提交一步：FIFO、输出所有权与真实 HTTP 的边界

第29章的 runner 每步提交一个原生图、等完成，再把预测 token 交回调度器。本章尝试让主机提前排好下一步：GPU 仍按反馈依赖顺序执行，CPU 不必等当前步结束才编码后继。这是 **V55 depth2 FIFO + V56 C1 runner**，不是跨 token 并行计算。

这条路径通过了限定工作负载的完整数值资格，但真实 HTTP 只完成2/9个计划臂便因电源域变化 STOP。已完成 C1 block 的用户侧交互速度为 native **28.7584**、oMLX **29.7171 token/s**。本章没有采用该路径，没有正式曲线或 Pareto 胜出结论。

冻结源码、原始运行文本和独立 CPU 审查见[归档索引](30-native-ahead-source/README.md)、[结果来源](30-native-ahead-source/provenance.json)。归档中的 `CPU-READY`、`unqualified` 等原型文字按历史原字节保留；当前证据边界以本章与实际运行目录为准。

## 先看同步路径的所有权

第29章沿用 V41 adapter：普通 runner 负责 prefill 和不适合原生路径的整个请求；只有新开始的 greedy、C1、固定容量256请求进入 native。进入后，活跃 token 与128个缓存范围都由 native 持有，直到 drop。

同步每步大致是：

```python
call("v34_replay_step")       # encode -> commit -> waitUntilCompleted
call("v34_copy_output", prediction)  # 只拷4字节预测
return {request_id: prediction}
```

每请求开始上传131个范围：token、position、lens 加128个缓存范围。之后不把参考 token 或参考状态喂给 B。原生图包含真实模型、argmax、next position/lens；其反馈链推进下一步。

同步路径已经保证正确反馈。V55要改变的是 CPU 提交时机，算术图和 token 依赖仍然保留。

## 两张票，两个 stage，S bank 仍只有两份

V55把一个请求最多两步留在队列里。每张 ticket 记录 `slot`、`inputBank`、`outputBank`；`wait_step` 只能交付最早的票。实现见 [FIFO core](30-native-ahead-source/prototypes/v55-native-async-build1/pipeline_core.h) 和 [原生提交/等待实现](30-native-ahead-source/prototypes/v55-native-async-build2/replay.mm)。

| ticket | stage slot | S input | S output |
|---|---:|---|---|
| 1 | 0 | A | B |
| 2 | 1 | B | A |
| 3 | 0 | A | B |

两个 stage 分别保存非 pingpong 输出，避免 ticket2 覆盖 ticket1 的 logits、prediction、position/lens 等交付结果。48个合格 S pair 仍按整份原始 allocation 换 bank，不为每张票再复制一份大 S。

```text
ticket1: graph -> graphDone -> stage0 -> stageDone -> feedback -> feedbackDone
ticket2: wait(ticket1.feedbackDone) -> graph -> stage1 -> feedback
CPU:     wait(ticket1) -> consume(ticket1) -> submit(ticket3) -> wait(ticket2)
```

这里有三个独立的所有权要求：

1. `graphDone / stageDone / feedbackDone` 约束 compute 与两个 blit encoder，下一张票的 graph 先等待上一张的 feedbackDone。
2. CPU读的是 `last delivered` 的 slot 与 S output bank，而不是“最近提交”的 bank。wait1 后，ticket2可以读取 B；在 CPU消费 ticket1 前，不能提交会写 B 的 ticket3。
3. command buffer 在提交前 retain，存入 jobs；wait 观察完成或错误后才 release。slot busy 在 FIFO wait 完成后释放，不能因“主机已经提交”就复用。

`copy/compare` 把最后交付的结果标为 consumed。生产 adapter 只读 prediction；资格程序若要比较全部132个输出，必须在下一次 refill 之前完成。这个消费约定不能简化成“wait过就能随意覆盖”。

静态检查只说明这些约束在源码中存在，不能代替实际 Metal 执行资格。

## V55 build1 的 CPU 竞争与 build2 修正

build1没有运行 GPU。独立 CPU review 找到一个条件性竞争：`v34_write_input` 在拿 mutex 前检查 `canWriteSeed()`。若两个宿主线程交错，seed线程可先通过检查，submit线程再提交第一步；seed线程随后拿锁时 `F.steps` 仍为0，旧检查可能允许向 GPU 正读取的输入 memcpy。

build2将 mutex 移到 async seed guard 之前，使检查与写入相对 submit 原子化；同时明确 drain 的合同：非空 drain 丢弃剩余票，空 drain 保留最后交付结果。来源见[第一轮审查](30-native-ahead-source/reviews/v55-independent-cpu-review-build1/REVIEW.md)和[修正复核](30-native-ahead-source/reviews/v55-independent-cpu-review-build2/REVIEW.md)。

实际 tiny 使用 build2：三个请求、15步、75项严格输出检查，包含变更 seed、depth2、FIFO、reset、只读与 drain。这里 `model_loads=0`，它是小图资格，不是27B完整模型资格。build1的历史 CPU-READY 和未运行状态不会被 build2的 PASS 覆盖。

## V56：先补满两张，再等最早的一张

完整 adapter 见 [V56源码](30-native-ahead-source/prototypes/v56-native-async-runner-build1/native_runner.py)。省略 ABI 绑定与连续性断言后，核心调度顺序是：

```python
# budget = max_tokens - len(already_emitted_output_ids)
while len(pending) < 2 and submitted < budget:
    pending.append(submit_step())
    submitted += 1
wait_step(pending.popleft())  # 必须是 FIFO 最早的票
prediction = copy_current_prediction()  # 消费当前交付结果
return {request_id: prediction}
```

第一次 prefill 已产生一个 token。若 `max_tokens=32`，native预算是31，不是32；所有提前提交共享同一个总预算。正常请求末尾最多提交31步，因此不会仅为填满depth2而额外运行第32步。

adapter检查 request id、position、上一个 token、C1 batch和无 `_ahead` 状态。刚开始不合格的请求整体交回普通 runner；已经进入 native 后不在中途回退到普通缓存，因为普通 runner 的 seed 缓存此时已过时。连续性破坏或 native API 错误直接失败，不伪装成安全回退。

## EOS：已经算出的未来，也必须丢弃

若当前交付的 token 触发 EOS，队列里可能还有一张后继票。它依赖 native 自己的反馈，可能已经执行完，但它没有被交付给调用方。

```python
def drop(request_id):
    drain()              # 等剩余 native 工作，并标记 discard
    pending.clear()
    active = None
    original_drop(request_id)
```

顺序不能反过来：普通 runner 删除请求、下个请求重置输入之前，必须确认 native 的旧工作不再使用这些资源。丢弃后继不增加可见 token 数，也不能把它记为 delivered 性能收益。

实际 early-stop 用第7个输出 ID `103725` 作停止 token：prefill1个，加 native交付6个；第7张 native票已完成但被丢弃。该案例只证明 IDs、FIFO/drain/drop 和只读边界，**没有对这六个交付步逐步比较全部132项缓存/输出**。

## 完整数值资格到底覆盖了什么

V56实际加载一次27B模型。前三个请求是原prompt、相同prompt reset与真实changed prompt，每步检查logits、128缓存及三个反馈标量共132项，且逐项核对旧资格 A 哈希。

| 检查范围 | 数量与边界 |
|---|---|
| 三个核心请求 | 93步 ×132 =12,276项 |
| 整条回退后重新进入native | 最后一步132项 |
| early-stop后重新开始请求 | 最后一步132项 |
| 全部严格输出检查 | 12,540项；不是161步都各检查132项 |
| native累计 | 161 delivered；162 submitted/completed；1 discarded；6 resets |

另有每请求131范围上传、原始对象只读、whole-request fallback、prefill-only、正常尾部清理和早停后重用检查。原始 receipt、owner wait/reap、外部严格不存在证明都在实际运行归档里。有限工作负载通过仍不意味着所有prompt、批量或连续服务状态均已证明。

## 真实 HTTP：两个完整臂，第三个遇到电源域 STOP

V58在真实服务上使用V56路径，比较相同模型、相同prompt、greedy32 token、C1、无缓存复用与无投机的请求。oMLX沿用此前已测配置 `balanced / burst=0.03`。完整计划有9个臂；先后完成 native、oMLX 的第一个C1 block，Lithos第三臂在第二个measure之后的资源采样发现 Battery Power，按原AC守卫停止。

| 完整C1 block的描述值 | x：用户侧估计交互速度 | y：整条请求吞吐 | 文本TTFT中位 |
|---|---:|---:|---:|
| native V56 | 28.758411 tok/s | 25.871638 tok/s | 158.990 ms |
| oMLX | 29.717137 tok/s | 25.442339 tok/s | 214.577 ms |

x由首个文本之后的估计token数除以剩余客户端时间得到；y为32个输出token除以完整客户端时间。两轴不同，不能用较好的TTFT或y替代落后的x。native的x比oMLX低约3.23%。上述是两臂各三个measure的中位数，只作已完成block描述，不能将2/9个臂称为正式吞吐曲线或Pareto胜出。

Lithos保存了三个warm和两个measure文件，但没有完整result；它们全部归档，不补齐、不并入对照中位数。全局没有 `suite-complete.json` 或 suite-final `owner-cleanup.json`；已启动的三个服务各自有wait、严格PID不存在及端口拒绝证明，root owner/driver也已reap并由外部证明释放。

V59是单独Battery域的CPU准备草案，**UNRUN**。不能把它算作V58的重试结果，更不能把停止前后的AC/Battery数据拼成一个结论。

## 两条没有晋级的旁支

### SG4、SG8与更长prime：数值正确，控制漂移仍STOP

V49/V52/V53只测试L0 gate投影，输入/权重范围相同，R4数学与解码/求和/cast保持不变；候选分别为SG4、SG8、SG8长prime。每次36项严格数值与只读检查通过。它们没有执行完整模型推理，也没有性能晋级。

固定门槛从未放宽：每个同臂位置控制比值须在 `[0.95,1.05]`，每个臂两次控制比值的中位须在 `[0.98,1.02]`；性能还要求至少5%中位耗时降低、6个AB/BA对里至少5胜。每轮有10对测量，包含4对控制和6对比较；每对有两个连续计时 slot，共20个 timed envelope 和20个 prime envelope。

| 版本 | prime/timed | A控制比值（中位） | B控制比值（中位） | 描述性降低/胜数 |
|---|---|---|---|---|
| V49 SG4 | 2048/2048 | 1.002701、0.999221（1.000961） | 0.896073、0.996989（0.946531） | 1.087741% /3胜 |
| V52 SG8 | 2048/2048 | 0.974166、0.985598（0.979882） | 1.004876、0.996719（1.000797） | 5.334018% /5胜 |
| V53 SG8长prime | 8192/2048 | 1.071572、0.978467（1.025019） | 0.899941、0.996742（0.948342） | 1.032277% /5胜 |

三轮均为 `CONTROL_DRIFT_STOP`。V52虽然描述值越过5%/5胜，A控制中位仍低于0.98；它不是性能PASS。V53四个2048-call prime chunk留下了时间记录，原守卫没有变；延长prime后仍STOP，不能据此确定漂移的物理原因。

V52 protocol里B标签仍写SG4是元数据笔误；实际helper、launch、资格记录为SG8/256。历史文件原样保留，不能据此把已执行的SG8改写成SG4。独立诊断见[V49/V52 timer review](30-native-ahead-source/reviews/v49-v52-timer-review-build1/REVIEW.md)。

### V51：屏障移动通过完整资格，B-only screen没有显示收益

V51尝试将满足证明条件的屏障移到前一个操作之前；每次试移都检查全部冲突边，不改变强制边界。完整模型93步×132=12,276项严格输出及旧A哈希核对通过。实际1625条dispatch、225次尝试、177次接受、48次拒绝；最终屏障1047，原合并屏障1046。接受移动的次数不等于减少屏障的次数。

B-only screen的三个measure中位为34.921190 ms/步、28.635909 decode token/s、25.345074 whole32 token/s；历史V44为34.698668 ms/步、28.819550 decode token/s。该轮没有显示预期收益，也没有公平配对A/B，不能由跨轮差值判断因果。完整数值PASS只解决正确性门槛，屏障变换没有因此获得采用资格。

## 归档与复核方式

本章保存V49/V52/V53、V51、V55两版、V56、V58及UNRUN的V59，包含10次实际运行的全部JSON/log/md/txt文本。V57及V60以后属于后续章节，不在本章范围。

`source-map.json`保留每个原始绝对路径与SHA；相同内容可引用24–29章原字节副本。`evidence.tar.gz`采用固定mtime、固定tar元数据和空gzip文件名；`evidence-index.json`记录每个成员的原路径、长度和SHA。native/runtime二进制、权重和张量没有复制，只保留外部身份与所需pin；这不是可脱离原运行环境直接执行的模型包。

独立CPU校验逐字节比较源码与tar文本、重算三个micro的原始时间戳/控制比值、核对V51/V56核心132项旧A哈希，并核验两个完整HTTP臂及全部保存释放证据。结果见[verification](30-native-ahead-source/verification.json)和[manifest](30-native-ahead-source/manifest.json)。校验不导入归档代码、MLX或native库，不运行GPU，也不授予新的性能或采用结论。
