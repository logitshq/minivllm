# 32. 残差carry与四投影：通过数值门槛之后

第31章留下两条方向：减少图中的残差读写与调度，或者把GDN的四个投影放进一个原始指针kernel。本章分别记录残差carry的V65/V70/V71、四投影的V68/V73，以及R8复用的V69微基准。

两条整模路径通过了限定工作负载的数值资格；B-only screen没有证明明显收益，不能采用或宣称Pareto胜出。V69虽然6/6对获胜，中位耗时降低只有1.046412%，低于预先声明的5%门槛，仍为STOP。源码、历史准备状态与实际结果见[归档索引](32-native-residual-projection-source/README.md)及[provenance](32-native-residual-projection-source/provenance.json)。

## V65：保留舍入边界，把残差传到下一次norm

旧路径反复执行BF16残差相加，再读取相加后的tensor做RMSNorm。V65不删除残差加法，而是在融合helper里先得到同样舍入的残差，再计算norm，并把该残差留给后续层。

```text
原来：BF16 add -> 存储rounded residual -> RMSNorm读取它
carry：helper生成同样的rounded residual + norm输出 -> 后续继续使用两者
```

不能把 `BF16(left + delta)` 改成任意FP32加法后再归一化：平方和、FP32规约树、precise rsqrt、centered权重的舍入和fp-contract策略都影响结果。此实现保留原helper的这些顺序。源代码入口与差异见 [carry入口差异](32-native-residual-projection-source/prototypes/v65-residual-carry-source-build1/qwen.diff)。

首个input norm仍保留。64层各做一次mixer-add/postnorm，前63层各做一次MLP-add/next-inputnorm，末层做MLP-add/modelnorm，共128次融合调用。它不是把64层折叠成一个计算，也没有改变token间依赖。

入口默认OFF、参数要求严格bool；只接受限定的正常B1 decode、原runtime类型及129个5120维norm元数据。未知元数据或embedding不符在cache mutation前回退；图已经开始后的错误传播，不在半条图中回退。该guard不宣称完整验证任意被外部修改的模型内部结构。

### 先观察实际模型，再执行资格

V65 worker在任何token之前记录实际加载的129个norm的shape/dtype/centered/epsilon与参数身份，不用历史元数据代替当前观察。实际捕获观察到128次残差helper调用、48个不同GDN的卷积/activated norm reuse路径；计数hook在捕获后移除，不是每个native replay都调用128次Python hook。

参考A把carry关掉，并临时绑定原accepted25 GDN入口。A完整forward、构建132项输出、eval与synchronize都在该上下文内，之后恢复；每个A输出又与旧V38和V44保存哈希核对。因此这不是V65对自身的自比较。

实际加载一次模型，原prompt、相同prompt reset和changed prompt各生成32个IDs；93步×132=12,276项严格输出，包含logits、128缓存和三个反馈标量。原始seed/捕获输出/请求seed只读，1349个参数对象身份保持一致。有限请求通过不扩展为所有prompt或任意批量的证明。

## V70：carry接进direct async，覆盖仍分两种

V70把V65候选接到不变的V62 direct pipeline与V64 adapter，没有包含四投影。FIFO、最多两票、消费后refill、EOS未来票discard及quiescent读取合同沿用第31章。源worker见 [V70资格worker](32-native-residual-projection-source/prototypes/v70-residual-direct-async-fullqual-build1/fullqual.py)。

| 模式 | 严格输出哈希检查 | 覆盖范围 |
|---|---:|---|
| diagnostic | 12,540 | 核心93步×132，加回退后和早停后新请求的两次最终132 |
| production | 660 | 五个quiescent请求末尾各132，另核对161个delivered IDs |

两种独立进程各加载一次模型，均为161 delivered、162 submitted/completed、1 discarded、6 resets，bank84/78。production不能借用diagnostic数字宣称逐步全状态检查；早停六个交付步也不做完整132项逐步声明。参考始终是旧accepted25/V44 A，未改成carry自参考。

## V68：四组原始weight/scale指针进入真实模型

第31章的V66只有局部396项数值资格。V68将同一helper接进48个GDN：一个输入x，加qkv/z/b/a四组原始weight/scale，产生四个BF16输出，再走原卷积、gates和output norm后缀。

```text
qkv(x), z(x), b(x), a(x)
          ↓
four_original_pointers(x, Wqkv/Sqkv, Wz/Sz, Wb/Sb, Wa/Sa)
          ↓
原conv/gates/状态更新后缀
```

没有weight concat、repack或新packed权重。源码ABI是9个输入指针加4个输出指针；实际native40报告没有逐argument表，因此 **13个pointer是源码ABI证据，不能写成逐项反射已观察到13项**。实际捕获仍有PSO、完整原始parameter buffer/range与只读角色证据。

准备阶段额外执行一次普通decode，独立观察48个原input norm输出和384组weight/scale描述符，共432项；它不计入93步资格或性能计时。后续输入受限于不变的原norm布局合同，不在每个token重复layout probe，也不把任意外部x纳入资格。源码差异见 [四投影入口与guard](32-native-residual-projection-source/prototypes/v68-native-fullqual-build1/candidate.diff)。

实际准备保留原8个operand对象/层；384个目标范围共2,278,195,200字节在prepare前后与结束时stream hash核对，1349个参数对象身份不变。绑定对象变化时在cache mutation前丢弃旧helper并回到当前原操作，避免复用陈旧指针。

实际捕获48个不同GDN、48个不同helper、每次四个BF16输出；三请求93×132=12,276项与旧A一致。参考A显式关掉四投影，完整forward在原accepted25上下文中。V68不包含残差carry，不能把V65和V68的资格自动合并成新候选资格。

## 调度条数减少，速度需要另测

| 捕获路径 | 每bank dispatch | 重建PSO | barrier |
|---|---:|---:|---:|
| 历史V44 | 1625 | 30 | 1046 |
| V65 residual carry | 1497 | 30 | 918 |
| V68 四投影 | 1481 | 31 | 1046 |

这些是实际图调度计数，不是硬件指令数或加速比例。四投影也可能改变原qkv与conv的重叠时机；当前screen没有给出该效应的因果测量。不能用少144条dispatch直接推断相应百分比收益。

## V71与V73：先画清计时边界

两轮都为三个warm加三个measure、greedy32个输出，没有真实HTTP或平衡配对A/B。V71是carry + production direct async的引擎路径；V73是四投影的同步raw-native路径。

```text
V71 engine：t0 -> prefill首token -> t1 -> reset/131seed上传 -> decode31 -> t2
V73 raw：   t0 -> prefill + seed准备完成 -> t1 -> decode31 -> t2
```

| 各自三次measure中位 | V71，decode含seed设置 | V73，t1在seed之后 |
|---|---:|---:|
| `(t2-t1)/31` | 34.690042 ms/步 | 34.846147 ms/步 |
| `31/(t2-t1)` | 28.826717 token/s | 28.697578 token/s |
| `32/(t2-t0)` | 25.929581 token/s | 25.360245 token/s |

V71累计186 delivered/completed、0 discard、6 resets，最终792项哈希核对；V73每个请求末尾132项与旧A一致，六次共792项。哈希复制、资源采样在各自计时区间外；原始capture owner只读是保存的运行时证据，不是退出后重新构造的字节。

**不能直接相减两列为carry或四投影收益。** 两者执行路径和t1边界不同，也没有配对控制；whole32两个值同样不能替代公平对照。结果没有证明明显改善，维持未采用、无HTTP或Pareto胜出结论。

## V69：6/6胜，也没有越过预声明门槛

V69在原SG2/64线程下，把每SIMD独立行数R从4增至8，一组已加载/转换的16个activation值供8行复用。每行K512循环、FP4/scale解码、qdot、FP32累加、simd_sum与最终BF16 cast保持原顺序；没有factor fold或weight repack。更多accumulator也不自动意味着更高occupancy或更快执行。来源见 [R8 geometry](32-native-residual-projection-source/prototypes/v69-nvfp4-r8-reuse-build1/geometry.py)。

实际只资格L0 gate的N17408/K5120与9个输入，额外weight为0；36项严格有限数BF16检查通过。source-eligible形状不等于这些形状都已GPU资格，不代表完整模型或GDN路径收益。

性能比较两侧都固定shape `mx.compile`：A为native QMM，B为R8 helper。固定prime2048、timed8192，8个固定synthetic输入轮转；另一个历史真实输入只参与数值资格；10 pairs含4个同臂控制与6个AB/BA比较，共20个prime与20个timed区间。没有自适应挑选或重试。

| 项目 | 实际结果 |
|---|---|
| A/B臂中位 | 239.515526 /237.009206 µs |
| `1−B中位/A中位` | +1.046412% |
| 比较对获胜 | 6/6 |
| 控制 | 每次[0.95,1.05]、每臂两次中位[0.98,1.02]均PASS |
| 性能门槛 | 至少5%降低且至少5/6胜 |
| 最终结果 | STOP-MICRO-GATE，未采用 |

胜数满足只是两个条件之一，不能用6胜覆盖缺失的5%收益。降低率是两臂中位数的比值，不是逐对降低率的中位数；数值PASS和控制PASS也不会改变性能STOP。

## 冻结范围与历史元数据

本章只收V65/V68/V69/V70/V71/V73的冻结源码、审查和八次实际运行。V72/V74仍属其他工作，不收源码或运行目录；后续合并候选也没有在这里追加未发生的资格或性能结论。

V71的 `source-pins.pending.json` 与CPU-PENDING文件按历史原字节保留：它们在实际qualification绑定更新前描述准备状态，不能覆盖最终source-pins和实际receipt。历史UNQUALIFIED/未来协议措辞同样保留，通过[历史清单说明](32-native-residual-projection-source/historical-preparation-manifests.json)与当前provenance区分。

文本采用精确字节与确定性gzip/tar，重复内容引用24–31章。MLX MIT及Lithos Apache全文与第三方声明分别保留；runtime/native二进制、权重和tensor不复制，只保留外部身份。归档不是可重定位执行环境，也不授予新运行或默认启用权限。

独立CPU校验核对全部文本、绑定pins与保存释放证据，重算两种screen原始时钟和R8 micro门槛，并将V65/V68/V70/V71/V73保存的输出哈希对回旧A。结果见[verification](32-native-residual-projection-source/verification.json)、[manifest](32-native-residual-projection-source/manifest.json)与[source-map](32-native-residual-projection-source/source-map.json)。校验不导入MLX/native，不执行模型或GPU。
