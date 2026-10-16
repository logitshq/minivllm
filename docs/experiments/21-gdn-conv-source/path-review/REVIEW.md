# V23 independent path/action review — static case supports numeric qualification only
2026-10-16；CPU stdlib读取/推导；无MLX/GPU/模型/Git/engine变更；root唯一GPU owner。
accepted15 payload gdn.py:162–174与W gdn.py:197–209目标链均rawQMM→concat→BF16 product→sum(axis1)→compiled nn.silu→tail view；W normreuse默认OFF。
不要把prefill的Python sum helper或V18 postconv当本decode；W没有gdn_postconv_metal.py，真conv仍未融合。
dense前提下xp[1,4,10240] strides=[40960,10240,1]；w[10240,4] strides=[4,1]，w.T[None]有效广播strides=[0,1,4]。
本地0.32.3 common/binary.h:19–33选General；:84–92只donate row-contig同size输入或默认dense out，故product静态预期C-row同xp。
此为固定dense分支的源码推导，actual flags/strides仍须记录；不能拿任意相同shape或ensure_row_contiguous代替原product布局证据。
B1 tail=xp[:,1:]逻辑[1,3,C]、offset C，singleton批stride不影响row-contig；逻辑60KiB可保留xp80KiB backing，不应误判B4同样。
V23 tap映射old0/old1/old2/raw，w[c,0..3]；newtail为old1/old2/raw独立输出；每channel唯一owner不写原输入。
BF16 multiply返回及每次BF16 add边界必须保留，禁止跨product FMA/全FP32 taps；local +0/product-first和ascending partial merge须与实测Y相同。
原生PSO已由root实测M1024/width32→Y4（无dispatch）；只有runner实际product flags/strides匹配该kernel selector前提，才能冻结Y4，禁止按byte结果挑树。
原Sigmoid内部auto/float表达式不等于外部可合并：V18 typed BF16 sigmoid→typed BF16 multiply的662B原文已保留；safe math不代替GPU rounding/denormal证据。
先原conv、act、newtail全finite/dtype/shape/raw BF16 bytes及旧输入immutability；任一失败停该build，再谈原递推o/S和链式state/fullmodel。
机制有实际差异：原3个compute primitive（Multiply/Sum/compiled SiLU）→1个V23，另省concat数据移动；tail原是view，不能虚报省一个tail-copy launch。
48个GDN/normal decode；若各compute primitive各一kernel则少96个compute边界/forward；concat的真实copy/kernel数仍UNKNOWN，不能把Python调用当实测dispatch。
参考profile31 decode model forwards/32 outputs，对应最多1488处conv候选；原QMM、gates、norm、recurrence/head-repeat均未减少。
每层逻辑payload xp/product各80KiB、conv/act各20KiB，V23 act20KiB+tail60KiB；B1 tail backing理论少20KiB×48=0.9375MiB。
按buffer读写模型，baseline约552960B/layer，V23约245760–307200B（是否重读tail）；48层少约11.25–14.0625MiB/token，均非实测DRAM。
因此潜在价值更偏小kernel/依赖/graph边界，而非多GiB allocator池；没有实际pool bytes证据，不据此sweep池大小。
理论上可能贡献~6%x缺口：若当前有效decode约35ms/token，要x+6%需省1.981ms/token≈41.27µs/GDN；这是所需预算，不是收益预测。
若只靠上列11–14MiB流量，示例100GB/s也仅约0.12–0.15ms；要接近缺口需实测显著launch/依赖或host graph收益，不能靠payload大小断言。
现有profile steady async33.6445ms/model-graph2.4552ms不拆kernel占比；V18<1%E2E不排除真conv，但局部micro也不能证明C1整体胜出。
下一行动：review v23-conv-numeric-build1，先将actual product布局匹配已测PSO的Y4条件，再单支strict numeric；过后才站稳微计时，root决定完整资格/E2E投入。
完整路径/SHA、静态stride、逐层logical traffic与预算公式见evidence.json；当前没有新的GPU GO。
