# Independent V23 CPU prototype review — PASS for structure only; no GPU authorization
2026-10-16；3工具轮纯CPU stdlib/source/metadata/doubles；未import MLX、未GPU/model/compile/launch/计时；未改source/raw/26-source/defaults。
API SHA256: 06f293fbe0077ed3fdf3e71fd2d165264adef9f9861d8c579f8ed55ba6bee951
Body SHA256: 59278c8a2bc16810088fb1f69b7a690500a215f2d422c0f6982bd5a51cb6648a
Native header SHA256: 963a9408cc963262146461b7dc23bf00ff7eb19bd95c19f661b850cd9ff1e026
Tests SHA256: 8662052d15066858e34882b5b3d1692d9914df08130f9c0265fea8d3df0ffd72
12个CPU结构tests rc0；最终4文件SHA复验一致。模块import只读本地文本，kernel cache为空，无自动backend import。
API仅raw[1,10240]/old_tail[1,3,10240]/w[10240,4] BF16；错shape/dtype/缺metadata先拒绝；显式backend后检array与GPU。
`sum_y`为无默认keyword-only参数，仅type-int 1/2/3/4；bool/浮点/缺省拒绝，无自动推断、调优或按数值挑树。
未来必须先记录原product flags/strides及actual pipeline M/launch Y再冻结一支；本次不满足该选择前提。
GRID10240/TG256整除；每channel唯一owner，act写1处、new_tail三段索引覆盖0..30719且不重叠，inputs无写入。
conv输入tap0/1/2取old_tail对应行，tap3取raw；w用channel*4+tap，输入ensure_row_contiguous保障本原型索引。
new_tail=[old_tail[:,1],old_tail[:,2],raw]，独立新output定义；只写raw内容，不把SiLU输出误存tail。
逐product经V18NativeMultiply<T=BF16>返回BF16；每y以+0起，tap=y,y+Y递增，product-first BF16_add。
partial[0]起按ascending y执行partial[y]+conv，每次显式BF16边界；无FP32accumulator/FMA跨product政策、atomics或simd树。
已对官方reduction/reduce_col.h:25–86及Sum<U>::operator()对照：本体描述指定Y下的y-local folds/ascending merge，而非宣称实际Y已知。
662B activation header与实际V18 build2 HEADER literal原字节一致；typed sigmoid BF16中间值和multiply BF16边界均保留。
MLX0.32.3 Sigmoid/Multiply表达式及math_mode=safe沿用；safe编译请求不等同新kernel已具舍入/denormal保证。
Apple copyright/full MIT保留；source-notes明确65,280 prior words只属PASS_ACTIVATION_ONLY，NOT V23 qualification。
不接engine，不含gates/head-repeat/norm/recurrence或runtime enablement；CPU结构通过不认证Metal解析/alias/输出数值。
未来actual只能先原布局/M-Y证据、冻结一支、conv/SiLU/tail strict bytes，再原递推o/S/chained-step及logits IDs cache lifecycle；本次没有GPU GO。
本PASS只针对列明SHA的未接入原型；任何文件后改均需同步复核，不继承V18/V20/V21/V22资格或收益。
