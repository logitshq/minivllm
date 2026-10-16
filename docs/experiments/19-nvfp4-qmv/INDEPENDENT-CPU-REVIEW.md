# Independent V22 prelaunch CPU review — PASS for frozen small-target experiment
2026-10-16；3CPU calls，source/header/geometry audit及CPU tests/--help；无MLX/GPU/model/探针执行；未修改source/raw。
Driver SHA256: dd071185991c93b41a9c01997912f4baae1bb20c7e9a4a72d6c3ffa8d5d5c279
Header SHA256: 7373cccf7ee7cf6bec4242502b725529c7906835fe6216bbf53631dfd60b508a
Body SHA256: 6d60381de86ab832d29f873ccfcc65b6f3952cbc2254e6e6f3c10f2127b59cb3
Protocol SHA256: 4631124cb8632e9bbaf470b659c148bc3f38e0acf400121073494e612ff2806d；prototype manifest SHA256: b007644922d50634bee67575917a70e651bbd0e2409c2ba07a7c6fa07b277e37
6个CPU tests及--help均rc0；identity()无MLX import，冻结7文件与5 upstream SHA复验。
官方本地0.32.3 fp4/fp8完整#pragma-once起正文逐字包含；fp_quantized helpers/fast_impl逐字及SHA匹配。manifest extraction是1-based start/end-exclusive。
MIT LICENSE全文、Apple copyright及上游URL保留；header未添加FP pragma，driver未设置math mode override。
fp4 nibble→half→float、E4M3 scale decode、float qdot组合顺序、每K块scale乘及simd_sum→BF16 cast均为官方原文。
only results_per_simdgroup=R取4或2；has_global_scale=false，未增加dequant/requant、global scale、clamp或改变accumulator dtype。
TG(64,1,1)恰2×32SIMD；grid(64,2176/4352,1)→x仅1group，body强制tile.x=0，local/32及%32正确。
枚举验证两R输出row恰覆盖0..17407且不重叠；K5120=10×512、lane×16读取0..5119，scales0..319，weight每row最高byte2559。
K/N以constant int全局传给const constant int&，weight/scales/x/output均device pointer；nullptr global scale由false分支消除。
只L0 gate U32[17408,640]+U8[17408,320]共50135040B；memmap定range并copy detach后close，无full shard/model load。
8个唯一synthetic+L0历史real共9输入；历史receipt/hash固定，非本轮capture，旧整体STOP保持。
先全部9项R4-QMM严格finite/shape/dtype/BF16 byte门，再请求R2并对同9项做R2-QMM及R2-R4；预期27checks。
任一资格失败assert→FAIL-STOP，qualification.json只在27项全过后写，timing之前有resource门；无自动放宽。
A确为原mx.quantized_matmul(x,weight,scales,transpose=True,NVFP4group16/bits4)；B仅R2，不以R4或手写math冒充原primitive。
相同输入轮换/每call eval全N输出；prime128/repeat512、6ABBA+4AA/BB、pair后hash、tight control/≥5%且5of6门保持。
Battery/thermal0/pressure1/SO与swapMB不增长；direct Popen child须owner_pid==ppid，170s wait及strict absence cleanup保持。
CPU PASS不证明Metal可编译/真实builtin selector或JIT math一致；R4实际bytes门用于识别这些差异，未过不进入R2/timing。
仅允root sole GPU owner执行精确冻结实验；无adopt/fullmodel/E2E/AC frontier资格，不继承V20/V21收益。
