# GLM-5.3-Flash 普通加载：现资源门的静态准入复核

2026-10-17。只读现有小回执、配置和源码；未运行测试/help、模型/MLX、服务/进程查询、下载或大文件 hash。以下路径相对于 `/Users/ryanli/code/minivllm`，包路径 `P` 指 `.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages`。

**结论：静态不足，当前不能把既有 cold-cycle PASS 当作 GLM 加载准入。** 185.708 GiB 的 retained checkpoint payload 本身没有证明在 256 GiB 上必然失败；普通非 lazy 加载的峰值也没有得到足以授予安全余量的上界。另有确定的启动接口不兼容，直接把现 probe 填入 stable owner 不会执行加载。此结论不是改变资源门或要求新的用户授权。

## 实际门与作用范围

| 位置 | 真正检查什么 | 不能据此声称什么 |
| --- | --- | --- |
| `resources_v2.py:41–51` → `resources.py` → `resource_base.py:41–52` | 原始 pressure/free_percent/swap 必须已知；pressure≥2 拒绝；已识别外部推理进程拒绝。连续两段 swap 增长各超过64 MiB，或连续两段 Swapouts 增加，才触发原 sustained guard。 | **没有最低 free GiB/free_percent 门，没有 child physical 上限，也不是任意新增 swap 即拒绝。** `free_percent` 只要求能解析，不能把其读数直接当物理剩余 GiB。 |
| `cold_cycle_v2/owner.py:110–124` | 每次检查原服务 PID/job/8001 缺席，调用上述 guard；无 suspended 豁免，pressure 必须1、AC、thermal0/1。poll独立控制 child≤1800s/总工作1830s，finally TERM/KILL/wait/strict absence。 | 资源采样和硬回收不等于 Metal 分配硬限额，也不限制两次采样之间的瞬时峰值。 |
| `cold_cycle_v2/controller.py:539–557` | 固定 RAM=256 GiB；冷稳定系统 footprint≤40 GiB；两个样本间隔5s（有效4–60s），footprint变化≤1 GiB、pressure1、第二次swap不增长。 | **40 GiB 是卸载后/恢复前的系统 footprint 上限，不是 GLM 加载期间上限。** |
| 同 controller `resource_gate(loaded=True)`、`monitored_load:593–607` | 恢复原 oMLX 驻留模型时系统 footprint≤240 GiB（保留16 GiB）、pressure1、相对恢复基线 swap不增长。 | **240 GiB 门保护原服务恢复，不会自动在实验 child 运行期间执行。** |
| `glm_flash/load_probe.py:152–164, 420–435` | 核对外部 admission 的 owner/plan/source/path/时间和证据 SHA；独立 EnginePool，无 server enforcer callbacks。记录 MLX active/cache/peak/limit。 | admission 布尔值是外部绑定断言；probe不自行证明物理余量。标量 boundary 也不是连续物理峰值。 |

独立 EnginePool 的默认行为是确定的：`P/omlx/engine_pool.py:344–375, 878–941` 明示 callbacks 尚未接线时无条件准入，返回 ceiling=0。不能借它的方法文档“pre-load memory checking”推断此 probe 会被官方服务内存门保护。`P/mlx/core/__init__.pyi:867–904` 同样明确 MLX memory limit 是 guideline、可使用 swap；默认1.5×recommended working set，cache limit 默认等于该值。此轮未读取设备 recommended 值，不能猜一个上限。

## 已知字节与必要余量

来源：已关闭 `.artifacts/m5-ultra-256gb/glm-flash-download-v1/header-inventory.json`，明确 `peak_memory_measured=false`、`approval_to_load=false`；该回执记录旧 download SHA，当前 payload 未重哈希。

| 项目 | 原始逻辑字节 |
| --- | ---: |
| 62 shard 文件合计 | 203,992,076,296 B =189.982426 GiB |
| 全 tensor payload | 203,976,457,080 B |
| sanitize 去掉的第45层 MTP payload | 4,573,936,256 B =4.259810 GiB |
| 仅扣 MTP 的 payload `W0` | 199,402,520,824 B =185.708069 GiB |
| backbone routed experts | 183,911,841,792 B =171.28125 GiB |
| 保留视觉权重 | 1,127,254,016 B =1.049837 GiB |
| 一层新 gate/up fused payload `F` | 2,717,908,992 B =2.53125 GiB |
| 最大单 projection stack 输出 | 1,660,944,384 B =1.546875 GiB |

`W0` 不是实测 model active/physical footprint：sanitize还会stack/转置/转换部分dtype，allocator、映射、driver驻留也不能直接用文件字节替代。下面仅把它作为明确的逻辑预算基准，不给 physical peak 上界。

- 完全不留系统量的文件减法：`256−W0=70.291931 GiB`，这不是可用工作区。
- 若未来明确在 **GLM child 阶段**沿用16 GiB系统余量，实际条件应是 `S + M + L + K ≤ 240 GiB`：`S`为当时其余系统占用，`M`为实际保留模型物理量，`L`为加载/融合/allocator临时量，`K`为请求状态与workspace。现代码没有给出 `M,L,K` 的上界。
- 在仅作预算演算的 `S=40 GiB, M=W0` 情形，剩余只有 `240−40−185.708069=14.291931 GiB`；当前层融合同时保留旧投影与新输出时再扣 `F`，余量为 **11.760681 GiB**，仍未扣runtime、dtype变化、cache/driver retention、请求与加载峰值。不能称40 GiB冷门通过就足够。
- C1已知持续状态规模较小：34层KDA FP32 S为136 MiB，BF16conv约4.78125 MiB；11层BF16 DSA latent在2048tokens约22 MiB。不包括pooling/indexer/top-k/prefill/扩容。把cap32缩为cap8不能消除加载期巨大权重临时量。

## 非 lazy 与 fusion 的真实约束

`P/omlx/engine/vlm.py:2045–2181`：GLM普通路径 `vlm_load` 不传 lazy=True；当前probe关闭expert offload。`P/mlx_vlm/utils.py:941–954`注册各shard；`1180–1192` strict load后整参数树一次 `mx.eval`。后续 `materialize_lazy_state` 分块不能回头限制已经发生的整模型eval。

原sanitize构造42层、每层288专家的stack图。源码能证明输入/输出的依赖关系，不能证明所有原expert输入在整个eval结束前都物理驻留，也不能证明逐层及时释放的实际峰值。若全部原expert数据与 `W0` 同时额外驻留，则逻辑量已为 `171.28125+185.708069=356.989319 GiB`，**这个同时驻留方案确定超256 GiB**；但现证据不允许把它说成实际loader必然发生的峰值。

`P/omlx/patches/moe_gate_up_fusion.py:96–137` 已逐层concat→eval→替换gate/dropup→同步清池。不是全42层新旧gate/up同时保留，不能额外加106.3125 GiB就宣判失败；也不能只用2.53125 GiB宣称总加载峰值安全，因为此前stack/eval、pool及driver释放另有生命周期。`glm-flash-load-budget.md` 已正确记录这些限制。

旧常驻模型历史physical约142.82 GiB仅作说明：若仍驻留，和 `W0` 合计约328.53 GiB，未计系统即不可能全驻留；SIGSTOP不解决。现在的冷恢复协议已能真正释放，不能把旧PID/历史余量当此次新准入证据。

## 一个确定的接口阻断

`cold_cycle_v2/owner.py:81–96` 仍调用固定Qwen/E024 `4dff…` prepare，保留其runtime/model/environment检查，再仅生成：

```text
python -I -B qualifier --plan ... --plan-sha256 ... --out ...
```

GLM probe `load_probe.py:590–602` 必须显式 `--run --admission ... --admission-sha256 ...`；没有 `--run` 会打印合同并exit0。stable owner有意只判断执行/资源/reap，不解释模型结果。因此直接填probe会有“进程成功但没加载”的误用风险；其Qwen准备也不等于GLM62shard/runtime身份验证。controller还要求child的 `qualifier_sha256/deadline_s`，与probe目前 `deadlines` 和外部admission不同。

这不是安全门需要放宽，而是未来必须有一个明确的GLM启动/准入适配：保留既有stop/reap/recover机制，绑定真实GLM pins/固定runtime、显式CLI和父子admission；独立检查probe完成4请求及shutdown，不把owner exit0当资格。当前不实施、不改冻结源。

## 可以决定的下一步

1. **现状归类：静态不足，尚不能批准完整load probe执行。** 硬件没有仅由185.7 GiB得出的必败结论；现guard也没有安全装载的证明。
2. 可继续小配置/源码准备；未来必须由外层获得当次冷释放后的实际physical/pressure/swap证据，并明确GLM实验期间的physical预算和停止条件。不能把恢复期240 GiB门或旧单层50 GiB free门冒充已经覆盖此child，也不能把“没有触发 sustained swap”改写成“零swap增长”。
3. 预算中的初次stack/eval和cache/driver峰值仍未知。任何有限加载探针须明确这是待验证风险，受外层硬deadline/reap和已登记资源门约束；采样不能保证瞬时分配不会越界。若普通路径被门拒绝，保留失败，不能静默开offload、改lazy/cache、降精度或放宽门后沿用原baseline名称。

本轮未改变任何配置、源文件或阈值；没有新load成功、physical安全或性能结论。
