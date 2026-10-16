V99 集成源码独立审查：PASS，未发现 source blocker。CPU-only；未导入 MLX/NumPy、加载 native library、运行 GPU/资源探针或修改冻结文件。

审查关系：我编写过 V95 helper。本结论是对 V99 接线、native 改动及资格门的独立复核，不是 V95 数学实现的独立重写或第二份独立数值资格。

- 153 个 pins 全部按原字节复核；worker、manifest、native source/library 与 root 给定 SHA 一致。候选 28 个 Python 文件中，其余 26 个父文件保持；移除三个明确增量后 model/gdn.py 完整还原 V44。新增 helper 逐字等于冻结 V95（7f2d0136…19c5a6）。
- flag 在构造时 False，仅接受严格 bool，不新增 array 参数；只在原 V44 受形状/类型限制的 eligible 路径替换 helper。A baseline context 将 flag 关掉并绑定 accepted25 原 _decode，finally 恢复全部 flag/方法；B 资格要求 48 个不同 GDN、48 次精确 helper 调用且挂钩在 replay 前恢复。
- native 整文件逆改动后逐字等于 V89；仅添加精确 V95 PSO 的 S6 readonly / S_new9 nonreadonly 白名单、pos>=capacity 前置检查和已有 active MTLArgument 元数据导出。旧 range/alias/unique-use、fence、ICB 与 feedback 规则保留，ABI.h 等于 V89。
- native 仍只接受容量 64 或 256；Python 在 ctypes uint32 缩窄前拒绝负值/溢出/非 int/未知容量。pos<0 或 pos>=capacity 在 pos+1 前短路，修复继承的 INT_MAX 加法风险。已核冻结 UBSan log 的 748 cases 身份，并在独立 stdlib 模型重走 748 标量例；未重新执行 native/Metal。
- full10 schema 当前明确 PREDICTED_NOT_OBSERVED。worker 必须在首次 v34_replay_step 前从实际 PSO_records 得到唯一精确函数、reflection_observed=True、全部10个 index/name/type/access 完全一致，再逐对要求实际 S6=RO0、S_new9=RW1、同精确 PSO/offset0，且 48 eligible /83 staged；不把 RW1 伪称 WriteOnly2。
- actual95 local PASS receipt 和 release 原 SHA、128-row lane probe、30 o/S strict、15 readonly 与完整95 pin子集均被 source_check 硬绑定。它不能替代新 V99 的实际反射或整模数值资格。
- 保留 V89 canonical_output AST：真实 CAP64，256 仅 CPU BF16 每头补零和物理尾部零检查；old38/44 双 gold 93×132 输出原始字节与两 arm 全96 IDs 均是硬门，未引入新 math profile/容差。

复跑 6 项原 CPU checks、独立整文件反向源码/pins/门顺序验证和 --inert 均通过。root 后续实际 GPU 资格仍需证明 full10 schema、48 层 binding/eligibility、93×132/96 IDs、资源与释放；当前不声称数值、性能或采用 PASS。native 顶层旧 scope 文案仍继承 V89，实际 ICB 结论应依据 feedback.icb 和实际运行。

receipt.json / refsSHA.json / verify.py 保存可复核证据与定位；所有工具已返回，无活跃子进程，IDLE。
