# E015：先证明官方 MTP 真在工作，再比较速度

2026-10-17，M5 Ultra 256 GiB。当前阶段：CPU 准备完成，尚未 GPU 验证。

M4 的 V109 已验证自己的官方 external MTP。该结果说明值得在 M5 复现，但不能作为 M5 的数值资格。此前 M5 的外部前沿只覆盖非投机配置；要追求最快推理，必须把可用的官方 MTP 纳入对手集合。

## 一条真正的官方路径

使用 `.artifacts/omlx-baseline/.venv/bin/python` 中安装的 oMLX 0.7.1.dev1、MLX 0.32.3、mlx-vlm 0.7.7、mlx-lm 0.31.4.dev132+g94cdcae13。实验 child 通过 `EnginePool.discover_models` 和 `get_engine(runtime_settings=ModelSettings(...))` 进入正常模型选择、加载与 drafter 附加流程。随后调用实际 `VLMBatchedEngine.generate`，由官方 scheduler 决定路由。

目标模型为 `mlx-community/Qwen3.8-27B-nvfp4` revision `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`，drafter 为 `mlx-community/Qwen3.8-27B-MTP-nvfp4` revision `b9b764dada894b8c416f89cbc8cf99bc7e20b831`。使用固定的原始 19 个 prompt IDs，greedy、32 个输出、正常 EOS、不改 thinking。MTP 保留官方 checkpoint 的默认 block size 3。跨请求 prefix cache 关闭。

这是一条官方 engine API 资格路径，**不是完整 HTTP server 资格**：没有 HTTP 编码、路由和 server admission enforcer，因此不能从它推出 HTTP 性能。普通与 MTP 各一个独立进程，每个顺序处理 3 个请求。主线程负责唯一 GPU owner 和外部服务暂停/恢复。

## 防止开关开了、实际却回退

官方加载 drafter 和路由都允许 soft fallback。只看配置中的 `vlm_mtp_enabled=true` 或文本看起来正常不足以证明投机已运行。观察器以 call-through 方式记录以下事实，不改输入、张量、采样和返回值：

- 真实 `load_vlm_mtp_drafter` 返回非空，resolved kind 为 `mtp`。
- 真实 `set_vlm_mtp_drafter` 后，scheduler 持有同一个 drafter。
- 每请求 `_route_to_vlm_mtp` 返回负 uid，且实际注册在 MTP active map。
- 每次 `_mtp_verify_target` 的输入宽度，与本轮提出数量加 1 一致。
- round 观察计数、drafter 的 `draft_lens` / `accept_lens`、实际 counter owner 的累计计数差三者一致。installed oMLX 的 `_MTPResetBindingProxy` 只代理属性读取，累计计数赋值落在 proxy 上；因此观察 `_record_speculative_round` 的真实 model 参数，验证请求内 owner 不变、每轮 before/after 连续。underlying drafter 计数另存为诊断，不能误当累计 owner。
- `_process_batch_responses` 返回的原始 32 个 token IDs 与 fresh 普通官方控制逐项一致；旧 native 输出不作为 oracle。

原始 events 每步落盘，失败时保留现场。所有 loaded source 路径、官方 distribution 的源码/本地库 hash、模型配置与权重 hash、实际 observer 接缝 hash 被保留。因为观察器写文件干扰延迟，本阶段**不输出性能排名**。

## 可复现入口

资格 child 为 `bench/m5_ultra_256gb/official_mtp/qualify.py`，参数为 `--mode none|mtp --workload ... --drafter ... --out NEW_DIRECTORY`。两个模式分别由已有 `guarded_v2.py` 和暂停恢复 lease 串行启动。输出目录必须未存在。`compare.py --control ... --candidate ... --out NEW_RECEIPT` 对 fresh 两臂作独立结果比较。

CPU 检查仅解析 AST、`--help`，不导入模型运行入口。GPU 运行仍待主线程调度；不得把 CPU 准备标记为资格 PASS。

## 后续无观察器计时计划

资格通过之后，再预注册独立的完整 HTTP server 配对实验：同机同 target revision、相同请求/greedy/EOS/cache 设置，普通与 external MTP 每臂 3 个 fresh process；平衡先后顺序，每进程 3 warm + 3 measured 请求，固定进程间冷却并保留所有点。正式计时使用原始官方 CLI，不安装这些 observer；官方运行日志辅助核对 MTP route，但不得靠本阶段的插桩计时宣称速度。完成后再与 native、Lithos、DFlash2 的对应质量合同和工作负载比较。
