# MLX upstream follow-up — 2026-10-17

只读官方源码/PR与本机小配置；没有安装、构建、运行模型、测试或改变现有环境。结论：**最值得单独筛选的是 generic two-pass SDPA 的 TK=4；整包升级不是一个算术不变的性能开关。** 当前正式实验继续固定 MLX 0.32.3。

版本锚点：[v0.32.3](https://github.com/ml-explore/mlx/releases/tag/v0.32.3) 发布页标 2026-09-29、tag commit `64ea011`；[v0.32.4](https://github.com/ml-explore/mlx/releases/tag/v0.32.4) 标 2026-10-14、tag `36969c0`。以下三项均已合并，而且列在 0.32.4 release notes 中。日期采用 GitHub 页面日期，不额外断言时区。检查范围为该发布版和对应 PR/diff，未穷尽发布后 main 的全部变更。

| 变更 | 已核实合并 | 性质 | 对本机的判断 |
| --- | --- | --- | --- |
| [#4596: generic SDPA TK=4](https://github.com/ml-explore/mlx/pull/4596) | 2026-10-03，`0e3ff36` | 性能；仍须本机逐位资格 | 当前 masked decode 的长分支能命中，优先级最高 |
| [#4641: FP32 split-K partials](https://github.com/ml-explore/mlx/pull/4641) | 2026-10-08，`494c8a6` | 数值精度修正；改变舍入 | NVFP4 prefill 可达；不能沿用 E025 的 BF16 partial oracle |
| [#4633: Metal residency refresh](https://github.com/ml-explore/mlx/pull/4633) | 2026-10-06，`a59cc23` | OS residency/空闲延迟 workaround | 可测 idle→request 延迟，不能据此预报持续吞吐收益 |

## 1. SDPA：确实可达，但与 E024 bound 资格分开

[实际 diff](https://github.com/ml-explore/mlx/pull/4596/files) 只展开 generic `sdpa_vector_2pass_1` 的 key 循环，先读取四组 K/V，再依序做 score/online-softmax；specialized GQA kernel 不变。维护者报告 M4/M5 Max 多数形状更快，但 M5 Max 的 QH32/KVH32/context32768 有 0.9× 回退。PR 的正确性报告是误差阈值，不能当作我们的 BF16 逐位证明。

本机固定 Qwen config 的 QH=24、KVH=4、D=256；原 `attention.py:79–90` 使用显式 boolean mask。已保存 [v0.32.3 dispatch 源](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/scaled_dot_product_attention.cpp#L1014-L1019) 在 `g17d`、K≥1024 走 two-pass，而 specialized GQA 分支还要求无 mask。由此可推得：E024 bound1023 分支不受这项 kernel 改动影响，bound1536 分支会进入被修改的 generic 路径。这是静态可达性判断，未测收益。

最小未来实验：独立 runtime/单 patch 沙盒，固定真实捕获的 BF16 q/k/v/mask，B1/B8、L1023/1024/1536，先完整输出逐位+finite，再 paired operator timing。只有通过后才复用 E024 的全状态、changed-parameter、ahead/HTTP 门；serving 保持既有 gates ON/norm OFF。不能同时换 bound 或给当前正式进程换包。

## 2. split-K：升级会改变原数值合同

[实际 diff](https://github.com/ml-explore/mlx/pull/4641/files) 同时修改 affine 与 `fp_quantized.h` 路径：partial 从输入 dtype 改为 FP32，最终使用 `steel_gemm_splitk_accum`。BF16/F16 partial buffer 元素宽度从 2B 变 4B。PR 明确范围是 `qmm_splitk`，不含 `qvm_split_k`；其 M4 Max 测试不是 M5 Ultra 加速证据。

本机 E025 `prefill_nax_same_split/candidate.py:17–31,176–192` 的 T19 z/down/out 分别保留 split2/2/3、BF16 partial 和 axis0 sum。上游修正与这个原资格合同不同。因此新版本即使更接近高精度参考，也不能解释成 E025 原逐位门通过，更不能覆盖原失败。可行动点是未来 runtime 迁移时单独检查这三种真实 projection 的原 dtype 输出及内存；它不是当前算术不变优化的首选。

## 3. residency：只提出 idle 场景假设

[PR 与 diff](https://github.com/ml-explore/mlx/pull/4633/files) 增加 `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS`，默认 0，不启用刷新；正值启用定时 residency 请求。作者报告 M5 Pro/macOS27 GPU 空闲后恢复延迟，维护者明确称其为 macOS 问题的临时规避。它不改变 tensor 算术，但并非内存容量保证。

最小未来实验：同一新 runtime 的开关 OFF/1000ms，固定 0/3/10 秒请求间隔、pressure1、同 prompt/normal-EOS/count 门，比较 TTFT 和完整 POST→DONE，并保留实际物理内存证据。无需制造内存压力。当前连续补位的 64-request formal 无这些 idle gap，不能把该 PR 当作已有 throughput 改善；仅在 0.32.3 设置这个新变量也不能假定实现存在。

## 排除与边界

没有核实到可直接继承本机 E027 NVFP4 小 batch 逐位资格的新 NAX/shared-W 性能路线，也未找到可据以跳过 E024 changed-value 捕获门的 compile 参数捕获改动。已知 #4556 是异常清理，不列作性能成果。上游支持/合并不等于本机资格；现有 archive、runtime pins 和失败记录全部保留。

本地只读定位：

- `/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6/config.json:44,125,127`
- `/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine/model/attention.py:79`
- `/Users/ryanli/code/minivllm/bench/reviews/sources/mlx-v0.32.3-scaled_dot_product_attention.cpp:550,1014`
- `/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/prefill_nax_same_split/candidate.py:17,176`

来源阅读范围：两 release notes；三 PR 描述、merge event、维护者评价与相关 Metal diff；本地上述小文件。没有使用 PR 页面内第三方 cross-reference 对其他仓库的性能/正确性声称。
