# 06 DFlash2 投机解码：单流 25 → 56–62 tok/s，无损

日期：2026-10-14。文件：`minivllm/spec/dflash2.py`（drafter，230 行）、`minivllm/engine/spec_runner.py`、`cache.py` 的 snapshot/rollback、
`tests/test_spec_rollback.py`、`bench/spec_check.py`、`bench/spec_diag.py`。

## 为什么单流能翻倍

decode 一步 40 ms，其中 35 ms 是把 16 GB 权重读一遍，算力几乎闲着。一步验证 k 个 token 的成本（tutorial 00 的表）：k ≤ 4 几乎免费，5 要 +32%，8 要 2 倍。
投机解码用一个小模型猜接下来的几个 token，目标模型一次前向验证，猜对几个就白赚几个，猜错的从第一个错的位置截断，**结果和不投机逐 token greedy 完全一样**（无损）。

## DFlash2 的 drafter

- 5 层 Qwen3 式 transformer（32/8 头、head_dim 128），1.9B 参数，加载时量化成 4bit（1.2 GB），接受率和 bf16 一样。
- **上下文 = 目标模型第 5/19/33/47/61 层的输出拼起来**（每个 token 25600 维），过 fc + norm 成 drafter 的"记忆"；每层的 K/V 由同一份上下文算出来并缓存。
  所以目标模型 prefill 和 decode 时都要顺手把这五层的输出留下来（`capture`）。
- **一轮**：块 = [锚点, mask × 4]（用目标模型的词嵌入）过 5 层：动态卷积(prepare) → 注意力（q 来自块，K/V = 上下文 + 块，块内非因果：block diffusion）→ 动态卷积(finish)；MLP 同样夹一对。
  最后几个位置过目标模型的 lm_head 得 logits，每个位置 top-16 候选，**选路器**按位置顺序挑一条连贯的路：score = logit + (前一个 token 的 codebook ⊙ proj(hidden)) · 候选的 codebook。
- **验证**：目标模型一次前向 [锚点, 4 个草稿]（就是 prefill 路径），posterior = argmax，接受最长匹配前缀 n，再白拿一个 bonus token；下一轮锚点 = bonus。
- **回滚**：注意力 cache 把 offset 退回去就行（多写的几列下次覆盖）；GDN 的状态不能退，所以 verify 前快照、verify 时把每个 token 的递推输入记在"磁带"上，接受 n 个之后从快照重放 n 步（1 ms）。
  `test_spec_rollback`（0.8B fp32）：verify 8 个、只留 m 个、继续 decode == 直接按这 m 个 decode。

## 数字（27B nvfp4，decode-only，128 token，greedy 和不投机逐 token 相同）

| prompt | 不投机 | block 8 | **block 5** | dflash-mlx（block 5，4bit affine） |
|---|---|---|---|---|
| 数学推理 | 25.4 | 52.9（每轮接受 4.8/7） | **56.5（2.23×，3.3/4）** | 64.2 |
| 写代码 | 25.4 | 52.3 | **62.3（2.46×，3.6/4）** | — |
| 短中文问答（45 token） | 25.3 | 19.9（0.79×） | 26.6（1.05×，1.25/4） | — |

每轮成本：draft 14 ms + verify（块 5）约 55 ms + 回滚 1 ms。块 8 的 verify 要 95 ms，在 Mac 上不划算，所以默认块 5；
短中文回答接受率低（1.2/4），自适应策略按逐位置接受率低于 0.45 时缩到块 3。和 dflash-mlx 差的 12% 在 verify 的小 M 矩阵乘（它有专门的 Metal kernel）。

## 踩的坑

- 第一版引擎里接受率只有 1.7/7、6 tok/s：投机路径改了 `SchedulerOutput.decode`，调度器收不到 token，每轮都在同一个锚点上重来。drafter 单测（`bench/spec_diag.py`）是对的，整合错了。
- 自适应块长第一版用"平均接受数 + 1"：块缩到 3 之后最多接受 2 个，平均数再也涨不回去，正反馈锁死。要用逐位置接受率，它不随块长变。
- 多序列 decode 时投机不划算（verify 的 M 成了 B×k，算力瓶颈），所以只在"只有一个请求在跑"时开，进了 batch 就关；请求的上下文特征全程保留，回到单流时 drafter 补吃。
