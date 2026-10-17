# E017：Lithos DSpark 实际路由与短请求资格通过

两个 fresh process 各跑三次相同 19→32 greedy 请求，全部 **192 个 raw token IDs** 与普通 Lithos 控制一致。正常 EOS 配置为 `[248046, 248044]`，没有关闭 EOS；本请求未生成 EOS，因此不宣称覆盖 EOS 分支。prefix cache 未命中。

每次 DSpark 请求实际 **14 轮、98 proposals verified、17 accepted、31 decode tokens committed**，加上 prefill 产出的 1 token，共返回 32 tokens。接受 histogram 为 0×7、1×2、2×4、7×1，覆盖零接受、部分拒绝与全接受。本轮最后一块刚好到达预算，没有 overshoot；一般允许完整块多提交的语义仍与返回数分开记录。普通臂每次 31 decode rounds。这些计数不等于速度收益。

实际设备为 Apple M5 Ultra / 80 cores / Apple10，backend `m5_ultra_80c`，recipe 为空；没有套用 Max 配置。DSpark decode program 为 dynamic bound 8、586 ops，其中 43 个 draft ops，存在 Markov W2、verify_select 与 accept_scan；普通 decode 为 bound 1、535 ops，无 draft ops。完整 Program、每 kernel source SHA/macros 和原始 accept/conf 二进制日志已独立重算核对。

两臂 target pack capacity 为 2048 / 2054，DSpark draft capacity 为 2054；两个 program 的有效 target context 上限均为 2048。target 为 258 个 NVFP4 slabs；draft 为 27 个 NVFP4 + 1 个 BF16 slab，合并后的 packed slab 数不能冒充 checkpoint 矩阵数。quantization=None，原生 layout interleaved16/block；权重原字节的前后 SHA 由两次运行保存，CPU 审计不重复读取 30+ GiB payload。

19 个资源快照 pressure=1、thermal=0，swap=268.81 MiB 与 Swapouts=303419 没有增加。两个子进程 exit 0，wait/reap 与严格不存在检查通过。外部 omlx-server PID 74384 身份匹配并从 T 恢复 S。

`evidence.tar.gz` 保存完整 run/owner/lease/jobs、download/comparison/metadata、221 份被测 Lithos source/native binary、3 份 harness、4 份完整 Program、模型非权重 metadata 和原始 pack manifest；`manifest.json` / `tar.verify.json` 逐字节校验全部 327 个成员。`summary.json` 为独立 CPU 审计重算结论。

资格只覆盖同一个短 prompt 的重复请求、真实路由及 returned IDs，未验证全部 attention/GDN/conv/draft state，也没有 HTTP、长生成或速度排名结论。观察器、pack 构建、哈希与 Program 序列化都在此运行中，后续性能必须用独立无观察器流程。

## 外部源码许可证补充

本目录 `licenses/` 保存本机 Lithos 固定 commit `223658090ae549e59220bdec459957cbdad7926a` 的 `LICENSE`、
`third_party/MLX-LICENSE` 与 `third_party/NOTICE`。三份文件已逐字节核对该 Git commit；
适用于证据 tar 中的 `external/lithos-metal/` 外部源码及 native binary，具体范围按许可证与 NOTICE 原文。
它们作为独立补充文件保存，原 `evidence.tar.gz`、原 manifest 与 tar 校验文件保持不变。

| 文件 | 来源 | SHA-256 |
|---|---|---|
| [LICENSE](licenses/LICENSE) | `/Users/ryanli/code/lithos-metal/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| [MLX-LICENSE](licenses/MLX-LICENSE) | `/Users/ryanli/code/lithos-metal/third_party/MLX-LICENSE` | `ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01` |
| [NOTICE](licenses/NOTICE) | `/Users/ryanli/code/lithos-metal/third_party/NOTICE` | `cd80bddd6b6204dbc61c6ac762ed206c73d9264755d9884383d0860de29d185d` |

`licenses/manifest.json` 记录这三份补充文件的来源、固定 commit、大小及 SHA。
