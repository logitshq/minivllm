# E022：先验证 HTTP 发送是否增加了延迟

状态：**v1 在第二个进程 readiness 时失败，停止矩阵；不进入 Pareto。** 原计划和结果保留，修复使用独立 v2。

E020 的当前 native C1 锚点是 716.10 ms / 44.69 tok/s，慢于同轮 oMLX target-only 的 679.32 ms / 47.11 tok/s。完整请求 latency 包含模型执行、排队和 HTTP 发送；所以应先检查具体等待，不能把整个差距归因于 kernel。

本机 CPython 的 `StreamRequestHandler` 默认没有开启 `TCP_NODELAY`，无缓冲 `sendall` 与 `flush` 也不会自动关闭 Nagle。httpcore 客户端和 Uvicorn 的 asyncio 服务端通常已设置该选项。E010 与 E020 native adapter 的 SSE payload/观察器不同，历史差值不是这一假设的因果证据。

最小候选仅设置 stdlib 现有的 `disable_nagle_algorithm=True`。两臂保持同一冻结 E020 native server、同一模型与 core snapshot、同一 SSE bytes、同一 client；每条 accepted socket 读取实际选项验证 OFF=0 / ON=1。首次 readiness 保存 receipt，计时中的每 token 不增加 observer。任一 socket setup/readback/receipt 失败都退出整个自有服务，防止 ThreadingMixIn 吞掉异常后被 readiness 重试掩盖。

固定三对顺序：OFF→ON、ON→OFF、OFF→ON。6 fresh servers，每个 3 warm + 3 formal，共 36 请求。全部 native raw IDs、visible IDs、19/32 usage、normal EOS、无 cache 命中、资源与回收条件通过后才有有效速度点。预注册推进门为：三个 paired E2E speedup 都 >1，几何平均 ≥1.05，每个 paired TTFT speedup ≥0.97。全部失败/负收益保留，不追加挑跑。

这只是 HTTP transport 实验。即使通过，也不能称为模型计算加速，不能覆盖 E020 原始结果，不能证明旧 payload 差异导致了历史性能变化。

16 项纯 CPU 测试已由实现者和主线程分别通过。其中一个 smoke 实际加载冻结的 E020 lifecycle/client/contract/launch，使用 fake process/socket/HTTP 完成 OFF/ON 各六次请求及 finally/reap；没有打开 socket、启动子服务或导入 MLX。另检查了所有 7 份 Python AST 和 3 个 CLI help。

源代码和完整设计见 [native_transport](../../../bench/m5_ultra_256gb/native_transport/README.md)。固定计划、CPU 结果、启动预览、jobs 及派发记录在 [prereg-v1](../../../bench/m5_ultra_256gb/native_transport/prereg-v1/plan.json)。Plan SHA256：`b239f6117c4a2b7799b5cf366b31c0a8277182f77773b5cc78c49f218f321fc1`。23 份新增 small pins 已核对，继承 E020 的 8658 份模型/runtime/core pins；正式矩阵前后再完整核对，CPU prepare 不重读大权重。

运行入口沿用已授权的暂停/恢复外层 lease，jobs 直接拥有 E022 controller，不套另一个模型 owner wrapper：

```sh
env MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 \
  MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
  .venv/bin/python -B bench/m5_ultra_256gb/suspended_service.py \
  --pid 74384 \
  --jobs .artifacts/m5-ultra-256gb/native-transport-v1-jobs.json \
  --out .artifacts/m5-ultra-256gb/native-transport-v1-service-lease
```

本次 CPU 资格不是运行结果；实测和最终服务恢复将在新小节追加，冻结源码与原门不修改。

## v1 实测失败：布尔 socket 选项不是固定整数 1

GLM 下载校验结束后执行了 v1。OFF 第一进程完成六个请求，ON 第一进程读取到
`TCP_NODELAY=4`，冻结校验 `actual == 1` 失败并 exit 70。ON 没有进入 warm/formal，
因此三对矩阵未完成，没有速度比或采用结论。原 16 CPU mock 未覆盖非零 flag 读回，
这是测试入口的可移植性缺口。不能将它写成模型数值失败，也不能用 OFF 单点替代配对。

两个模型进程已回收，端口消失，外部服务 identity 不变、T→S 恢复。
[66 文件完整归档](../../experiments/m5-ultra-256gb/E022-native-transport-v1/README.md)
包括源码、计划、所有请求、资源、失败及恢复凭据。v2 须先用真实 socket 核对布尔语义，
保留 raw 值，覆盖 0/1/4 正反测试，再重新冻结全新矩阵；原性能门不改。

## v2 修正与新计划

独立 [native_transport_v2](../../../bench/m5_ultra_256gb/native_transport_v2/README.md)
按 `bool(raw)` 校验，并保存原始 integer 与 normalized boolean。Apple 固定版本的 XNU
源码把 getter 写为 `t_flags & TF_NODELAY`、该 flag 为 0x4；本机真实 loopback probe
分别读取 OFF=0、ON=4，发送字节一致、全部 fd 和端口已关闭。证据与固定源码 URL 在 v2 目录。
18 项 CPU tests 经实现者和 root 分别通过，原 common 中顺序和性能门逐字节不变。

v2 独立计划 SHA 为 `e850dcdcc1d160ecddf512529758b32fc9a8241c6af515fb72901c4a6b28c81c`，
[prereg-v2](../../../bench/m5_ultra_256gb/native_transport_v2/prereg-v2/plan.json)
固定 24 份新增 small pins 与全部继承 E020 pins。仍是 6 fresh servers / 36 请求，
不用 v1 的 OFF 数据。当前为已准备、待提交后执行；没有新增速度结论。
