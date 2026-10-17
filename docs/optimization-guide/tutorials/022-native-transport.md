# E022：先验证 HTTP 发送是否增加了延迟

状态：**CPU 资格通过，计划已冻结，尚未运行 GPU 或真实服务。** GLM-5.3-Flash 下载与校验结束后，才进入唯一性能测量队列。

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
