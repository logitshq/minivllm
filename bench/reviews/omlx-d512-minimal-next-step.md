# oMLX D512：最短下一步依赖表

2026-10-17，只读源码盘点。未访问服务、账户配置/凭据、模型 tensor 或 GPU，未运行测试或重新散列大 runtime/model 文件。当前 `omlx_d512` 已有 CPU 准备的官方 CLI observer 与 raw SSE join；**尚无这组 D512 C1/C8 的实机资格或性能点**。下一步先完成 target-only 的 8 + 16 请求资格，再接 observer-OFF 正式计时；不需要另建通用客户端或恢复框架。

## 固定比较对象与真正入口

- 同一 `mlx-community/Qwen3.8-27B-nvfp4` snapshot `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`；沿用模型清单与原 tokenizer 文件，不换量化。
- 同一 [prepared-D-v1/fixtures.json](../m5_ultra_256gb/pareto_realistic/fixtures/prepared-D-v1/fixtures.json)，注册 SHA `6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db`：8 个完整 coding prompts，886–978 tokens，normal EOS，cap512。资格 C1 请求8个、C8请求16个；C8同时 outstanding，不能串行伪装。
- 独立 wheel 环境 `.artifacts/omlx-baseline/.venv/bin/python` / `bin/omlx`。本轮只读 dist-info 确认 oMLX `0.7.1.dev1`、MLX `0.32.3`、mlx-vlm `0.7.7`、mlx-lm `0.31.4.dev132+g94cdcae13`。这不是用户常驻服务的安装路径，不能混用其配置或身份。
- [serve_probe.specification(out,C,port,key,observed=True)](../m5_ultra_256gb/omlx_d512/serve_probe.py) 生成新私有 `state/`、单模型 symlink、env 和**同一直属进程内运行官方 `omlx.cli.main()`**的命令；`observed=False` 返回原 `bin/omlx serve`，配置相同。新 key 仅用于本次隔离 localhost 服务，不读取常驻账户 secret。
- 全量模型设置来自 [E020 注册 plan](../m5_ultra_256gb/external_spec_http/prereg-v1/plan.json) 的 `qualified_model_settings['omlx-none']`，原 SHA `e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc`。保留 MTP/VLM-MTP、DFlash、SpecPrefill、TurboQuant、ANE/CPU offload 等既定 OFF；不得只复制一部分字段。`balanced`、`chunked_prefill=false`、`prefill_priority=speed`、`decode_fairness=true`、`max_concurrent_requests=C`、cache disabled、keepalive off；四个 MLX 环境变量沿现有比较合同。

官方实际路径是 `server.create_completion:4110` → `stream_completion:5437` → engine `stream_generate` → `AsyncEngineCore.add_request:1446` → `Scheduler._process_batch_responses:12720`。预期 Qwen VLM 路径为 `VLMBatchedEngine`（`engine/vlm.py:4856/4945`）；`engine_pool.py:3727` 建立它，3807–3862 也存在 LM/VLM fallback。因此 **CLI 声明不能替代实际 loaded engine 身份**：保存本次加载日志、resolved model path、实际类/模块路径和 scheduler config；现 observer 已观察 target-only switches，但其六文件 PINS 不是完整 runtime/model identity，也没有直接记录顶层 engine 类。可在新注册观察版本补这几项普通 Python 元数据，或用充分的原日志闭合，不能默认授予 VLM 路由证据。

## 可以立即复用与仍缺的闭环

| 顺序 | 最小产物/入口 | 已有内容 | 本次必须补齐的证据与退出条件 |
| --- | --- | --- | --- |
| 1 | 固定 oMLX qualification spec / plan | `serve_probe.specification`、`jobs_for`，原 E020 source/runtime/model 清单 | 用全量 E020 oMLX inventory 加本次 probe/join/client/lifecycle 源，单列实际外层 Python/HTTP/tokenizer；模型/运行时前后核验，保存 settings 原字节、symlink realpath、argv/env。不要直接套 Lithos 的 pack/source 五组名称或 native model prepare。 |
| 2 | 单 fresh server qualification trial | `pareto_realistic/native_cleanup.run_group`、`http_capture_v2.System` 的 PID/listener/TERM→KILL→wait 基础方法；现有 trial 的独立 cleanup task 模式 | 小 oMLX trial 只换命令、私有配置物化、官方 readiness、HTTP job 和 join。`/v1/models` 只证可见模型，不证明已加载；实际首请求/加载记录补 loaded identity。保留全部原 stdout/stderr、observer、SSE、启动/关闭回执。 |
| 3 | C1/C8 两阶段 owner | 已审多直属 child owner 的 intent→Popen→identity、resource checkpoint、known/unknown spawn、reap 和 finally recovery 模式 | 真实阶段名 `omlx-c1`、`omlx-c8`，C1失败停止后续；前一个 server 严格 absent 后才启动下一个。复用参数化 OS/resource/recovery helper，绑定届时最新已审 RESTORED predecessor；不冒用 Lithos phase 或复制另一套服务控制。 |
| 4 | 关闭后独立资格核对 | `omlx_d512.join.join(group, observer_lines, jobs, fixtures_by_id, decode_text, probe_sha)` | C1/C8 全24请求：实际 full prompt、raw/committed IDs、greedy sampling、route/cache、UUID与原 wire 一一对应。C8 必须实际观察至少2条 response 的 batch；否则保留支持范围失败，不给C8资格。单独资源/恢复审计，join 自身不授这些范围。 |
| 5 | observer-OFF 正式 trial | 同 `specification(...,observed=False)`，现 SSE client 的 formal qualification机制与共同窗口统计；现 native/Lithos warm8→formal64模式 | 先绑定上步真实独立资格 SHA，再生成每次 request→fixture/body SHA映射；`usage_count_rule='identity'`、`raw_ids_count_rule='none'`。独立重读原 SSE 的UUID、finish、usage/cache、文本和时钟。没有正式 raw IDs，不能声称每次新 arrival 已逐状态数值验证。每 fresh process 8warm+64formal；完整闭环后才授性能点。 |

直接复用函数而非整条 native runner：`native_formal.py` 的 `/identity` 与 native `benchmark_*` 字段校验不适用 oMLX；Lithos NDJSON parser、`eval_count−EOS`、FIFO 和 pack 检查也不适用。现 SSE client 已有队列、立即补位、POST→DONE时钟和统计，无需重新实现。沿用其 bounded response close 合同；它不 drain DONE 之后的 EOF，记录应称“已保留 wire + protocol DONE + close 成功”，不能伪称完整 EOF 观察。

## 输入与 normal-EOS 必须重新观察

**不必把正式入口改成 chat。** 原 `/v1/completions` body 直接发送冻结 `prompt_text`；同时保留 fixture 原 `messages`、模板/处理参数及全部 `prompt_ids` 作为来源。不能重新套一层 chat template，也不能用 token 数相等替代全 ID 相等。官方 `create_completion` 会 encode prompt，observer 又在 scheduler 读取实际完整输入；join 已要求两者逐 ID 等于 fixture。若出现 prompt 后缀、thinking 标记或模板不同，保留失败并单独修订输入适配，不能改冻结 fixture 或在审计中截掉差异。

**需要本轮实际 normal-EOS/count 观察。** scheduler 12770 附近在 `finish_reason='stop'` 时不追加终止 token；wire `completion_tokens` 直接来自这个 committed 计数。因此 oMLX formal 的 committed count **等于 usage**，不再减一。probe 在处理前存 raw token，处理后存 committed IDs/count；join 验证 stop 的 raw=committed+[终止EOS]、length 的 raw=committed、无 post-EOS、cap512、完整解码文本和 cache=0。`stream_completion` 有 synthetic think-prefix strip，故不能忽略实际 text/ID decode门。

D8若没有 early EOS，明确 `observed_terminal_eos=0`；不能把源码检查或 CPU fake stop case写成实际 early-EOS 覆盖。若需要独立实测该分支，应另注册小诊断，不改变正式 D8/cap512 或强制 ignore-EOS。target-only完成后才另议 MTP：C8实际并发可能使官方投机 fallback，不能把 C1 MTP资格迁移给 C8。

## cap32 旧结果能继承什么

[E020](../../docs/experiments/m5-ultra-256gb/E020-external-spec-http-v1/README.md) 提供真实官方 HTTP 路径、隔离配置、官方包与模型清单、采样/无缓存设置、raw SSE时钟/文本/count重建方法、PID/reap/恢复模式。E015提供官方MTP attach和实际计数观察经验。这些可作为实现与来源复用，**不能继承为 D512 的 full-input、长decode、normal-EOS分支、C8多请求/多行、MTP并发或性能资格**；旧 cap32延迟/吞吐不得拼入新D512图。

本轮仅核小文件 SHA：

| 文件 | SHA256 |
| --- | --- |
| `omlx_d512/serve_probe.py` | `226aeb6818b954ab00f62062470868a4e4a18665a9b96fc8635d077aa000fe67` |
| `omlx_d512/join.py` | `e31b135db7e57ce99daa6bc93c6d51a67a09ebefc2a257a1a8b832a0a656bafe` |
| `pareto_realistic/native_cleanup.py` | `6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6` |

优先交付是 **一次 C1/C8 observed qualification 冷循环 + 关闭后独立 join/resource/restore**。正式计时仍为下一阶段；same weights/inputs、各自 route/count 正确不等于跨框架 logits/state或coding quality相等，报告须保留这个范围。
