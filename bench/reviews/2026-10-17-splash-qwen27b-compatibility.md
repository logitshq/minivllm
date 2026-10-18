# Splash 1.3.2：当前 Qwen27B NVFP4 checkpoint 的接入判断

2026-10-17 研究任务；仅源码、release metadata、3 个 safetensors 文件头。**值得做独立引擎对照，尚未构建、加载或推理，没有本机性能点。** 官方 DFlash8 可以作为单独配置参加对照；收益须包含 draft 成本，不能称为 target-only kernel 收益。

独立 checkout：`.artifacts/splash-baseline/source`，commit **`be3b083c5f413673d44e44881042e875e674e333`**，亦为 [官方 1.3.2 release](https://github.com/incoai/splash/releases/tag/1.3.2) 对应提交。没有修改 minivllm/官方 runtime、venv、模型或服务。检查证据位于 `.artifacts/splash-baseline/`；header 回执明确 `payload_verified=false`。

## 当前权重可以从哪条 loader 进入

| 问题 | 已核源码与本机 metadata；不是运行证明 |
| --- | --- |
| 必须转换成 ModelOpt 吗？ | **不需要。** `SafetensorsImage.cpp:185–226` 原生识别 MLX `weight:U32` + `scales:U8`，4 bits / group16 选 NVFP4；ModelOpt 的 U8/F8/global scale 是另一条分支。当前 config 的 `quantization={group_size:16,bits:4,mode:nvfp4}` 被 `ModelDescriptor.mm:318–341,530–541` 接受。 |
| 当前命名/shape 是否吻合？ | 本机 snapshot `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6` 的 **1,349 个语言模型 tensor** 均符合源码推导的 dtype/shape/name；3 个 header 与 index 的全部 **1,682** 键相符，另 333 键未纳入语言模型 shape 门。固定 64 层、48 GDN / 16 attention、5120 hidden、248320 vocab；不用换量化。 |
| 是否重新量化 target？ | `GgufPreparation.cpp:61–84,87–138` 将 MLX codes/scales 拷贝进 native block，MLX 缺省 global scale 为 1；目标存储重排不重新量化。这不证明计算位同。`SafetensorsImage.cpp:105–120` 将 a/b 合成 padded 256-row projection；GDN decay 转换也不同。 |
| 数值风险 | [开发说明 1664–1699](https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/DEVELOPMENT.md#L1664-L1699) 明示 `float(-exp(double(A_log)))` 可与 MLX float exp 差 1 F32 ULP。独立 Metal reductions、projection packing、DFlash verification 不能继承 minivllm full-state 位同资格。 |

对应主源码：[MLX loader](https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/SafetensorsImage.cpp#L185-L226)、[codes/scales writer](https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/GgufPreparation.cpp#L87-L138)、[27B geometry](https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/Qwen3_8.hpp#L19-L45)。header 检查不读取 tensor payload，也不等同于执行 C++ loader；正式加载前仍需原完整 model pin 门。

## 必须显式登记的引擎差异

- **DFlash2 / 8-row verification**：`NativeArguments.cpp:81–93` 要求 `target/`、`draft/`，`Runtime.cpp:2603–2606` 固定 anchor + 7 proposals；此提交公开选项没有 target-only。`temperature=0` 不关闭 draft。需固定 draft repo、revision、payload、实际占用及 accept/verify/commit 计数；目标权重相同仍只是引擎对照。
- **KV / S / ANE**：`server/serve_options.py:411–428,604–611` 支持 `--kv-format bf16 --state-format fp32 --disable-ane`。先用这组贴近当前质量条件；默认 INT8 KV、可选 BF16 S、ANE W8A8 另列配置，不能静默混入。
- **cache**：`--max-cache-disk 0` 仅关闭 SSD。`frontend.py:600–614` 的每请求独立 `cache_salt` 可隔离共享 prefix，但 RAM cache 管理仍存在。资格要求所有 warm/formal 请求 `cached_tokens=0` 且实际 `deduplicated_tokens=0`，不能把 salt 说成全 cache OFF。
- **两个进程**：Python HTTP frontend 在 `server/runtime.py:510–518` 真实 `Popen` C++ engine，并支持重启。现有单 wrapper PID 资源采样不够；下一 owner 必须记录 frontend/native 两 PID 与启动身份、实际物理占用、`restart_count=0`、所有退出/reap。不是简单替换 native server argv 就可开跑。

## HTTP / raw token 最小资格接线

`/v1/completions` 的 string prompt **不套 chat template**，但会按 tokenizer 加 special tokens；也接受 token ID 数组（`frontend.py:510–559`）。公平对照优先继续同一冻结 prompt text，并资格核对其完整 prompt IDs，避免再次 double-template。保留 greedy、正常 EOS、cap512，禁止强制忽略 EOS。

[SSE 1425–1483](https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/server/server.py#L1425-L1483) 输出 text、finish reason、timings、可选 usage/metrics、`[DONE]`；**公共 SSE 未提供完整 raw IDs**。最窄 observer 是 CPU 侧绑定 prepared prompt/public ID ↔ native request ID，记录真实 `TokensEvent(offset,tokens)` 与 `DoneEvent`；原 `protocol.py:1031–1050`、`runtime.py:283–345` 已有严格 offset/count 合同，不能把 re-tokenize text 冒充原 IDs。formal 关闭该 observer。

`Runtime.cpp:1835–1847` 会把 EOS/最后 cap token 作为 terminal anchor 发出；`Engine.cpp:1529–1532,2039–2042` 计入 completion count。**源码预测 EOS 也在 raw/usage 中**，须由实际正常 EOS fixture 验证，再显式推导公平 committed 口径；不直接套 Lithos 的减一规则。draft proposed/accepted 数量也不能当 committed throughput。公共 `timings` 的 decode rate 排除首 emission，不作为最终点：统一用客户端首 POST → 全 DONE 窗口、逐请求完整 latency p50/p95。

最短下一门：metadata/model-check → pinned draft + 实际 loader → 8/C1、16/C8 raw qualification（prompt、UUID、EOS、cap、decoded text、usage/cache、真实 batch/restart）→ 独立离线核验 → observer OFF、8 warm + 64 formal、C1/C8 各 3 fresh processes。跨引擎 full-state 未位同必须明确；质量要求由实际 tokens/约定质量门证明，不能由 speculative 算法名推断。

## 可以避免本机编译

官方 release 有 **`splash-1.3.2-arm64-macos26.tar.gz`，74,590,927 B，SHA256 `c3deb1dd971461ffafcf6b82c318789d6698b22aa865e490a11a0ed943686c0b`**。[资产清单](https://github.com/incoai/splash/releases/expanded_assets/1.3.2)与官方 Homebrew formula 相符；[固定 package.py:86–109,179–209](https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/dev/tools/package.py#L86-L109) 打包 native binary、`splash.metallib`、Python 和依赖，并在 `release.json` 写 binary/metallib SHA。**本次只取小 metadata/源码，未下载或执行该 bundle。**

本机 metadata：macOS 27.0.1；`DEVELOPER_DIR` 未设置，selected developer 为 `/Library/Developer/CommandLineTools`，`xcrun ... --find metal` 失败；标准 `/Applications`、`~/Applications` 未见 Xcode app，PATH/常见 Homebrew/AppSupport 位置未见 Splash。不能据此断言非标准路径绝无 Xcode。Makefile 要求 macOS ≥26.4、Metal4；当前不编译/安装，优先 root 串行取上述固定 bundle，免改已有环境。

以下仅为**之后串行 CPU 准备命令，尚未执行**；只解包到独立 artifact，不启动 launcher、模型或设备：

```sh
cd /Users/ryanli/code/minivllm/.artifacts/splash-baseline
mkdir release-download
curl --fail --location --output release-download/splash.tar.gz https://github.com/incoai/splash/releases/download/1.3.2/splash-1.3.2-arm64-macos26.tar.gz
printf '%s  %s\n' c3deb1dd971461ffafcf6b82c318789d6698b22aa865e490a11a0ed943686c0b release-download/splash.tar.gz | shasum -a 256 -c -
/Users/ryanli/code/minivllm/.venv/bin/python -I -B - <<'PY'
import hashlib, json, tarfile
from pathlib import Path
dst = Path('release-unpacked'); dst.mkdir()
with tarfile.open('release-download/splash.tar.gz') as t:
    t.extractall(dst, filter='data')
p = dst / 'splash-1.3.2-arm64-macos26'
r = json.loads((p / 'release.json').read_bytes())
assert r['version'] == '1.3.2'
for name, key in [('splash','binary_sha256'),('splash.metallib','metallib_sha256')]:
    assert hashlib.sha256((p/'engine'/name).read_bytes()).hexdigest() == r[key]
print('Bundle verified; no native/Python package entry was executed.')
PY
```

随后可单独执行 `release-unpacked/splash-1.3.2-arm64-macos26/engine/splash model-check mlx-affine none <固定snapshot>/config.json`；`runtime/main.mm:390–416` 的此分支只检查 config，先于 `runNative`，但本次尚未运行。不要用 `splash serve` 代替准备：launcher 会 device-check、组装/下载 target+draft；packaged 默认还写用户 AppSupport。真正资格前需独立缓存/assembly 路径与双进程 owner，保持原模型缓存只读。

可借鉴的设计假设：C++ 固定命令图摊薄逐 token Python 构图；DFlash 的多验证行摊薄权重读取；prefill 首 token 立即交付与 terminal anchor 省一次无用状态更新。三者分别改变 host 调度、投机算术/工作量、输出与 KV commit 时机；应分别核实际路径/计数和质量，不把它们合并归因成 NVFP4 kernel 更快。**尚无本机 win、速度预测或 Pareto 点。**
