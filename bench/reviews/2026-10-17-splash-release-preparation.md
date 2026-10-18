# Splash 1.3.2：实际隔离准备回执

接续[兼容报告](2026-10-17-splash-qwen27b-compatibility.md)。此次获得 **release 完整性 PASS + target/draft config 检查 PASS**；没有模型加载、draft 下载、推理、GPU 测试、服务启动、安装或 venv 修改。先前报告的“未下载/未执行”是前一阶段历史状态，完整原回执保留。

## 首次尝试

目录 `.artifacts/splash-baseline/release-prepare-v1`。实际运行 `/Users/ryanli/code/minivllm/.venv/bin/python -I -B /tmp/prepare-splash-release-v1.py --run`；runner 原文、阶段 JSONL、HTTP response headers、完整 stdout/stderr 和终场 result 均保存。第一次执行 rc0，3.853 秒，未重试/覆盖。

| 对象 | bytes | SHA256 |
| --- | ---: | --- |
| 官方 `splash-1.3.2-arm64-macos26.tar.gz` | 74,590,927 | `c3deb1dd971461ffafcf6b82c318789d6698b22aa865e490a11a0ed943686c0b` |
| `release.json` | 记录见原文件 | `f069fdc152dc5b3da57c50a4ab21d1425c0a8349a02f671004fea0362efe46a8` |
| `engine/splash` | 2,603,576 | `7acb1a0bc4f6e746a8bc206696ad3bc1dbff631958171f56662dcb57a16d54f2` |
| `engine/splash.metallib` | 36,807,832 | `a611aff2c9e61879345397ac5bea28a58f2278fcc3864176e7842ad20bd02b11` |

发行 archive SHA 与官方 release metadata 相符；后两项实际重新计算并匹配包内 `release.json`。8,127 个 archive members 先检查重复/绝对/`..`/非预期顶层/特殊文件/展开上限，再以 Python `tarfile.data_filter` 解包，随后核 symlink 最终目标仍在独立输出目录。原包、member inventory 和解包内容保留；未将任何程序放进系统 PATH。

## 为什么只执行 config 分支

固定源码 `be3b083c5f413673d44e44881042e875e674e333`：`runtime/main.mm:390–416` 在解析原生 serve 参数前分派 `model-check`；`ModelDescriptor.mm:744–759` 仅读 target/draft JSON；`:29–69` 限 JSON ≤1 MiB；`:682–704` 生成普通值结构。其 `kvLayout/gdnStateLayout/draft.stateLayout` 是 constexpr 值描述。Metal 设备创建只见 `MetalBackend.mm:1090,1166` 的 backend 构造和独立 device-check 路径，当前调用链不进入。**这是源码路径论证，不是硬件 trace；普通 host JSON/descriptor 分配仍存在。**

实际一次运行（0.489 秒，rc0）：

```text
<release-prepare-v1>/unpacked/splash-1.3.2-arm64-macos26/engine/splash
  model-check mlx-affine none
  <config-check-v1>/target-config.json
  <config-check-v1>/draft-config.json
```

输出 `{"family":"Qwen3.8-27B"}`，stderr 为空。两个输入是已保存的小 config 副本，不是 weights 路径。证据在 `.artifacts/splash-baseline/config-check-v1/{command.json,result.json,stdout.txt,stderr.txt}`。这只增加官方 config 兼容证据，不能升级为真实 loader/权重完整性或生成资格。

## 官方匹配 draft，固定 metadata

`install/families.py:24–28` 将 Qwen3.8-27B 对应到 [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/015e795645c74b1a0eeef3b570031fb62e769bc5)。本次 API 解析出的 commit 是 **`015e795645c74b1a0eeef3b570031fb62e769bc5`**；随后 config/README 均从该精确 revision 读取。

- 唯一 `model.safetensors`：**3,848,817,896 B**；HF LFS 提供的 SHA256 为 **`67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`**。**未下载/独立哈希该 payload**；文件大小不等同于 runtime live memory。
- 固定 config：1,239 B，SHA **`873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`**；BF16 源、5 层、5120 hidden、block8、selector rank256/top16、target layers `[5,19,33,47,61]`，与上述官方 config 检查相符。
- API、完整文件清单、config、README 与来源 URL/SHA 保存于 `.artifacts/splash-baseline/draft-metadata-v1`。API 全仓 listed size 3,849,113,004 B（包括图片/文本），不是需要加载的 tensor 总内存。

公开 `--revision` 固定 target；`upstream.py:554–574` 的首次默认 draft 选择仍解析 draft 默认分支。后续不能只传 repo 然后声称固定此 commit。最短可控方法是先独立下载/核验上述精确 draft snapshot，再用官方支持的 `--draft-model <绝对本地目录>`（`models.py:150–158`），冻结 assembly 记录及源 SHA，并采用 offline 启动。此步骤尚未执行。

root 下一串行阶段可复用已验证 bundle，无需 CPU 编译：先准备独立 assembly/cache（原模型缓存只读），完整 target/draft payload pins，然后补 raw `TokensEvent` 与 HTTP 绑定、正常 EOS/committed 计数、cache/dedup 零命中、Python frontend + native engine 两 PID 资源/回收，执行质量资格。DFlash8 独立引擎配置保留 draft 全成本；不据此宣称 target-only 算术相同或本机吞吐领先。**当前大下载、哈希、解包及 native config checker 均已结束；不在正式计时期间继续这些工作。**
