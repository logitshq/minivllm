# 给另一台 256GB Apple Silicon Mac 上 agent 的交接任务书

请接手 minivllm 教学引擎优化：先完成 V19 完整模型严格正确性，再测端到端，最后挑战 oMLX/Lithos。先用中文给出简明运行摘要，然后按下面顺序执行，保存失败证据与每次优化的 commit/tag。
目标目前只确定为“256GB Mac”；先 probe 芯片型号、GPU 核数、内存、系统和 Metal 能力。Max 与 Ultra 是不同芯片，不能互称；更大内存不保证更快。
本包是**可用的版本化 Python 源码 + harness 参考文本 + 任务书**，不是一键可运行的迁移 harness；没有模型、raw 数组、旧 GO 或 latch，也没有远程连接、push、PR 或发消息。

## 1. 先核对源码身份

| 角色 | Git commit | 作用 |
|---|---|---|
| accepted15 | `f976d6a547a9dc1946b52b1de6db376d799b34d7` | 已接受的教程基线；引擎25文件，GDN compiled gates 显式 opt-in |
| V19 code | `b158c0fbf95b84fb3567e0039b0ab1ffb1a46bb6` | 从 accepted15 分出的默认关闭 norm reuse；引擎26文件 |
| V19原始静态docs | `8aba03941634c73e0847c23d42a6051d43549525` | 仅修正讲解/证据身份；引擎与 b158c0f 逐字相同 |
| numeric archive | `abcbe74a1f54ce1db24d921d790ef462accf63f2` | 独立残差norm kernel + 数值结果文档/JSON；没有集成进引擎 |

`SOURCE-VERSIONS.json` 给出每个源码相对路径和 SHA；accepted25 原 manifest=`31fa053925bb732ce2a68b4feb8e4d2351d73939138f6a6c4a7d56df2b5cb41c`，V19 26=`d86be282646c1fc9d26155ad54014cf461f608c8222e32cd66d11088e180a32e`。
核心位于 `accepted15/tutorial_code/miniengine/` 和 `v19/tutorial_code/miniengine/`；V19 只改 `model/gdn.py`、`model/qwen3_5.py`，新增 `kernels/gdn_norm_reuse_metal.py`，其余23文件相同。
原 main 的用户修改 `tutorial_code/check_layers.py` 完全不在包内；不要覆盖原仓库用户改动。源码由指定 commit 的 blob 导出，没有读取工作区改动来打包。
私有 origin 为 `git@github.com:logitshq/minivllm.git`，不假设目标机有权限；先用本包，不需要 clone 才能读/运行引擎。导出目录没有 `.git`，目标创建自己的新仓库保留 provenance 即可。
不要把 V19、V18、残差norm、mask、packed projection、SiLU 等实验全部默认打开。先做单一 V19 treatment；不要无差别重试已失败方向。

## 2. 解包、环境与模型

本目录随实验分支/tag 提交；clone 后进入 `docs/handoffs/256gb-mac/`，或将整个目录复制到目标机。先执行 `shasum -a 256 -c SHA256SUMS`。自行选择目标目录；以下变量均指目标机，不能沿用源机器绝对路径：
```sh
TASK_ROOT="$PWD/minivllm-256"
TASK_ARCHIVE="$PWD/source-reference.tar.gz"
mkdir -p "$TASK_ROOT"
tar -xzf "$TASK_ARCHIVE" -C "$TASK_ROOT"
cd "$TASK_ROOT"
python3 -c 'import json,pathlib,hashlib; m=json.load(open("SOURCE-VERSIONS.json")); [(lambda p,h: (_ for _ in ()).throw(AssertionError(str(p))) if hashlib.sha256(p.read_bytes()).hexdigest()!=h else None)(pathlib.Path(r)/"tutorial_code/miniengine"/f,h) for r,k in [("accepted15","accepted15"),("v19","V19")] for f,h in m[k]["files_sha256"].items()]; print("accepted25 + V19 26 hashes PASS")'
command -v python3.13
python3.13 -m venv .venv
.venv/bin/python -m pip install 'mlx==0.32.3' 'mlx-metal==0.32.3' tokenizers numpy httpx huggingface_hub
.venv/bin/python -c 'import sys,importlib.metadata as m; print(sys.version); print({k:m.version(k) for k in ["mlx","mlx-metal","tokenizers","numpy","httpx","huggingface_hub"]})'
```
若没有 Python 3.13，先安装可获得的原生 arm64 Python 3.13；源机准确版本是 `3.13.14 (main, Jun 23 2026, 15:20:43) [Clang 22.1.3 ]`，优先一致，差异必须写入新冻结记录。不要假定 uv/hf/omlx CLI 已存在。
模型固定 `mlx-community/Qwen3.8-27B-nvfp4`，revision=`5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`；仓库名是 Qwen3.8，代码架构是 Qwen3.5，NVFP4 bits=4/group_size=16、BF16 activation。
下载只在目标机进行；不要从源机器复制约16GB模型。命令会打印目标 snapshot 路径，记录为目标模型目录：
```sh
.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("mlx-community/Qwen3.8-27B-nvfp4",revision="5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6"))'
system_profiler SPHardwareDataType SPDisplaysDataType
sw_vers
sysctl hw.memsize
```
先核验 config/tokenizer/权重索引和完整文件 hash。最小环境 probe（仅由唯一 GPU owner 执行，保存 stdout）：
```sh
.venv/bin/python -c 'import mlx.core as x; print("metal",x.metal.is_available()); f=getattr(x,"device_info",None) or getattr(x.metal,"device_info",None); print(f() if f else "inspect pinned MLX device API"); a=x.array([1,2],dtype=x.bfloat16); x.eval(a+a); print(a.dtype,a.tolist())'
```
另对实际 device family、BF16 中间舍入、NVFP4 group16 quantized path、threadgroup/grid/最大线程与 safe math 做小输入验证；不能只看到256GB或 import 成功就跳过。核验本地 MLX 0.32.3 header/二进制/Metal 库；更换版本须新建实验域。
固定 controls：`MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0`；没有 FAST_MATH 控制，不要误写。清理未知 MLX_/METAL_/DYLD_ override；父/子环境边界按重建后的协议冻结。

## 3. 先迁移 harness，再谈 actual GO

`reference/` 是只读历史文本：旧绝对路径、Python binary hash、native library hash、PID、电源域、freeze、GO校验都绑定源机。不能直接执行旧命令，不能仅 sed 路径就把历史 proof 当目标证据。
历史 fullqual build2 有1329 pins；本包没有迁移全部依赖/旧raw，不能声称1329 pins在目标全部可运行。E2E build2 为 CPU_BIND_READY_QUAL_PENDING，27 CPU checks/1421pins，未独立审查、actual E2E=0。
在新 root 重建最小可运行 harness：参数化 Python、模型、源码、输出路径；保留原算法、正确性范围、资源守卫、单次 attempt 和清理语义。对本机环境/输入重新冻结，先 CPU tests + fault rejection + 独立审查，再由目标 owner 生成新的授权与 latch。
可从 `reference/v19-fullmodel-qual-build2/{owner,driver,worker,replay,check_cpu}.py` 和 API 阅读原协议；具体实际有哪些文件以 `REFERENCE-INDEX.json` 为准。不要由模板或旧文件推断已经授权。
E2E 客户端/统计参考是 `reference/v19-e2e-incremental-build2/{driver,replay}.py`、`reference/bench_gdn_gates_e2e_v2.py`、`reference/bench_service_ac_frontier_v1.py`；bootstrap和资源helper也已附。依赖链须逐一审查，不承诺已封闭。
`reference/workload.json` 原 SHA=`65be876cac2397bf9afa76105c8e3fcce23de5e6c8652a28cb1b5f39640c153c`；只替换目标 model_dir，保留19 IDs及固定续写对照，再生成目标 workload SHA。
19 IDs=`[248045,846,198,95826,110827,98682,108829,81726,6297,1710,248046,198,248045,74455,198,248068,271,248069,271]`。
saved15 参考在 `reference/v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1/*/bank.json`（9 real+6 synthetic），带原raw文件名和SHA；没有数组，不可伪造replay，目标需重新捕获相同定义的bank。
全部参考文件的准确原路径、包内路径、字节数和 SHA 在 `REFERENCE-INDEX.json`；`norm-reuse-stationary-micro-build1/{frozen.json,driver.py}` 固定局部micro A/B与输入定义。
本任务交付范围到源码和任务书；**完整可移植 harness 尚未实现**。目标 agent 的第一项实现工作是上述迁移，不能跳过或将本包宣传为一键基准套件。

## 4. 正确性验收：优先 V19 fullmodel

只允许一个实际 GPU owner；accepted 与 candidate 串行加载，禁止同时常驻两份27B。独立读取原始资源快照后再 guard，pressure必须1、thermal0/1、swap/pageout稳定、进程清单已知、实际电源域固定；pressure2即STOP，不放宽阈值。
先 accepted15 gates ON；再 V19 candidate-off → candidate-on → reset-off，检查输出raw token IDs严格相等、logits与完整state签名按原严格策略、参数identity、cache生命周期、真实dispatch、fallback与hooks恢复。
原fullqual范围：4×32输出、128 raw logits、16,384 state signatures（每round4128 canonical数组hash）、18 fallback；复用 `replay.py` 的具体比较规则，不用“文本一样”替代状态正确性。
保留 BF16 sigmoid→BF16 multiply→FP32 的中间舍入；norm维度d=0..127串行FP32顺序、epsilon、两个rsqrt顺序、state表达式结合顺序不改；不要提前把scale乘进ready向量。
目标的 fullmodel PASS + 原raw独立 replay/review + 真实进程 wait/reap + 工具返回后所有实际owner/driver/worker严格PID absence，才允许性能测试。进程退出不等于进程内unload成功，首个失败与cleanup错误分开。
每次新优化独立commit/tag，保留默认OFF/严格bool/normal eligibility和exact fallback；失败也记录，不覆盖旧attempt。

## 5. 端到端协议与采纳门槛

固定19-token prompt，greedy/temperature0、max_new_tokens32、正常EOS（不能强关）、cache0、noDraft；原prompt正常生成32，无EOS时按实际32计。调度batch4不代表实际GPU batch4。
每个 fresh process 3 warm + 3 measured；6个fresh依次 off/on/on/off/off/on，构成3 paired，gates编译两臂都ON，只切换V19 norm reuse；32 raw IDs每请求严格一致，warm route检查后恢复hook，measure无新增hook。
客户端 x=`(实际N−首非空chunk的canonical token数)/(incoming DONE−firstText)`，先request→group median→process median3；它是stream delivery proxy，不是GPU token/s。
y=`实际总tokens/(last DONE−first POST)`，包括HTTP queue/prefill/服务开销；保留完整POST/firstText/DONE时间戳、TTFT和所有正负pair。
采纳须同时：3个pair的 y 均>0；aggregate y提升>=1%；aggregate x提升>=0。严格正确性先于performance；未达到就DO_NOT_ADOPT，不重试挑胜者。

## 6. 官方基线和独立Pareto

本机最佳已试 oMLX=`0.7.1.dev1` balanced/default burst budget .03（C2/C4），C1 balanced/aggressive近并列；不能称全配置全局最优。
Lithos固定源码 commit=`5485ad6bac0a5649d66a347f3e01a655ea29d937`，实际accelerated/noDraft；目标机必须真实选择和记录自己的profile，禁止冒用M5 Max 40c或源机M4配置。
`OFFICIAL-REFERENCE.json` 保存原launch argv、scheduler、source/native inventory与SHA；`official-source-locations/*.json` 保存实际环境/源码位置；`official-source-text/` 附已核对SHA的引擎Python文本，缺native/pack/build依赖，不能称完整可安装发行版。
旧 oMLX CLI argv 为 `omlx serve --model-dir <models> --base-path <state> --host 127.0.0.1 --port <port> --api-key <key> --headless --no-hf-cache --no-cache --max-concurrent-requests <C> --sse-keepalive-mode off`；mode/scheduler来自记录，不凭空加flag。
旧 Lithos argv 为 `<python> -m monolith.serve --model <snapshot> --no-draft --pack <packs> --local-files-only --host 127.0.0.1 --port <port> --max-context 2048 --keep-warm 0 --no-warmup`；先按目标安装源码和实际help核对命令，不假定已安装CLI。
不要将安装目录父仓库的Git HEAD误认成官方版本；以附带 actual_environment/source inventory 及真实official checkout metadata核对。目标完整获取/构建官方发行版仍需完成。
先在目标重测全部baseline C1/C2/C4及候选，在同一实际电源/系统/MLX域做fresh配对再宣布胜出；官方若只有usage+canonical text就如实记录，不能宣称有官方raw IDs。
目标机单独画x/y Pareto，不pool源机AC、Battery、accepted14、V18/V19或新机点；保留负收益和异常点。oMLX吞吐、C、TTFT、真实GPU batch分开解释。

## 7. 继承事实与禁止夸大

源机实际是M4 Max 40GPU/48GiB，native/oMLX Python3.13.14、MLX0.32.3；Lithos独立Python3.12.13环境见actual_environment，不强迫混装。V19局部15banks strict PASS；saved15局部host-wall A148.959340µs→B129.099457µs，降低13.332419%，不是端到端。
A为V17 AST（nn.silu→split/repeat→legacykernel），B为N2两kernel/V19同源码；Q/K成对norm逻辑body24576→16；A/B SiLU均10240。55296→10240只适用V18中间实现→V19，不能解释本次A/B局部收益。
fullqual build1（AC）和build2（Battery）均accepted load+sync后pressure2、token0前STOP，candidate forward=0；不是numeric FAIL，也不是fullmodel PASS。旧两次STOP永久保留，GPU后续释放与资格结论分开。
残差330 banks与mask352 banks仅synthetic数值PASS；3次微测controls无效、0 AB/BA有效比较，没有可采纳性能结果；abcbe74只归档，不默认集成。
V18曾fullqual PASS，但E2E aggregate y+.632032%低于1%门槛，DO_NOT_ADOPT；其他packedproj/SiLU等负结果同样保留。当前**整体baseline win=0**。
加载静态审计未证明harness长期额外保留多GiB；active/cache/peak与系统compressor不是进程物理footprint，不能断言CPU两份+GPU一份。
截至本交接快照：用户已明确恢复本机测试。fullqual build3 已在实际 AC Power 完成 accepted/OFF/ON/reset-OFF 四轮并通过独立 FULLMODEL_QUAL_PASS 审查；128 FP32 logits逐字比较、16,384 state signatures、18 fallback 严格通过，ON六条真实生产路径每路1,488调用，22份资源快照pressure1/thermal0，owner/driver/worker共4 PID工具返回后strict absent。跨worker参数key/shape/dtype稳定字段一致，object_id只在各worker内部稳定；receipt无参数raw-byte hash，不能声称全权重bytehash比较通过。审查SHA=`dfc5c77df0f28c45d178de45c64b744235538440a6f2cf6eb2fe52c0c41533cb`，审查收据SHA=`af7b23397e16d615d19f8de4704d5c57d038797940d37bc3597c3056d6b78177`，详见同目录审查文件与LOCAL-RESULT-ADDENDUM.md。E2E 唯一实际 attempt 的6 fresh已运行，36请求/1,152 tokens原始记录完整；第6个pair3-B-on请求完成后的after-measurement-before-unload因Sustained Swapouts（最近两delta>0）资源STOP，driver completed_configs=5，第6result.json与suite-complete.json缺失，不是metadata bug。8个实际owner/driver/server PID工具返回后strict absent，外部已释放；原owner GPU_released=false保留。无正式E2E/性能/采纳资格，无整体win；默认OFF、不采用、不合并main。旧两次STOP保留，旧暂停已被明确恢复撤销；后续读取最新STATE，本快照不是新运行许可。
本机普通app退出授权不自动转移到目标机；目标用户允许后才正常quit，拒绝保存/取消则停；禁止forcekill/discard，不碰Codex/Claude/editor/terminal、远程compute及未保存Excel面板。
最终先报告：目标硬件/环境、源码hash、fullqual verdict、E2E三pair/aggregate x/y、官方对照与Pareto、已保存负结果、下一步；未完成项直说，不能用局部micro代替全模型胜出。

本机48GiB结论：fullqual已独立PASS，但6fresh E2E末尾Swapouts resource STOP；完整请求raw不能代替成功suite。目标256GB须重建harness、重新冻结并同机重测，不重跑源机同latch或放宽guard。精简索引见 `E2E-RESOURCE-STOP.json`。
