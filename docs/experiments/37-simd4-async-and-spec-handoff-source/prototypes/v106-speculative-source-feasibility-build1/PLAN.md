# V106：同一 27B target 的 speculative decode，本地源码可行性

结论：有可复用的真实代码和本地 draft，方向可行；不是现成的 native 加速开关。最小先做现有 hardened DFlash2 路径的当前 target 数值闭环，MTP adapter 作第二条更轻的实现候选。没有 GPU/模型加载、权重 body 读取、下载、benchmark、shader/native 修改或版本实现；本目录仅 facts/blueprint。

1. checkpoint 与本地 draft

当前 target 是 workload.json 固定的 Qwen3.8-27B-nvfp4 revision 5ff8ef…67c6，qwen3_5、64层、hidden5120/vocab248320。config 的 mtp_num_hidden_layers=1 只是声明；全部1682 index键与三个header一致，无 mtp/draft/nextn 或 pre_fc_norm 权重，实际 target 不含可直接开启的 MTP head。

| 本地候选 | metadata/header 已核事实 | 尚未成立的结论 |
|---|---|---|
| Qwen3.8-27B-MTP-nvfp4 b9b764…b831 | 独立adapter，23 tensors，238,933,307B，qwen3_5_mtp，NVFP4 g16，block3；text_config 等于 target，tokenizer.json/vocab.json 字节相同；README声明同Qwen/Qwen3.8-27B并给出当前target配对 | 未加载/运行；不能把header完整叫tensor内容SHA或数值资格 |
| z-lab Qwen3.8-27B-DFlash2 50307d…e8a4 | 81 BF16 tensors，3,848,817,896B；5层、hidden5120、vocab248320、64 target层、taps[5,19,33,47,61]；fc[5120,25600]、两个codebook[248320,256]；block8。没有独立tokenizer，代码借target embedding/head | 契约与现有loader吻合，不是当前native加载/数值/性能资格 |

两个snapshot无缺失symlink，header所有data_offsets连续并覆盖文件payload；MTP index23键与header完全一致。这证明metadata/文件长度自洽，不验证未读取的payload内容。不会把MTP adapter换成target，也不加载第二个27B；target权重保持原revision。

2. 已有路径和真实历史

minivllm/engine/spec_runner.py 已按 [anchor,drafts] 一次target verify，逐位argmax接受最长匹配前缀+一个target bonus，cache保留anchor+accepted个输入；模型支持feature taps，DFlash2 loader默认将大Linear转affine4bit/g64。tutorial_code 版本另有严格block范围/hidden/tap/temperature0 guards，应优先以当前候选内的hardened版本为基准，不能直接退回缺少这些guard的旧minivllm文件。

旧 chapter10-checks/spec-checks.json 确有同target+同DFlash2、16个raw IDs完全一致，且等于当前workload前16；保存stats=[7accepted,8rounds]。但GDN replay误差在keep1/3为2.98e-8/5.96e-8，是容差检查，不能升级成V99全部state bitexact证明。原bench/results/spec_check_v3.txt 保存三prompt的45/128/128 IDs匹配与block5 26.6/56.5/62.3 tok/s；属于旧decode-only文本记录，math/code还ignoreEOS，非当前32-token/HTTP/控制器sealed性能成绩。旧docs的MTP速度引述和估算不是本地MTP运行凭证；本次检查的examples/P final记录没有找到local MTP actual receipt，不声称磁盘全局不存在。

3. oMLX 与 Lithos 的实际能力

实际安装oMLX默认 mtp_enabled=False、vlm_mtp_enabled=False、dflash_enabled=False。当前target没有embedded head，所以不能仅设置mtp_enabled就假称MTP启用。可用外部路径是vlm_mtp_enabled + vlm_mtp_draft_model + block size：engine_pool加载 qwen3_5_mtp 后 attach；loader失败会soft fallback到普通engine，未来必须保存实际drafter/counter/acceptance证据。外部MTP与其他spec路径互斥；非中性repetition/presence penalty或guided grammar冲突，thinking budget有专门processor支持。DFlash2也可在qwen3_5 VLM上经batched Lightning verify路径接入，需要dflash_enabled+draft路径，不能同时叠MTP。实际mlx_vlm有prefix+bonus、事务cache commit(accepted+1)和Qwen batch-invariant verifier代码；它们只作结构参考，vendor normalization/GDN数学不能替换本方旧target数学（V93失败仍保留）。

实际安装Lithos与固定5485ad…源的七个关键文件逐字一致：serve --draft-kind仅dspark；底层注册dspark/lm drafter，GPU round有accepted/commit/rewind。catalog为特定target配LithosAI DSpark，未知target需显式配对。HF cache目录未见该DSpark；现有DFlash2的dynamic conv/selector不能冒充DSpark权重，现有MTP adapter也不是Lithos现成MTP插件。不能调用会下载模型的自动pairing来补齐本次调查。

4. 最小闭环蓝图（先数值，后计时）

- 固定当前27B target/old38+44/V99 raw greedy定义，不修改target权重或math profile。第一候选复用现有DFlash2 loader与SpecRunner；先单请求小block2/3/5、固定真实draft，保持greedy、EOS、token budget与context容量。draft只看已提交的target features/anchor，绝不读取saved continuation/gold未来token作提案或B反馈。
- 核心缺口是target verify与rollback的singleton等价：现有多token verify走GDN chunk，T1/native走held32 Metal；量化projection的多row归约也可能与T1不同。抽象接受算法正确不保证有限精度target不变。先以实时serial target独立生成参考，逐prefix比较IDs、logits/全部canonical cache状态，构造首错/中错/全接收、重复reset、changed prompt、EOS/预算临界案例；保留严格失败，不放宽旧gold或用vendor wholeprofile绕过。若chunk不等价，下一项工程才是保当前math的block verifier/事务state，不先写性能声称。
- 部分接受必须提交anchor+accepted输入而不把bonus提前算入cache；注意KV offset/未提交尾列、48层GDN S与conv tail、draft上下文和feature frontier都一致。现有GDN磁带回放用chunk重算，需同样验证；不能只把最终S留在两bank便宣称可回滚。
- 原生V99固定T1图没有multirow verify、feature taps输出或prefix状态提交接口。先普通MLX数值闭环；数值通过后再另行捕获小block图、pin/range/reflection/ICB资格，不能直接复用T1 whitelist冒称spec native已支持。CAP64还要给未提交verify行留余量，最后remaining budget/EOS必须限制已交付tokens与状态提交数。
- MTP备选只加238.9MB adapter：还需minivllm侧fc(2H→H)、pre-fc norms、一层full attention及自己的KV、最终hidden/seed hook、target embedding/head绑定、accept/trim/replay runner。当前minivllm没有此runner，mlx_vlm接口/cache/layout与当前target不同，不能只import类便视为接通。

5. 资源与公平比较边界

当前P final C1-block1-omlx实际记录balanced、temperature0、normalEOS、draft=None；Lithos launch显式--no-draft。这些只是已测固定非spec点，不是全局最好。将来自身spec若快，必须允许oMLX同一27B target使用实际可用的MTP/DFlash2及其经资格的强配置重测，并允许兼容Lithos draft（若权重可用）；每种配置的scope/输出语义/失败fallback都明记，不能只挑旧balanced.03作唯一对手。

必须计入draft额外权重/量化临时峰值、KV、feature history、verify scratch、target rollback/tape与第二套state所有者，采样资源并保持独占串行运行。DFlash2 raw文件3.85GB不是加载后内存；MTP238.9MB也不包含activation/KV/head调用成本。吞吐要计完整draft+verify+commit/rollback+输出，engine与HTTP、首token/首块、prefill、实际接受/交付数分开记录。不同引擎可预声明各自合法block配置；不能拿target-only时间除以假定接受数，或把历史56–62投影成当前收益。

细节与SHA：checkpoint-metadata.json、draft-metadata.json、historical-evidence.json、source-map.json、refsSHA.json。当前结论是可行性及缺口，没有新数值或性能资格。
