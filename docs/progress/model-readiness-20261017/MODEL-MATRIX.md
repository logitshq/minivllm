**模型与基线准备情况，2026-10-17。** 本轮只读源码、metadata与已保存receipt；没有模型下载/加载、GPU或Git修改。性能未测记为UNRUN，不记为0。

| 模型/主战场 | 当前自研引擎 | 本机oMLX0.7.1.dev1 | 本机Lithos5485ad6 | 实际边界 |
|---|---|---|---|---|
| Qwen3.8-27B NVFP4，M4主线/M5各自实测 | frozen44已有整模严格数值；投机模块分别资格，不能混称整引擎通过 | 普通+external MTP已有M4实际数值V109；新配置须各机重新固定并实测 | 注册Qwen3_5；已有该家族baseline路线；DSpark属于额外资格 | 现有结果按机器、数学profile、功耗与计时域分别报告；不称已胜过所有OM/Lithos配置 |
| GLM-5.3-Flash，M5 256GB新主线 | **架构适配未实现**；不是已跑失败 | **源码发现/加载/实现存在，GLM实际UNRUN** | **此固定安装无GLM架构注册**，入口会拒绝；不外推最新版 | 本机48GiB M4不适用于该约190GiB全驻留checkpoint；M5仍须实际加载/数值/资源资格 |
| Qwen3.8-Flash-Next，M5次要候选 | QSA/MoE/gated residual/ngram等尚未接入 | qwen4_exp vendored模型与pre-load兼容存在；实际UNRUN | 此固定注册表也无Qwen4-Exp | 未选定量化checkpoint，容量与性能UNKNOWN；不是180B dense |

GLM官方说明320B总/18B激活，采用稀疏+线性attention和mHC。[官方模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash) 所选MLX卡声明62个shards共203,992,076,296bytes（约190GiB），mixed4/5/6bit、group64，走mlx-vlm的glm5_next，非mlx-lm；这里没有下载或验证那些shards。[量化模型卡](https://huggingface.co/mlx-community/GLM-5.3-Flash-4bit)

oMLX的GLM支持PR3184已于2026-08-27合并，但上游支持不等于本checkpoint本机资格。[官方PR](https://github.com/jundot/omlx/pull/3184) 本机已核实际链：glm5_next发现→VLM-native-text→VLMBatchedEngine→for_vlm pre-load compat→vendor路径校验→mlx-vlm加载。源码含DSA/MLA、MoE、vector-gated KDA、mHC、PoolingCache；TurboQuant明确绕过该复合cache。是否实际调用native或fallback要在M5运行中记录。

Lithos结论来自实际registry与loader，不是README零命中：仅注册Llama、Qwen3、Qwen3MoE、Qwen3_5、Qwen3_5MoE；serve/Session对未注册architecture直接raise。六个关键安装文件与5485ad6检出逐字节相同。peer05d75cf文档另外记录M5引擎2236580（upstream5485）；**b169bc4是DSpark的HF权重revision，不是引擎commit**。该peer快照说三shard已验SHA、Session尚待资格；这里没有检查远端机器，也不称Lithos最新版本整体不支持GLM。

GLM移植可复用MLX基础QMM/norm/elementwise与现有owner、资源和严格对照工具；不能照搬27B专用Metal/ICB资格。新增合同至少包括：vector-gated KDA与状态；稀疏MLA/DSA及pooled indexer cache；MoE路由/共享专家与clamped SwiGLU；mHC多流/Sinkhorn；per-module混合量化loader、tokenizer/视觉/MTP。现有Qwen scalar-g GDN、GQA、NVFP4g16布局不是这些算子的直接替代。

Flash-Next官方分解为125B主体/6B激活，另51B n-gram和4B MTP；其GDN还配合QSA、MoE和gated residual，不能只按参数相加后称dense模型。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

M5下一步先固定GLM量化revision/全部文件SHA、真实安装包/source/native库身份及tokenizer配置，再跑oMLX基线加载和短数值/资源检查；保存实际模型实现路径、路由/fallback、停止条件和输出。256GB候选机仍需计KV、workspace、OS和量化加载峰值，不能从190GiB文件大小断言可跑任意上下文。模型专属reasoning/template设置须固定；之后才讨论同条件吞吐与自研新架构。Lithos先核M5实际revision注册表，缺入口则记录该版本N/A，不伪造0性能。

精确路径/行号/SHA和各项未知值见MODEL-MATRIX.json、source-map.json、official-source-facts.json；无新的HTTP/Pareto胜出结论。
