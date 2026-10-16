# 第33章冻结文本归档

[正文](../33-native-range-compiled-step.md)只记录V72build1FAIL、build2numeric和STOPmicro、V74fullqual、V77screen；[provenance](provenance.json)保留五次实际运行及历史准备状态。V76/V78/V79等后续候选不收。

- [source-map](source-map.json)、[dependency-map](dependency-map.json)：原始路径/长度/SHA；重复文本引用24–32章原字节副本。
- [evidence-index](evidence-index.json)、[evidence.tar.gz](evidence.tar.gz)：五次运行189份原始文本，包含失败、资源、owner/release；确定性gzip/tar。
- [独立review](reviews/v72-v74-v77-results-review-build1/REVIEW.md)：72/128/STOP统计与74/77旧gold核对。原review的tensor复核在原机器执行过，离机不重新构造tensor。
- [terminal-fixture-index](terminal-fixture-index.json)：464份原始终态tensor文件仅存身份，不复制payload。
- [verification](verification.json)、[manifest](manifest.json)与[CPU verifier](verify_archive.py)：离机重算文本/hash/statistics；`--originals`额外核本机原件、native原始终态bytes与有限数。

每组8dispatch有barrier，native128和timed160仅是每loop/block最终八slot检查，不代表8192次中间状态逐次核验。V72controlsPASS但-0.516559%/0胜仍STOP；V77raw t1afterseed、B-only28.884518/25.520283不是配对或HTTP胜出。

实际安装oMLX GenerationBatch._step及scheduler wrapper无whole-generation-step mx.compile；V74是独立方案，compiled leaf不等于整step。原型PENDING/CPU_READY/noGPUyet文字保持历史原字节，实际结果以五份receipt为准。

MLX与MLX-LM MIT、oMLX与Lithos Apache许可分别适用；完整原许可文本及安装包METADATA按原字节归档或引用。archive不包含权重、tensor、native/runtime二进制，不是独立GPU执行包。

[旧gold依赖压缩包](historical-gold-dependencies.tar.gz)和[索引](historical-gold-dependencies-index.json)保存三份原字节 V24/V25 参考 JSON，只改变存储形式；它们不是本章的新运行。
