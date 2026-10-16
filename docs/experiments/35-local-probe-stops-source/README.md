# 第35章局部probe文本归档

[正文](../35-local-probe-stops.md)只收9actual runs：78num/79num/80num+perf/81num/83num/84num+perf/85perf；不纳76/82/86/89/90实际运行。

- [evidence-index](evidence-index.json)与[evidence.tar.gz](evidence.tar.gz)：278份原始文本，确定性gzip/tar；九次owner/release/资源记录。
- [source-map](source-map.json)、[dependency-map](dependency-map.json)：原始路径/长度/SHA，跨24–34章精确成员去重，支持`tar.gz::成员`；旧大gold不再裸复制。
- [terminal-fixture-index](terminal-fixture-index.json)：1076份终态raw只存身份；[external identities](external-fixture-identities.json)及[omitted binaries](omitted-binaries.json)不含payload。
- [provenance](provenance.json)、[review provenance](review-provenance.json)：区分实际资格、历史准备、独立review与作者自查。83旧cpu-inert1标签不代表actual runtime；85author-self不是独立审计。
- [CPU verifier](verify_archive.py)、[verification](verification.json)、[originals](verification-originals.json)、[manifest](manifest.json)：保存结果核算与全文件SHA。

80慢65.26%；84+.591%/6wins和85+3.3246%/6wins仍均低于5%门槛，controls通过也不采用。81scaled state scalesarg3实际inactive，14/10等源码参数数目不能替代actual reflection。所有native终态数值检查只覆盖完成loop/block的最终slots，不代表全部dispatch状态。

默认离机只核文本/identity/statistics；`--originals`另需本地原fixtures并复核可用raw bytes/finite。无MLX/native/GPU执行或性能重跑；完整MLX MIT/Lithos Apache及来源许可按原字节保留或引用。
