# 第30章文本归档

本目录归档V49/V52/V53控制漂移STOP、V51完整资格与负筛选、V55 build1 CPU发现/build2实际tiny、V56完整资格/早停丢弃、V58部分HTTP电源STOP与V59 Battery UNRUN草案。主文见[30. 提前提交一步](../30-native-ahead.md)。

- [source-map.json](source-map.json)：原始路径、精确字节/SHA、24–29章内容引用；每份原型历史CPU-READY措辞保持原字节，不代表当前GPU资格。
- [provenance.json](provenance.json)：10次实际运行、PASS/STOP/UNRUN边界与owner/release指针。
- [evidence-index.json](evidence-index.json)与[evidence.tar.gz](evidence.tar.gz)：全部467份实际运行文本，确定性gzip/tar；不包含权重、张量或native二进制。
- [dependency-map.json](dependency-map.json)与[omitted-binaries.json](omitted-binaries.json)：外部runtime/保存输入/原型二进制身份；归档不是重定位后可直接运行的模型环境。
- [verification.json](verification.json)与[manifest.json](manifest.json)：CPU原字节、原始统计和保存结果复核。

V55 build1未GPU运行；build2只在指定tiny通过。V56的12,540项检查不表示所有161交付步均做132项比较；early-stop六步只验ID/生命周期。V58仅2/9完整臂且电源STOP，V59未运行。没有采用、正式曲线或Pareto胜出结论。

V57、V60/V61/V62/V63/V64以及后续实验不在本章范围。所有源码只作历史来源，不应因阅读或解压而自动执行；任何新运行都需要独立资格与root调度。
