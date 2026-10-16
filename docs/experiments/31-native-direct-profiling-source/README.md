# 第31章冻结文本归档

[正文](../31-native-direct-profiling.md)；[provenance](provenance.json)逐轮区分实际PASS/FAIL/STOP和历史CPU准备措辞。

- [source-map.json](source-map.json)：原始路径、精确字节、SHA及24–30章别名引用；包含suffixless LICENSE-MLX-MIT与Lithos Apache许可。
- [evidence-index.json](evidence-index.json)与[evidence.tar.gz](evidence.tar.gz)：16次实际运行的467份文本，包括V62最终reporter FAIL、V66布局guard FAIL及后续分开运行。
- [dependency-map.json](dependency-map.json)与[omitted-binaries.json](omitted-binaries.json)：绑定文本及外部native/runtime/保存tensor身份；不复制模型权重、张量或二进制。
- [verification.json](verification.json)、[manifest.json](manifest.json)：独立CPU原字节、旧gold输出哈希、原始时间戳及保存生命周期复核；不执行归档代码/GPU。

V63 timestamp原始单位UNKNOWN，分段encoder/fence/sampling有扰动，不能换算exclusive-ms或生产份额。V64 production660只覆盖五次末尾132，不是diagnostic12540逐步覆盖。V60/V61控制PASS但性能STOP；V67仅含seed设置的B-only engine计时。没有HTTP、配对或Pareto胜出/采用结论。

V65、V68/V69/V70只在正文提后续方向，不收源码或运行目录。当前文本包不得被当成新运行授权。
