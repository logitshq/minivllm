# 真 conv＋SiLU＋raw tail：未接入原型

这里保存源码阅读后写出的 V23 参考源码原字节副本。核心 Python 文件只读本地 Metal 文本；不自动 import MLX、建 kernel 或运行。当前引擎不引用此目录。

- 输入：BF16 raw `[1,10240]`、old_tail `[1,3,10240]`、weight `[10240,4]`。
- 输出：独立 SiLU 结果和 raw new_tail；保留 head split/repeat、gates、递推和 norm 原路径。
- `sum_y` 必填，不设默认，也不根据输出选树；未来须先核原 product 的 layout/flags 与实际 pipeline M/launch Y。
- 原目录的 12 项 CPU 元数据/结构检查通过，独立审查针对记录 SHA；未编译或执行 GPU，没有数值或性能资格。
- `cpu_structure_reference.py` 是历史检查源码快照，保留原字节并避开默认测试发现；其中 V18 资格文件路径是外部历史依赖，不能把它当作自包含测试发行包。
- `source-notes.md` 的相对证据路径也按原临时目录记录；用 `REFERENCE.json` 追溯。旧 V18 SiLU 资格不传递给本原型。

读取 API、Metal body、native helper 和 MIT 许可即可对照实现；运行前需在目标环境重新建立上述数值资格。
