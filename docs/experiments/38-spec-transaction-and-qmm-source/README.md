# 第38章精确源码与结果归档

配套[正文](../38-spec-transaction-and-qmm.md)。本目录保存V108–V112的冻结源码、独立审查及五次实际运行的公开文本证据。V108旧ID门FAIL、V111 MTP较慢均原样保留；V110仅CPU准备，V113不在本归档。

- `source-map.json`：187项原始源码身份；相同历史文本引用前面章节，新增文件按原字节保存。
- `evidence.tar.gz` / `evidence-index.json`：449个文本成员（417份原字节、32份凭据字段脱敏JSON view；含两份原始观察JSONL）、五次运行；无模型、raw tensor或原生可执行文件。
- `dependency-map.json`：516项冻结依赖。`external-fixture-identities.json`列未打包的外部文件身份；离机模式不验证其内容。
- `provenance.json`：实际状态、回收与释放凭证。数值探针的any-power物理释放与HTTP计时的AC资格分开。
- `omitted-binaries.json`：一个独立审查重复文本tar的身份，未重复打包；其中295份保持原字节、24份仅提供凭据脱敏view；private-original身份保留，离机校验不重新运行原审查tar。

在完整仓库checkout中运行（CPU/stdlib）：

```sh
python3 -B docs/experiments/38-spec-transaction-and-qmm-source/verify_archive.py
```

源机可加 `--originals` 核对原始路径及字节。离机模式检查公开文本、hash和保存的计数；32份脱敏view的原SHA仅为private-original身份，原件字节需 `--originals` 才验证，不执行MLX/native代码，不重新计算tensor，也不提供目标机运行资格。原harness绑定绝对路径；移植需要独立适配、编译、冻结和复测。依赖前章的精确内容，不能仅复制本目录当作独立发行版。

`publication-redactions.json`记录32处字段变换和原件／公开view双SHA，不含key值。完整私有原件保留在源机，未打包外发。

原始准备记录的“未运行”等文字按原字节保留；当前状态以本章正文和实际receipt为准。MLX/MLX-LM MIT、oMLX及Lithos Apache-2.0等许可在 `licenses/` 或前章源码映射位置保持。
