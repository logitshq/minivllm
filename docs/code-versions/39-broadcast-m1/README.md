# 第39章代码版本

配套[教程](../../experiments/39-verify-broadcast-m1.md)。`candidate116/`是冻结V116的完整Python候选源码，保留V114仅target verify的广播M1开关，并新增默认关闭的bounded draftloader。未修改仓库生产引擎。

- `108-to-114-layers.diff` / `108-to-114-verify.diff` 对照原批量路径与新verify作用域。
- `114-to-116-loader.diff` 对照一次保留全部raw量化输入与逐项pop/eval/release。
- `source-map.json`固定原路径、字节数与SHA；`verify_sources.py`只校验源码与语法，不导入MLX或执行GPU。

```sh
python3 -B docs/code-versions/39-broadcast-m1/verify_sources.py
```

V113同输入497项数值严格通过；V114整模尝试因资源压力中止，只有普通A96 IDs通过、B未执行。V116已实际完成草稿175参数严格对照，独立结果审查已通过。以上不提供整模投机、性能、M5运行资格或默认采用结论。

这是一份代码交换快照；此处未打包完整原始运行证据、权重、Python依赖或机器环境。源机运行证据另存，状态与SHA见[进展记录](../../progress/2026-10-17/README.md)和[M4交接](../../handoffs/m4-max-to-m5-ultra.md)。目标机必须适配路径、冻结依赖和独立验证，不能直接把源码复制当作通过。

MLX衍生代码的MIT许可原样保留在 `LICENSE-MLX-MIT`，其他归属见仓库第三方说明。
