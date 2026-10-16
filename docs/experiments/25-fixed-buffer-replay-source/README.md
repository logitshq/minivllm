# 固定 buffer 重放与固定容量资格记录

V33 native1–4 与 worker1–3 保留真实修复过程；native4/worker3 通过整模固定 L20 的两次重放，V34 direct 通过普通 MLX 路径固定 cap256 的32-token数值对照。尚无 native 连续生成、服务或性能资格。主运行时未替换。

`prototypes/`、`root-owners/` 按原字节复制，原路径与 SHA 在 `source-map.json`。`evidence.tar.gz` 包含四次小探针、两次整模的原始文本收据、资源与退出证明；成员 SHA 见 `evidence-index.json`。第一次小探针借用了当时尚未冻结的 worker helper，诊断停在未支持的 fence；它没有数值/性能通过资格。后续整模资格使用收据中明确的冻结源码和依赖 pins。

原始张量、模型及 dylib 不入包；原收据保留相应本地路径/SHA。脚本中的旧绝对路径与依赖需适配，不是独立安装包。编译命令见各原生目录的 CPU-READY.json。旧 STOP 不能视为新版本测试，也不能把固定输入两次重放称为 autoregressive inference 或 Metal ICB。
