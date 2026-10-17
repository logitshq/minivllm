# E013 成本定位

2026-10-17；基线E012开启，M5 Ultra / MLX0.32.3。9组32/256/1024历史与block3/5/8。
拆分计算与原draft的108个hidden/logits/KV数组严格相等，保留441条正式组件计时。
所有case最大组件是5层backbone；lm_head为原始完整draft的22.9%–28.9%。

[汇总](summary.json)、[69文件证据](evidence.tar.gz)、[哈希清单](manifest.json)、
[完整教程与失败记录](../../../optimization-guide/tutorials/013-draft-cost-profile.md)。
原v1因脚本名遮蔽stdlib profile导入失败，无模型或计时；v2正常完成，双方资源、回收和服务
恢复记录均在归档。同步组件不相加，也不据此宣称加速或GPU硬件占比。
