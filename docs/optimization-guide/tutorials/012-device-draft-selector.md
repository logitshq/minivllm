# E012：把DFlash2选路留在GPU

2026-10-17；状态：计划与可开关原型，尚未运行。基线`604ce07`。

## 从成本出发划模块

每个最终接受token的成本是`(draft + verify + commit) / 实际产出`。
draft内部又包括上下文投影、5层草稿模型、目标lm_head和顺序选路。
现在的选路每个位置两次`.item()`，读取argmax和token；7个候选最多14次主机等待。
自回归依赖不能删除，但可以让前一个token的索引留在GPU，由下一次gather使用。

原型保持top-k、每项FP32 score表达式和顺序，只在最后取回完整路径。
模块开关为`enabled(device=False/True)`，作用域退出恢复原函数；真实对照直接调用
未修改的`DFlash2Drafter._select`。没有新训练权重，不改草稿长度、verify或GDN状态提交。
生产路径尚未接入。后续若采用，另加明确的默认关闭开关，不做隐式环境切换。

## 预声明验证

固定DFlash2 revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`，默认affine4bit加载。
先做24组随机/零值/大尺度/非连续hidden检查，长度1/2/4/7；逐项score字节和选中IDs必须相等。
再用真实目标模型的数学、代码、中文prompt特征，block3/5/8，验证完整draft输出与全部draft KV。
选择器和完整draft分别跑三组ABBA；9组完整draft的加速比几何平均≥1.05，且每组≥0.97，
才进入独立接入资格。局部selector加速不作为整请求收益。

目标模型prefill只产生已经存在的历史特征；不向drafter传入gold未来token。
这项实验不调用当前存在已知GDN回滚资格缺口的SpecRunner，不能声称修复投机引擎或无损生成。
M4的literal-prefix-history事务与本模块独立；将来组合时重新验证。

[AdaFlash v1](https://arxiv.org/html/2607.19223v1)§3提示draft长度与验证成本应随接受情况调整。
这里先量清各模块成本，没有其训练出的长度头，也没有复现论文加速。

源码：[独立开关与选路](../../../bench/m5_ultra_256gb/draft_selector/candidate.py)、
[资格与计时](../../../bench/m5_ultra_256gb/draft_selector/screen.py)。
