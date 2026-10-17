# E012：把DFlash2选路留在GPU

2026-10-17；状态：原型通过，实际模块接入资格待测。基线`604ce07`，原型`e96c1a5`。

## 从成本出发划模块

每个最终接受token的成本是`(draft + verify + commit) / 实际产出`。
draft内部又包括上下文投影、5层草稿模型、目标lm_head和顺序选路。
现在的选路每个位置两次`.item()`，读取argmax和token；7个候选最多14次主机等待。
自回归依赖不能删除，但可以让前一个token的索引留在GPU，由下一次gather使用。

原型保持top-k、每项FP32 score表达式和顺序，只在最后取回完整路径。
模块开关为`enabled(device=False/True)`，作用域退出恢复原函数；真实对照直接调用
未修改的`DFlash2Drafter._select`。没有新训练权重，不改草稿长度、verify或GDN状态提交。
实际模块现已加入默认关闭的`draft_device_selection=False`参数，接入后的运行资格另测。

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

## 尝试1：原型初筛通过

24组合成和9组真实选择器score/IDs严格相等；9组完整draft的token及10个KV数组逐字节相等。
3组ABBA中，selector加速1.782–2.809×；完整draft加速1.077–1.252×，9点几何平均
1.177892×。这是同进程模块成本，不是整请求或HTTP吞吐收益。
[原始证据和审计](../../experiments/m5-ultra-256gb/E012-draft-selector-v1/)含51文件归档，
4个资源样本pressure1/thermal0，子进程回收和外部服务恢复通过。

## 尝试2：实际模块资格，运行前固定计划

核心新增`spec/device_selector.py`，只负责GPU上的依赖选路；`DFlash2Drafter`提供
`set_device_selection(bool)`，`load_drafter(device_selection=False)`及
`LLMEngine(draft_device_selection=False)`向下传递。默认关闭，切回False恢复原路径。
引擎拒绝非bool选项和未设置drafter却打开该开关，拒绝发生在加载模型之前。

`integration.py`直接调用最终核心；只在正确性诊断时捕获argmax输入来比较真实分数。
计时不捕获分数，不安装原型算法。运行前三次新进程已固定，每次都保留24+9组检查、
9组完整draft和三组ABBA。每次都须满足原门：全draft几何平均≥1.05且所有组≥0.97。
不因某次结果较慢而重跑，也不把三进程合并为更多独立prompt。

接口测试第一次失败于测试夹具：`object.__new__`不能创建MLX nn.Module子类。
改用`DFlash2Drafter.__new__`；没有为测试失败改变选路算法。
此处只授予模块资格，仍不授予SpecRunner回滚、输出分布或端到端推理资格。
