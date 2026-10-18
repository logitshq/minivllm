# E3：不改量化数学，先缩短源权重的存活期

[六篇目录](README.md) · [上一节 E2](e2-verify-m1.md) · [下一节 E4](e4-spec-cost.md)

目标模型能常驻，不表示加载草稿模型时也能越过资源门。首次完整投机检查已经完成普通参考，却在加载 DFlash 草稿模型时触发系统内存压力，投机验证尚未开始。这是资源中止，既不是算法数值失败，也不能用此前局部通过替代完整检查。

baseline 加载器把 BF16 源权重留在字典中，为所有需要量化的 tensor 建立惰性图，最后统一求值。源数组是输出图的依赖；只写下量化表达式，并没有完成计算或释放原输入。于是最终量化输出之外，还同时保留了一大批源数据。

关键改动只调整生命周期。遍历时仅保存字符串键；取出一个完整 tensor，用原参数量化，把 packed weight、scale、bias 三份输出一起求值，然后释放这一份源所有者：

```python
# 生命周期示意，省略名字映射和未量化分支
for name in tuple(weights):
    value = weights.pop(name)
    packed, scales, biases = mx.quantize(value, **same_config)
    mx.eval(packed, scales, biases)
    out[name] = (packed, scales, biases)
    del value
```

真实实现仍按原顺序处理完整 tensor，沿用 affine4/group64；没有切块量化、改舍入或替换 target 参数。norm、codebook 等未量化数组仍由输出字典保有，不能把每次 `del` 都说成释放了该数组。开关默认关闭，也不引入新的推理数学。

先做的 draft-only 对照没有加载27B目标：175个输出参数全部逐字节相同，原参考参数只读，两臂350个输出对象独立。随后完整模型检查通过。加载阶段记录的 MLX allocator 历史峰值从约19.99 GB变为16.40 GB，差约3.59 GB，等于待量化 BF16 源数组总量；这不是物理 RSS 降幅，也不是速度结果。完整检查后峰值还会包含真实历史等活对象。

后续服务曾再次在 ready 阶段触发压力门，说明缩短源生命周期没有保证所有启动环境都安全。allocator cache 上限也不是总内存硬上限。资源门必须保留，每次成功或停止都按实际阶段记录。

**理解题：** 若只把 `pop` 改好，却把所有 `mx.eval` 留到循环末尾，为什么仍可能保留全部源数据？

原始记录：[旧40章](../../experiments/40-m1-verify-and-bounded-loader.md)；固定实现：[加载分支](../../code-versions/39-broadcast-m1/candidate116/tutorial_code/miniengine/spec/dflash2.py)；[旧→新 diff](../../code-versions/39-broadcast-m1/114-to-116-loader.diff)；证据：[完整检查与 allocator 口径](../../experiments/40-source/v117-actual-summary.json)、[后续资源中止](../../experiments/40-source/v120-attempt-summary.json)。
