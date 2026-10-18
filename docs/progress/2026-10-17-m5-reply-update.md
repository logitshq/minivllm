# M4 / M5 最新进度核对

本次实际 fetch M5 至 `7ce72e5`，读取双向交接与 v9 性能记录。本方工作树起点 `cdb0649`；本次仅更新进度，没有执行新模型测试。

## M4：成绩未变，新研究尚未实现

最新实测仍为第44/45章，Qwen3.8-27B NVFP4、19输入→32输出、M4 Max 48 GiB。native C1/C2/C4 分别为26.0947/43.5117/63.3790 token/s，完整请求p50为1.2263/1.4706/2.0120秒。在已测短负载配置中组成联合前沿；oMLX来自同轮，Lithos来自后续独立AC轮次。

第46章为研究、源码对照与分工，不包含新候选实现或性能点。本次进程检查没有匹配到 Python、oMLX、Lithos 或 MLX 测试进程；电源AC、100%。不能把没有运行的候选描述成持续测试中。

## M5：Lithos长输出正式完成，oMLX仍缺正式点

相同八份coding fixtures、886–978输入、最多512输出、normal EOS。以下是配置的进程指标中位数，native每C一次，Lithos每C三次，跨周期、未配对。

| 引擎 | HTTP并发 | 总输出 token/s | 完整请求p50（秒） |
| --- | ---: | ---: | ---: |
| native | 1 | 45.3721 | 11.2833 |
| Lithos | 1 | 44.9102 | 11.4026 |
| native | 8 | 167.3989 | 24.4318 |
| Lithos | 8 | 44.9163 | 91.1835 |

Lithos六次fresh、384正式＋48warm完成，独立性能、资源和服务恢复检查通过。其C8为官方FIFO串行GPU执行。native点来自此前测量，不能称本轮新优化或配对增益，也没有新的三家完整前沿。

E024图复用已通过完整调度服务路径数值检查：37请求、12,411 committed tokens，以及42次参数诊断调用。真实HTTP capture/replay与正式性能仍待完成，局部forward约6%–7%的收益尚未成为HTTP点。

oMLX已通过24请求协议检查，确认C8实际decode batch达到8；首次正式周期在8个warm后，因客户端遗漏`input_tokens/output_tokens` usage别名而中止，未进入正式样本。失败证据及服务恢复已保存，`7ce72e5`登记严格别名一致性修复后重试；不能据预登记声称重试已完成。

来源：[M5固定v9图与数据](https://github.com/logitshq/minivllm/blob/7ce72e5/docs/optimization-guide/progress/2026-10-17-v9/README.md)、[同一提交的完整交接](https://github.com/logitshq/minivllm/blob/7ce72e5/docs/handoffs/m5-ultra-to-m4-max.md)。M4/M5的模型负载和硬件域分开，不合并曲线。

## 收到的研究反馈

M5明确已读本方`7ffe424`详细论文包和`cdb0649`回执，分工继续为M4混合prefill、M5图复用/补位长输出。

- decoded-W hoist对本方V124/E027自定义packed-W路径是新改动；M5原生MLX `fp_qmv_wide` 已有先解码后跨token复用，不能把它再称为原生M5的新优化。
- 投机成本表需要补 **T=3**：K=1/2/4个proposal加anchor，对应T=2/3/5，不能用T1/2/4/5插值代替缺失形状。
- M5后续交接指出E027的第19case在native multirow与per-row M1的桥接对照上失败，失败case未调用shared候选。不得把早期局部PASS外推为完整原生oracle资格，也不得称decoded-W已经被否定。

本方下一步仍是把这些提议变成独立实现和正确性/性能实验；当前尚无新增结果。
