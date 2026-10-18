# E4：接受更多 token，不等于完整请求更快

[六篇目录](README.md) · [上一节 E3](e3-draft-loading.md) · [下一节 F1](f1-http-measurement.md)

投机的目标是用更少的目标模型步骤交付相同输出，但草稿生成、整块验证、状态提交和服务传输都要付出成本。接受率只描述其中一部分。正确性资格通过以后，仍须从客户端发出 POST 开始，一直计到流中的 DONE，才能回答用户是否更早拿到完整结果。

这里的 baseline 是普通 greedy 解码，候选是已通过完整数值资格的 bounded DFlash 路径，并非另一轮官方外部 MTP。两臂使用同一19-token输入、最多32个输出和正常 EOS；输出预算与接受前缀仍约束实际交付，不能把草稿提案数当作交付数。

这次先启动一个新的 spec 服务，再启动一个新的 ordinary 服务。两臂都在挂接草稿前做一次普通启动请求，检查 raw32 和 runner 清理；只有 spec 随后加载草稿。启动检查完全在 HTTP 时钟外，之后每臂仍各做3次 HTTP warm、3次正式请求。一次新加载中的三次重复，不是三个独立 fresh 样本。

| 本次短请求 | POST→DONE 中位数 | 完成吞吐 |
| --- | ---: | ---: |
| ordinary | 1247.414 ms | 25.6531 token/s |
| speculative | 2555.926 ms | 12.5199 token/s |

固定32个输出时，吞吐就是 `32 / 完整请求秒数`，两个轴并不独立。spec 只有普通路径约48.8%的完成吞吐，虽然文本首 token 延迟略短。这说明更早开始输出与更早完成请求可以方向相反；本次只有一对服务，不提供统计显著性或完整前沿胜出结论。

12次 HTTP 请求共384个输出，另有两次启动检查共64个输出，全部 raw32 符合固定参考。资源采样在计时区间外，原来的压力停止记录保留。此次成功不能证明启动 warm 就是此前压力变化的根因。

本次没有保存 draft、verify、accepted 计数或独占 GPU 时间，因此只能确认完整请求更慢，不能猜测哪一段造成半速。长输出、不同并发或 lookup 路线需要各自的新资格与计时；旧46、47章只是后续研究。下一步应先补成本观测，再选择值得测试的改动，不能靠调接受率指标替代请求结果。

**理解题：** 若候选 TTFT 降低、接受率升高，但 POST→DONE 变长，你会报告什么结论，还缺哪类证据才能定位原因？

原始记录：[旧42章](../../experiments/42-startup-warm-short-request.md)；固定实现与差异：[源码 diff](../../experiments/42-startup-warm-short-request-source/source.diff)、[实际启动修正](../../experiments/42-startup-warm-short-request-source/corrected-launch.py)；证据：[请求白名单](../../experiments/42-startup-warm-short-request-source/requests-whitelisted.json)、[独立保存结果复核](../../experiments/42-startup-warm-short-request-source/independent-review/REVIEW.md)；后续研究：[46](../../experiments/46-next-ideas-from-papers.md)、[47](../../experiments/47-recent-paper-reading.md)。
