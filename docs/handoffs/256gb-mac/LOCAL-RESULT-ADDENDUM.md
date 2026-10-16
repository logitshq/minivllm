# 本机后续结果补记

用户恢复测试后，V19 fullqual build3 在 AC Power 完成两次串行加载、accepted/OFF/ON/reset-OFF 四轮×32 raw IDs。128份保留FP32 logits逐字比较、16,384状态签名、完整cache/observation签名和18个fallback严格通过；ON六条真实生产路径每路1,488次调用。22份原始资源记录pressure1/thermal0；工具返回后owner/driver/worker四PID严格absent，外部释放与原进程内GPU_released=false边界分别保存。

[独立审查](FULLQUAL-BUILD3-REVIEW.md) SHA256 `dfc5c77df0f28c45d178de45c64b744235538440a6f2cf6eb2fe52c0c41533cb`；[审查收据](FULLQUAL-BUILD3-REVIEW-RECEIPT.json) SHA256 `af7b23397e16d615d19f8de4704d5c57d038797940d37bc3597c3056d6b78177`。每worker 1,349参数fingerprint在规定边界不变，跨worker的key/shape/dtype稳定字段一致；object_id仅各worker内部稳定。收据没有参数raw-byte hash，不能声称全权重逐字节对比通过。

结论仅限本次有界 FULLMODEL_QUAL_PASS。旧build1 AC与build2 Battery的token0前pressure2 STOP永久保留。E2E唯一实际attempt已执行6fresh，末尾资源STOP，无正式E2E/性能/采纳资格，无整体win，V19默认OFF、不采用、不合并main。完整目标机harness仍未实现，256GB目标机须重新确认硬件兼容性、建立路径/冻结清单、完整模型对齐，并在同机重测所有baseline。旧GO、latch和机器专属收据不得复用。

后续E2E：36请求/1,152tokens raw完整，但第6个pair3-B-on在请求完成后的after-measurement-before-unload因Sustained Swapouts（最近两delta>0）STOP。driver completed_configs=5，第6result.json与suite-complete.json不存在；不能当metadata bug或成功suite。实际工具exit1、driver wait/reap rc1，6server wait_completed=true/rc-15；工具返回后8PID strict absent，外部释放与原owner GPU_released=false分开。证据路径/SHA见 [精简索引](E2E-RESOURCE-STOP.json)；不重试同latch、不放宽guard，不写有效加速或Pareto胜出。
