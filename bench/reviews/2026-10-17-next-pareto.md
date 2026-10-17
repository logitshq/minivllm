# 下一轮真实 Pareto：复用客户端，先固定合同

本笔记为静态规划，未注册/运行新的性能工作。正确性是硬门，主轴固定为实际整机
committed output tokens / 首POST至末DONE窗口，以及完整请求POST→DONE的p50；
p95、TTFT、长度/EOS/cap比例作为诊断。全模型仍需既定pressure1资源门与服务冷停/
恢复授权。E028两层合成state已过，真实模型37请求与64层refill状态门尚未跑。

## 最小native消融

复用E023 fixtures/client.run_group与现有owner/load/reap。先当前stack的E028 OFF/ON，
C1控制和C8 refill，各3fresh process，次序交错；不在正式计时挂step observer。仅新
server入口加默认off开关并在构造后调用integration，身份记录开关；不另造HTTP客户端。
原D组8条输入886–978token、greedy/cap512/normalEOS可作为负载候选；计划正式64请求、
固定8fixture轮换与完成后补充C个outstanding。此请求重复安排仍需解决下述prefix
cache合同才可冻结。各process点全部保留，配置点取各process统计中位数，不把请求
混池、不用64×512代替实际生成量。

官方对比另做同轮重测，连native胜出配置也重跑，不拼入旧19/32或前一消融作为新前沿。
oMLX的CLI/settings并发上限需都匹配C；target-only和MTP policy保留强合法配置，
实际MTP→target fallback必须记录且取得资格。

## Lithos实际可配置范围

已查当前editable安装0.1.2、clean commit `223658090ae549e59220bdec459957cbdad7926a`，
不是只根据旧429推断。`lithos-metal serve`→cli.py→serve.main，与E020的
`python -m monolith.serve`进入相同代码。源码SHA见同目录lithos-concurrency.json。

- serve.py:195–210用单GPU锁包住整次generation；427为app级FifoLock，490–491让
  非Ollama请求在重叠时429。单cached session和max_sessions编译variant不是请求batch。
- parse_args:731–770没有batch/concurrency/workers参数，uvicorn workers=1；
  kernel-config、prefill-chunk-size、draft-block-size调整kernel/prompt/proposal，
  generate.py的in_flight调command buffer，均不代表多条请求同时生成。
- 官方 `/api/chat`（662–668）允许真实并发HTTP在同一FIFO排队，一次生成一条。
  可以作为“HTTP C8 / single-generation FIFO”测服务吞吐与含排队的完整延迟；它是
  合法并发负载，不能因为GPU不batch而直接删掉。不能改为客户端串行、隐藏重试或
  多worker复制模型后冒充当前官方配置。OpenAI入口的429也须保留。
- 本版没有 `/v1/completions`。chat/ollama必须发送原messages并独立核模板后IDs，
  不将已rendered prompt塞chat再套一次模板；Ollama流的final done语义需要明确适配。

## 新长输入必须重新固定cache合同

serve.py:129–134强制prefix_cache=True，CLI无no-prefix-cache；默认内存cache预算
见generate.py:168–175。因此“8fixture反复warm/formal”可能命中，不能写成全框架
无prefix复用。下一计划必须先选择并验证无命中请求设计/明确命中服务工作负载，或
另注册窄adapter资格；不可静默修改官方实现、把不同prefill工作量合成一个前沿。

这不推翻已有E020图：其输入只有19token、Lithos chunk为128-exact，并且已冻结的
external_spec_http/contract.py:115–132要求全部六条Generation日志
cached_prompt_tokens==0；旧事实是“无命中”，不是“功能禁用”。本次没有重跑旧
数据，也没有新增吞吐/延迟点。
