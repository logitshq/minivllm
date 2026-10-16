V62 design：只组合 native55build2 FIFO 与57 strict no-alias direct feedback；不含51 hoist。

1. 新固定 ABI `v62_enable_direct_pipeline(2, diagnostic_snapshot)` 在 seal+ICB 后、任何 reset 前调用；使用 v55 reset/submit/wait/drain 和原 V55Core。旧 v55_enable_pipeline 与 OFF 同步路径保留。
2. 开启前重算57的两Sbank全部 registered output/input精确range无交叉重叠、input互不重叠和bounds；任何未知/identity/cycle/alias都拒绝。非S83条反馈仍逐字83个arena→arena blit；参数/PSO/math/geometry不改。
3. 诊断模式复用55的两个完整non-S stage slot：每步84个non-S outputs快照；48S仍按最后交付ticket.outputBank读取（与55相同，queued successor仅读该bank；全部132检查须在下一次submit前完成）。因此诊断可做93×132逐步严格检查。这里“全部132”不冒称132个都non-S。
4. 生产模式两个slot各仅4B，保存该步prediction（由role1 input的feedback link精确确定，不按名字猜）。compute→graphDone→prediction slot copy→stageDone→83 direct feedback→feedbackDone；下步compute等待前slot.feedbackDone。不拷20MB fullstage。旧F.stage allocation仍保留但不用，不声称释放。
5. 每个异步job保留ticket/CB/slot/inputBank/outputBank；普通v34 copy/compare在生产只允许最后交付slot中的4B prediction。48S/其他state/logits不暴露在inflight时。沿用core的FIFO、消费后再submit、depth2/slot占用保护。
6. 新 `v62_copy_quiescent_output(addr,n,dst)` 仅在pending和retained jobs均空、lastValid、last.id==requestSubmitted==requestCompleted==requestDelivered、当前request无discard/failed、无nativeerror时允许。读取最后outputBank的S或arenaOutput non-S，复制期间持S.lock。允许空drain保持last；非空discard/reset/下一submit会使上述条件失效。
7. 为什么最后全部132可直接读：无新job在写；直接反馈的所有dest都与所有graph output source不重叠；反馈不会破坏logits/cache/pred/nextpos/lens；S当前bank由最后交付ticket确定。这里只证明显式已注册range的copy，不猜shader访问extent。
8. 资格分两进程/固定mode、各只load一次相同candidate44权重。诊断：base/reset/changed93×132严格+readonly。生产：相同32IDs/三请求+readonly，普通每步只读4B；每请求完全交付且无pending后单次读取最终132并与goldA最后一步严格比。绝不声称生产有每步全132数值证据。
9. tiny两模式各3req×5steps：diag逐步75strict；prod15prediction+每请求final5并核全部输入/capture owner readonly。CPU测depth2/slot复用/FIFO/last失效、quiescence拒绝、mode读权限、source/dest bounds+两bank无alias。新tiny/full均须actual通过后才screen。

实际源码依据：native55build2 replay.mm:124–138（slot/job/core/read），317–350（enable/submit/跨slot fence），353–363（FIFO wait/drain），pipeline_core.h（lastValid/lastConsumed/pending语义）；native57 feedback_plan.h及enable函数（严格noalias），native57同步resultBytes不能直接用于异步。
