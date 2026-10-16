# Independent Battery results replay — PASS audit; STOP-MICRO-GATE retained
2026-10-16；仅CPU stdlib读取/复算，未import MLX、未运行GPU/模型/探针/process命令；原raw未改。
Driver SHA256: 6f2b41217dbe7d0ea46316aa4f30cd953c0f31c002f6b3bf6f2d30f45ace135a；identity一致，26个冻结source SHA及runtime两模块SHA均复验。
范围仅Battery、GDN L0/32/62、8个唯一synthetic BF16输入；没有完整模型或全shard加载。
独立核120条byte记录：完整唯一label覆盖3×(8输入+2个fallback)×4段，shape/dtype/nbytes正确、finite/exact真、差字节0、左右SHA相同。
原张量输出未保存；此项复核记录与hash一致性，不把它描述为CPU重算GPU tensor bytes。
10对schedule精确吻合：AA/BB前后各一次+6对均衡AB/BA；20个block均prime16、repeat64、3组/12段、duration=end-start与ns/call逐项一致。
各pair全部12段hash一致，block1结束≤block2 prime开始；源审计确认hash只在pair后且每call eval全部四段。
位置控制A ratios=.9887399338/.9644176504，中位=.9765787921；B=.9230164569/.8813533994，中位=.9021849281。
控制按原median[.90,1.10]、each[.80,1.20]通过；B接近下界且后BB slot2快11.86%，残余位置效应明显，不能称修正后稳定。
6对A/B中位分别466486.6484375/465799.8046875 ns，复算降0.14723760096898575%；仅4/6胜（AB3/3，BA1/3）。
预声明门为≥5%且≥5/6胜，两项均未过；独立重算状态STOP-MICRO-GATE，与result.json完全相同，无adopt。
27个resource样本全部Battery32%/thermal0 nominal/pressure1；known inference matcher为0，其scope不排除GUI/未知GPU任务。
逐样本从原vm_stat文字重解析counter一致；全程SO/SI/PO/Compressions增量0，Decompressions+9；不推导频率/热态因果。
active由284911616降至142475264B，减少142436352B；deleted-reference payload142387200B；MLX peak285419756B。
这是selected3 active观测；cache仍143373841B，不能称物理内存全归还或full48内存资格。
child29080记录exit2并finally退出；controller cleanup与root posttool absence均ps rc1、stdout/stderr literal empty；正常STOP已清理。
旧AC实验21.9%仍不是已证真实收益，旧STOP不改；跨power/input/protocol不能用本实验证明旧物理原因已排除。
adopted/full_model_qualified/E2E_run/AC_frontier_entry均false；没有全模型/E2E资格或进一步GDN重测建议，封存此STOP。
原raw每文件SHA、完整pair复算、控制/资源/清理明细见INDEPENDENT-RESULTS-RECEIPT.json；原raw SHA再次复验全部未变。
