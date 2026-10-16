# 36. CAP64与另一种RMS舍入：基线变了什么

本章只收六次实际运行：V89整模资格、V90原生screen、V91局部诊断、V93整模FAIL、V97新参考诊断、V100新profile screen。物理CAP64仍匹配旧gold；另换plain RMSNorm叶子后，能对新的独立参考严格一致，但输出profile已经改变，速度未显示提升。

| 运行 | 保存结果 | 适用边界 |
|---|---|---|
| V89 | 93×132旧gold严格、96IDs一致 | 限定同步工作负载的物理CAP64资格 |
| V90 | 29.044589 decode token/s | 旧数学、CAP64的B-only绝对时间 |
| V91 | 27cases/108有限输出，两个舍入族 | 局部诊断，不是瓶颈证明 |
| V93 | 旧warm32-ID断言FAIL | 保留原门槛失败 |
| V97 | 393seed+3logits及93×132新A/B严格 | 新plainnorm叶子profile执行保真 |
| V100 | 29.044564 decode token/s | 新profile的B-only绝对时间 |

六次receipt、owner/release及资源原字节见[证据索引](36-cap64-and-rms-profile-source/evidence-index.json)。V95/V99/V101等actual不纳入；历史HOLD/FAIL/PENDING原文保留，以actual receipt区分结果。

## CAP64是真实GPU形状

attention写K/V后按ctx.max_len切片，再调用原SDPA；物理capacity与有效K/V长度不同。所读实际安装oMLX缓存也按chunk扩容，却返回有效idx范围，不能由step256推断每token都算满256。[V88审读](36-cap64-and-rms-profile-source/prototypes/v88-capacity-source-audit-build1/REVIEW.md)保留精确来源。

固定prompt19+生成32需要31decode，写pos19..49；changed20prompt写20..50，CAP64足够。V89在任何token前将物理cache bucket改为64，实际seed/capture K/V均为64；native独立允许capacity64，没有给GPU补到256的state。

A仍是旧accepted25 exact-L数学，B为fixed64 native反馈。每步物理64结果先查有效长度至64的尾部全零，再按head在CPU追加原始BF16零字节为canonical256，对旧V38/V44完整gold；不是只比较valid prefix。

三请求93×132=12,276项严格、全部96IDs一致；32个KV输出/步共2,976条tail-zero。48S ping-pong、83staged links、两ICB和原owner readonly保留。该资格不自动覆盖async多算、服务或其他shape。

## V91实测两种舍入族

[V87源码审读](36-cap64-and-rms-profile-source/prototypes/v87-omlx-math-profile-source-audit-build1/REVIEW.md)先排除一个猜想：本checkpoint161个zero-centered gamma，经源码FP32减1→BF16→加1→BF16的CPU位模型回环，668,672元素差异0。这不是GPU参数回读，也不授权改loader。

V91用三个原始BF16 gamma、九个固定X做27cases：A为fast RMS；B为冻结旧custom norm；C先把normalized落BF16再乘gamma；D为旧compiled显式FP32归一化、乘gamma后末尾BF16。所有27例A=C、B=D；跨族21例不同，最大1个BF16 ordered-bit ULP、绝对差0.25。

108份raw与162个pair指标经root独立CPU字节audit复算。它只支持这组输入的舍入差异；实际PSO/FMA模式仍未观察，不能扩成普遍等价、质量或性能结论。

## V93失败，V97另立参考

V92只替换161个plain RMSNorm叶子：B在任何token前由live offset派生并缓存BF16 gamma，调用fast RMS；A用独立checkpoint raw gamma和独立nn.RMSNorm。两个161-gamma集合字节相同、owner独立，原1349参数身份不换。GatedRMSNorm、GDN、conv、qk/gates/state仍是原44数学，**不是整个oMLX数学实现**。

V93按预声明旧warm32 IDs检查，在该断言处FAIL。161对gamma准备通过，但A/B prefill配对为0、native capture/dispatch/replay为0，不能记整模通过。失败前warm流未保存，首个不同token及是否匹配donor均UNKNOWN。[失败与ID证据](36-cap64-and-rms-profile-source/reviews/v93-failure-output-id-evidence-build1/REVIEW.md)原样保留。

历史official C1中，oMLX/Lithos的可见32-token文本流本就相同且不同于旧native workload。其内部raw IDs字段均null，所列32IDs来自可见文本tokenizer重编码。旧native回归流因此不是已证donor等价目标；这个事实不能修复V93失败，也不能推断其未保存的新流。

V97明确另立诊断profile，不放宽或重标V93。三次A/B独立prefill保留各自seed，393项seed原字节、3项末logits与首token严格一致；B只由自己的prefill播种，再消耗自己的GPU argmax反馈，不喂A或保存的未来IDs。

三请求93×132个有限物理输出新A/B严格；canonical旧hash为241相同、12,035不同，差异完整保留。base/reset新raw32与保存donor可见文本重编码32相同；changed仍与旧native raw32相同。这不是donor内部raw流实测、旧profile PASS或whole-oMLX等价。

V97 capture观察161个plainnorm、48个GDN调用，临时hook在replay前移除；161个cached gamma完整native readonly range可核，322gamma与原seed/capture owner保持readonly。两ICB各1609dispatch，93执行/banks48,45。[独立结果审计](36-cap64-and-rms-profile-source/reviews/v97-independent-results-review-build1/REVIEW.md)保留限制。

## V100未显示新profile更快

V90/V100均一load、3fresh warm+3fresh measure：t0在fresh prefill前，t1在首token/seed eval、reset与131seed上传后，t2在31native步及每步4B预测观察后。完整复制/hash、readonly、资源与文件写入都在decode时钟外，无HTTP客户端或公平配对A。

| 三个measure中位 | V90旧profile | V100新plainnorm profile |
|---|---:|---:|
| decode ms/步 | 34.429821 | 34.429851 |
| decode token/s | 29.044589 | 29.044564 |
| whole32 token/s | 25.710165 | 25.683568 |

V100绑定V97诊断与独立review；六次各32新参考IDs、792终态canonical哈希、192条tail-zero通过，186步/6reset/banks96,90。绝对decode中位几乎相同，数学profile不同且不是配对实验，不能称加速、HTTP/Pareto胜出或采用。

## 归档复算

[来源](36-cap64-and-rms-profile-source/source-map.json)、[依赖](36-cap64-and-rms-profile-source/dependency-map.json)保留精确字节与跨章`tar.gz::成员`，不新增旧大gold裸副本。只归档文本，108份V91 raw及native/runtime二进制仅身份。V100 root audit只有JSON，不声称另存了独立脚本；本章verifier自行核算统计。

```sh
python3 -B docs/experiments/36-cap64-and-rms-profile-source/verify_archive.py
python3 -B docs/experiments/36-cap64-and-rms-profile-source/verify_archive.py --originals
```

离机只核文本/哈希/保存结果；`--originals`另核本机来源、gamma范围身份与V91 raw有限数/162指标，不执行MLX/GPU或重建退出进程readonly。结果见[verification](36-cap64-and-rms-profile-source/verification.json)、[originals](36-cap64-and-rms-profile-source/verification-originals.json)与[manifest](36-cap64-and-rms-profile-source/manifest.json)。
