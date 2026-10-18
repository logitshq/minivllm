import datetime, hashlib, json, pathlib, shutil, xml.etree.ElementTree as ET
R=pathlib.Path('/Users/ryanli/code/minivllm');D=R/'docs/optimization-guide/progress/2026-10-17-v15'
A=R/'.artifacts/m5-ultra-256gb';P=A/'progress-v15-prepared-v1';F=A/'progress-v15-render-v2';E=A/'progress-v15-execution-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
assert not (D/'chartdata.json').exists()
shutil.copytree(E,D/'validation/execution-v1')
shutil.copytree(A/'progress-v15-render-v1',D/'validation/first-render')
shutil.copytree(P/'inputs',D/'inputs')
for name in ('chartdata.json','preparation.json'):shutil.copyfile(P/name,D/name)
for p in F.iterdir():
 assert p.is_file();shutil.copyfile(p,D/p.name)
shutil.copyfile(pathlib.Path(__file__),D/'validation/finalize.py')
data=read(D/'chartdata.json');old=read(D/'inputs/source-v14-data.json')
assert data['points'][:26]==old['points'] and data['display'][:26]==old['display']
assert len(data['points'])==38 and len(data['process_metric_medians'])==14
for item in old['process_metric_medians']:
 row=next(x for x in data['process_metric_medians'] if (x['configuration'],x['concurrency'])==(item['configuration'],item['concurrency']))
 assert row['metrics']==item['metrics'] and row['fresh_processes']==item['fresh_processes']
svg=[]
for p in D.glob('*.svg'):
 root=ET.parse(p).getroot();assert root.tag=='{http://www.w3.org/2000/svg}svg'
 text=' '.join(root.itertext());assert 'GDN ON' in text and 'C1 zoom' in text and 'confidence intervals' in text
 assert 'nan' not in p.read_text().lower().split('<!--')[0]
 svg.append(dict(path=p.name,sha256=sha(p),xml_valid=True,labels_present=True,viewbox=root.attrib['viewBox']))
images={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in D.glob('*.png')}
visual=dict(schema='D512-v15-visual-review-v1',status='PASS',date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 images=images,svg_checks=svg,method='Both final PNGs inspected with view_image; SVG XML and retained text labels checked. SVG not separately rasterized.',
 findings=['C1 zoom makes GDN, E024 and competitor endpoints legible without altering coordinates or adding jitter.',
 'Titles, legend, axes and caveats fit; principal annotations do not obscure endpoints. Near-identical OFF/history observations genuinely overlap.',
 'Native frontier selects separate GDN ON C1 and E024 ON C8 configurations, never their combined switches.',
 'p50/p95 joint frontier derived from data remains oMLX C1 and E024 ON C8; no inference about C2–7.'],
 first_render_retained='validation/first-render',render_verification_sha256=sha(D/'verification.json'))
write(D/'visual-review.json',visual)
write(D/'render-history.json',dict(schema='D512-v15-render-history-v1',tests_first_attempt='PASS8/8',
 tests_receipt='validation/execution-v1/tests/receipt.json',matplotlib_version='3.9.4',
 attempts=[dict(name=s,status=read(E/s/'receipt.json')['status'],receipt='validation/execution-v1/'+s+'/receipt.json',
                receipt_sha256=sha(E/s/'receipt.json')) for s in ('tests','runtime','prepare','render-v1','render-v2')],
 change='Only render.py gained C1 insets and an explicit Native GDN C1 annotation after first PNG inspection; math, gates, data and tests unchanged.',
 current_renderer_sha256=sha(D/'render.py'),original_26_points_unchanged=True,original_10_group_metrics_unchanged=True,
 v14_untouched=True,current_link_untouched=True))
table=['| 配置 | C | 进程数 | 输出 tok/s | POST→DONE p50 秒 | p95 秒 |','|---|---:|---:|---:|---:|---:|']
for row in data['process_metric_medians']:
 m=row['metrics'];table.append(f"| {row['label']} | {row['concurrency']} | {row['fresh_processes']} | {m['aggregate_output_tok_s']['median']:.3f} | {m['e2e_p50_s']['median']:.3f} | {m['e2e_p95_s']['median']:.3f} |")
readme='''# v15：GDN 改善 C1，联合前沿仍由 oMLX C1 与 E024 C8 构成

![完整请求归一化速度与吞吐前沿](D512-frontier-interactivity.png)

2026-10-18 实际闭合后更新。新增 [GDN-only 十二进程正式结果](../../../experiments/m5-ultra-256gb/E029-gdn-http-formal-v1/README.md)：独立正式审计、资源与恢复均 PASS，96 warm + 768 formal 请求完整核验。v14 原26个进程点及10组汇总值全部保留不变；新增12点后，共 **38个观察、14组配置/并发汇总**。这是实测更新，不是预测。

GDN ON C1 的三进程吞吐中位数为 **47.860 tok/s**，完整 POST→DONE p50 为 **10.694秒**。与本轮独立 OFF 配对，三次吞吐增幅分别为5.567%、5.574%、5.191%。GDN ON C8 的吞吐中位数167.420 tok/s，相对本轮 OFF 167.697 tok/s未呈现收益；不将其描述为全并发优化。

按配置汇总点独立计算，Native 自身前沿可择 **GDN ON C1、E024 ON C8**；它们是两个不同的可执行配置，**没有运行 E024+GDN 的组合**。p50、p95 两种口径的联合非支配端点仍是 **oMLX C1、E024 ON C8**。GDN缩小了C1差距，但本机这些测量不支持“全线超过”。

![完整POST到DONE的p50与p95](D512-frontier-latency.png)

下表为各配置独立进程指标的中位数。历史 Native N=1 只保留为历史观察；其余每格N=3。两轮 OFF 分开、不合并成六次重复，也不把跨引擎或跨周期结果称为配对实验。完整三进程 min–max 与配对比值见 [chartdata.json](chartdata.json)。

'''+ '\n'.join(table)+'''

横轴 `512 / median_process(p50_POST-to-DONE)` 是完整请求的归一化速度，不是 ITL/TPOT 或逐请求速度的混合分位数。原延迟图保留p50/p95。所有点的实际正式请求均输出512 committed tokens。误差线为观察到的进程 min–max，**不是CI或显著性结论**。

彩色线按每个引擎可选择的独立配置连接非支配端点；黑虚线是联合前沿。线段只引导阅读，**C2–7未测**，既不能内插性能，也不代表可执行的引擎/开关混合。HTTP并发不是GPU batch大小。形式上的点估计非支配不等于统计显著优势。

正式计时的observer关闭。资格来自先前真实捕获历史，不能据图声称所有未来调度、正式GDN命中次数、通用质量或组合开关已验证。此次绘图消费已审归档的小回执与manifest，并不重放HTTP或重新解压/哈希整个证据流。

8项CPU测试**首跑8/8 PASS**，原始stdout/stderr、命令及源码前后SHA均保留；Matplotlib3.9.4预检通过。真实prepare首跑通过。首轮渲染发现C1拥挤，只增加放大框与GDN标签后第二轮通过目视检查；首图和两次源码快照完整保留，数据/prepare/测试未修改。SVG通过XML及文本检查，未单独栅格化。v14和current链接未修改。

[闭合输入与准备回执](preparation.json) · [实际图核验](verification.json) · [目视检查](visual-review.json) · [全部尝试记录](render-history.json) · [原始CPU记录](validation/execution-v1/tests/receipt.json) · [PNG首版](validation/first-render/D512-frontier-interactivity.png) · [SVG主图](D512-frontier-interactivity.svg) · [p50/p95 SVG](D512-frontier-latency.svg)

`source-manifest.json` 原样保留的是执行前 PREPARED_NOT_RUN 历史快照；实际运行以 `render-history.json` 与 `visual-review.json` 为准。最终文件SHA见 `result-manifest.json`。复现使用已有可用绘图环境并写不存在的新输出目录：

```sh
.venv/bin/python -I -B docs/optimization-guide/progress/2026-10-17-v15/prepare.py \\
  --run --repo "$PWD" --out /tmp/progress-v15-prepared-recheck
MPLCONFIGDIR=/tmp/minivllm-plot-cache .artifacts/m5-ultra-256gb/progress-render-venv/bin/python -I -B \\
  docs/optimization-guide/progress/2026-10-17-v15/render.py --run --repo "$PWD" \\
  --data /tmp/progress-v15-prepared-recheck --out /tmp/progress-v15-render-recheck
```
'''
(D/'README.md').write_text(readme)
manifest={str(p.relative_to(D)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='result-manifest.json'}
write(D/'result-manifest.json',dict(schema='D512-v15-closed-results-manifest-v1',status='PASS',files=manifest,
       points=38,summary_groups=14,original_points_unchanged=26,source_only_manifest_preserved=True,
       actual_render='render-v2',visual_review='PASS',no_gpu_or_service_operations=True))
print(json.dumps(dict(output=str(D),images=images,chartdata_sha256=sha(D/'chartdata.json'),result_manifest_sha256=sha(D/'result-manifest.json'))))
