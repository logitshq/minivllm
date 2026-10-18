import datetime, hashlib, json, pathlib, shutil, xml.etree.ElementTree as ET
R=pathlib.Path('/Users/ryanli/code/minivllm');D=R/'docs/optimization-guide/progress/2026-10-17-v15';A=R/'.artifacts/m5-ultra-256gb'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
old=D/'validation/second-render';old.mkdir()
for name in ('D512-frontier-interactivity.png','D512-frontier-interactivity.svg','D512-frontier-latency.png','D512-frontier-latency.svg','verification.json','visual-review.json','render-history.json','result-manifest.json','README.md'):
 shutil.copyfile(D/name,old/name)
shutil.copytree(A/'progress-v15-execution-v1/render-v3',D/'validation/execution-v1/render-v3')
shutil.copyfile(pathlib.Path(__file__),D/'validation/finalize-label-v3.py')
for p in (A/'progress-v15-render-v3').iterdir():shutil.copyfile(p,D/p.name)
assert sha(D/'chartdata.json')=='7f44aa9223732127745684b5dc74646a7cb6e65720ef5615d78e4ac5264e61c7'
visual=read(old/'visual-review.json');visual['date_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
visual['images']={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in D.glob('*.png')}
visual['svg_checks']=[]
for p in D.glob('*.svg'):
 root=ET.parse(p).getroot();text=' '.join(root.itertext())
 assert all(s in text for s in ('Green = minivllm','minivllm historical N1','C1 zoom','GDN ON','confidence intervals'))
 visual['svg_checks'].append(dict(path=p.name,sha256=sha(p),xml_valid=True,labels_present=True,viewbox=root.attrib['viewBox']))
visual['render_verification_sha256']=sha(D/'verification.json');visual['second_render_retained']='validation/second-render'
visual['findings'].append('Third render adds minivllm to subtitle, historical legend and GDN C1 label; both PNGs viewed again, no text overlaps or data changes.')
write(D/'visual-review.json',visual)
history=read(old/'render-history.json');q=D/'validation/execution-v1/render-v3/receipt.json'
history['attempts'].append(dict(name='render-v3',status='PASS',receipt=str(q.relative_to(D)),receipt_sha256=sha(q)))
history['change']+=' Third render changes only visible wording to identify green configurations as minivllm; first two renders remain retained.'
history['current_renderer_sha256']=sha(D/'render.py');write(D/'render-history.json',history)
text=(D/'README.md').read_text().replace('后第二轮通过目视检查；首图和两次源码快照完整保留', '后第二轮通过目视检查；第三轮仅将副标题、历史图例和GDN C1标签明确写为minivllm，重新检查两图通过。前两版图和三次源码快照完整保留')
(D/'README.md').write_text(text)
manifest={str(p.relative_to(D)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(D.rglob('*')) if p.is_file() and p!=D/'result-manifest.json'}
write(D/'result-manifest.json',dict(schema='D512-v15-closed-results-manifest-v1',status='PASS',files=manifest,
 points=38,summary_groups=14,original_points_unchanged=26,source_only_manifest_preserved=True,
 actual_render='render-v3',visual_review='PASS',no_gpu_or_service_operations=True))
for name,pin in manifest.items():assert sha(D/name)==pin['sha256'] and (D/name).stat().st_size==pin['bytes']
print(json.dumps(dict(output=str(D),images=visual['images'],renderer_sha256=sha(D/'render.py'),
 chartdata_sha256=sha(D/'chartdata.json'),result_manifest_sha256=sha(D/'result-manifest.json'),members=len(manifest))))
