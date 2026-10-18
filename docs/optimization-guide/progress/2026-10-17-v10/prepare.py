"""Prepare v10 only after CLOSED, independently audited oMLX formal-v2."""
from pathlib import Path
import hashlib,json,shutil,argparse
R=Path('/Users/ryanli/code/minivllm'); P=R/'docs/optimization-guide/progress'; OUT=P/'2026-10-17-v10'
SOURCES={
 'omlx-audit-supplement-verification.json':R/'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2-audit-supplement/archive-verification.json',
 'omlx-first-formal-failure.json':R/'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2/independent-formal.json',
 'v9-chartdata.json':P/'2026-10-17-v9/chartdata.json',
 'native-independent-audit.json':P/'2026-10-17-v9/sources/native-independent-audit.json',
 'lithos-independent-audit.json':P/'2026-10-17-v9/sources/lithos-independent-audit.json',
 'omlx-independent-audit.json':R/'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2-audit-supplement/independent-formal.json',
 'omlx-resources.json':R/'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2-audit-supplement/resource-summary.json',
 'omlx-restoration.json':R/'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2/restoration-audit.json',
 'omlx-archive-verification.json':R/'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2/archive-verification.json',
}
EXPECTED={'native-independent-audit.json': ('E023-native-formal-independent-audit-v1', 'b57e4e9657f3dddb04069290bdfbab4c99b0d6cbfb1427e454d0217d5463c870'), 'lithos-independent-audit.json': ('E023-lithos-D512-formal-independent-audit-v1', '768dc970dca201a700c94e07e17c5fdd399c903f444272ea150eb6336575c9c2'), 'omlx-independent-audit.json': ('E023-omlx-D512-formal-independent-audit-v2', 'a34a01882225ab1166fc8c4742c419c9ce2e697ed2e4931d49f7af65ae24232a'), 'omlx-resources.json': ('oMLX-D512-formal-resource-audit-v2', 'd488e30a1cef33663657a45ab74269753e821b21f1539b078c097be35ee206c5'), 'omlx-restoration.json': ('cold-service-restoration-audit-v1', '03476fd4f63087209e242ca4e8b072296c1983bca281bf3671e9e5134520c15e')}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_bytes())
def main():
 assert not OUT.exists()
 for n in SOURCES:
  if n not in ('v9-chartdata.json','omlx-first-formal-failure.json'):assert read(SOURCES[n])['status']=='PASS',n
 for n,(schema,auditor) in EXPECTED.items():
  d=read(SOURCES[n]);assert d['schema']==schema and d['audit_source_sha256']==auditor,n
 first=read(SOURCES['omlx-first-formal-failure.json']);assert first['status']=='FAIL' and first['raw_requests_opened'] is False and sha(SOURCES['omlx-first-formal-failure.json'].read_bytes())=='4b6f610dda24d0c4d52ae7ec22452709491a84bcd460f36814ae9ca1e416530f'
 a=read(SOURCES['omlx-independent-audit.json']);assert a['qualified'] and a['performance_qualified'] and a['formal_requests']==384 and a['warm_requests']==48
 assert read(SOURCES['omlx-resources.json'])['independent_resource_verified'] is True
 restore=read(SOURCES['omlx-restoration.json']);assert restore['qualified'] is True and restore['journal_sha256']==a['restoration']['journal_sha256']
 manifest={n:dict(path=str(p),sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for n,p in SOURCES.items()}
 supplement=read(SOURCES['omlx-audit-supplement-verification.json']);assert supplement['schema']=='E023-omlx-formal-audit-supplement-verification-v1' and supplement['status']=='PASS'
 assert supplement['original']['archive']==read(SOURCES['omlx-archive-verification.json'])['archive']
 for n,target in [('independent-formal.json','omlx-independent-audit.json'),('resource-summary.json','omlx-resources.json')]:
  assert supplement['audit_receipts'][n]==manifest[target]
 assert supplement['original']['sidecar_sha256']['independent-formal.json']==manifest['omlx-first-formal-failure.json']['sha256']
 assert manifest['v9-chartdata.json']['sha256']=='c0e5ac892a2d675b1d2e227cc1e8b214cb88630de7258fc018299f76d94a8534'
 assert manifest['native-independent-audit.json']['sha256']=='b669350e95ef07f4da5cac3a5cf6fb6fdaf2aea57e886d3ba60b9afd15e950bd'
 assert manifest['lithos-independent-audit.json']['sha256']=='562787c59f3a3b96a71b279538f61b38fd042dae9f2bc9c6117e022db7a6fb77'
 (OUT/'sources').mkdir(parents=True)
 for n,p in SOURCES.items():
  shutil.copyfile(p,OUT/'sources'/n);assert sha((OUT/'sources'/n).read_bytes())==manifest[n]['sha256']
 (OUT/'sources/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 template=Path('/tmp/progress-v10-render-template.py').read_text()
 (OUT/'render.py').write_text(template.replace('__SOURCE_PINS__',repr({n:m['sha256'] for n,m in manifest.items()})))
 shutil.copyfile(__file__,OUT/'prepare.py')
 print(json.dumps(dict(status='PREPARED',path=str(OUT),source_count=len(manifest))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
 if a.run:main()
 else:print('DESCRIPTION_ONLY: requires closed audited archive; no writes')
