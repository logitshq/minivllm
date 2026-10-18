"""Seal v11 small receipt copies only after the actual GDN37 cycle closes.

Default: description only. --run requires actual RESTORED + independent numeric,
resource and restoration PASS. It does not render, inspect ledgers, load models,
or create performance observations. Existing preparation is never overwritten.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
V10=HERE.parent/'2026-10-17-v10'
EXPERIMENTS=ROOT/'docs/experiments/m5-ultra-256gb'
ARTIFACTS=ROOT/'.artifacts/m5-ultra-256gb'
V10_RENDER_SHA='b0cc98f5a937d1aee07fe0504132e37475ed7ce60741cf4579ec8a194e9244aa'
ROUTE=EXPERIMENTS/'E024-route-http-v1'
CANARY=EXPERIMENTS/'E029-gdn-prework-canary-v1'
SERVING=ARTIFACTS/'gdn-prework-serving-v1-analysis'
OWNER=ARTIFACTS/'gdn-prework-serving-v1-owner'
CHILD=OWNER/'phases/gdn-prework-serving/qualification'
JOURNAL=ARTIFACTS/'gdn-prework-serving-v1-cold-journal/journal.json'
KNOWN={
 'v10-chartdata.json':(V10/'chartdata.json','8f5a5edba7d5171ee3c0ad7caf6a1dfece80fbda00a78b07f932c7c9291acbe8'),
 'e024-cycle.json':(ROUTE/'independent-cycle.json','8cbde8d54b246d25deeed2e7c25460d954087a2daf2083efdc9779402d2e313f'),
 'e024-c1.json':(ROUTE/'c1-independent-numerical.json','b9443261ca47dbea5782e469dca1987737e51df34a12c01539fbc46fe2a99ff4'),
 'e024-c8.json':(ROUTE/'c8-independent-numerical.json','4aac471df2778dd39ff74e10a7f965f2ad4690546c6dcaee9ce144e30c28e4d7'),
 'e024-restoration.json':(ROUTE/'restoration-audit.json','6bb4be00801ce4519a0ab399d171463e8c4a3e4c1802031295c965497316914e'),
 'e024-journal.json':(ROUTE/'restored-journal.json','dff835b4518710efe2fce1c6b3f12784432b0f3295a63aaa5503840e916ddd61'),
 'gdn-canary.json':(CANARY/'independent-numerical.json','8765a6089c362e37a89f66f8584c0bf39a45cdf929757a2f7bb87d764cd11df4'),
 'gdn-canary-resources.json':(CANARY/'resource-summary.json','8631f0dc594ba2ad639a0b2b386c8f2bacca6c268ed278f6beda4efa5d5c708a'),
 'gdn-canary-restoration.json':(CANARY/'restoration-audit.json','7ee69cc3babbe404438f526ddc8e97b068e24913386dd9b01c8298dc48fb24c4'),
 'gdn-canary-journal.json':(CANARY/'restored-journal.json','2091fa547a94a8538b44f64a3c8fdf964764113c5a8c5a7acf653861b7c8cf14'),
}
FUTURE={
 'gdn-serving.json':SERVING/'independent-numerical.json',
 'gdn-serving-resources.json':SERVING/'resource-summary.json',
 'gdn-serving-restoration.json':SERVING/'restoration-audit.json',
 'gdn-serving-journal.json':JOURNAL,
 'gdn-serving-owner.json':OWNER/'complete.json',
 'gdn-serving-child.json':CHILD/'complete.json',
}
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_bytes())
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def main():
 need(not (HERE/'sources').exists() and not (HERE/'qualification-sources').exists(),'exclusive new receipt snapshot')
 need(sha((V10/'render.py').read_bytes())==V10_RENDER_SHA,'unchanged original v10 renderer')
 # Inspect only small terminal files first. Never read a live steps/dispatch ledger.
 need(read(JOURNAL)['state']=='RESTORED','wait for actual service RESTORED')
 for name in ('gdn-serving.json','gdn-serving-resources.json','gdn-serving-restoration.json','gdn-serving-owner.json','gdn-serving-child.json'):
  need(read(FUTURE[name])['status']=='PASS','wait for actual terminal PASS: '+name)
 script=HERE/'render.py';unsealed=script.read_text()
 need(unsealed.count('QUALIFICATION_PINS = None')==1,'new unsealed renderer only')
 spec=importlib.util.spec_from_file_location('_v11_receipt_gate',script)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 paths={**{n:p for n,(p,h) in KNOWN.items()},**FUTURE};raws={n:p.read_bytes() for n,p in paths.items()}
 for name,(path,digest) in KNOWN.items():need(sha(raws[name])==digest,'fixed closed source: '+name)
 manifest={n:dict(path=str(paths[n]),sha256=sha(raw),bytes=len(raw)) for n,raw in raws.items()}
 prior,status=module.qualification_status({n:json.loads(raw) for n,raw in raws.items()},manifest)
 need(prior['performance_point_count']==14,'fourteen original observations')
 old_manifest=read(V10/'sources/manifest.json');need(set(old_manifest)==set(module.PINS),'original nine measurement sources')
 old_raw={n:(V10/'sources'/n).read_bytes() for n in old_manifest}
 for n,raw in old_raw.items():need(sha(raw)==module.PINS[n]==old_manifest[n]['sha256'] and len(raw)==old_manifest[n]['bytes'],'unchanged measurement source: '+n)
 # Validate every input before creating outputs; recheck copied bytes and originals.
 (HERE/'sources').mkdir();(HERE/'qualification-sources').mkdir()
 for folder,items in ((HERE/'sources',old_raw),(HERE/'qualification-sources',raws)):
  for n,raw in items.items():
   (folder/n).write_bytes(raw);need((folder/n).read_bytes()==raw,'exact source copy')
 write(HERE/'sources/manifest.json',old_manifest);write(HERE/'qualification-sources/manifest.json',manifest)
 need(all(p.read_bytes()==raws[n] for n,p in paths.items()),'closed source changed during preparation')
 script.write_text(unsealed.replace('QUALIFICATION_PINS = None','QUALIFICATION_PINS = '+repr({n:v['sha256'] for n,v in manifest.items()}),1))
 receipt=dict(schema='D512-progress-v11-preparation-v1',status='PREPARED',measurement_sources=9,qualification_sources=len(manifest),
  performance_points=14,previous_chartdata_sha256=module.V10_CHART_SHA,qualification_status=status,
  renderer_sha256=sha(script.read_bytes()),prepare_sha256=sha(Path(__file__).read_bytes()),
  scope='Small CLOSED receipt copies and source checks only. No rendering, model, live-ledger audit or new measurement.')
 write(HERE/'preparation.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if args.run:main()
 else:print('DESCRIPTION_ONLY: wait for actual GDN37 CLOSED independent PASS, then --run to seal small receipt copies; render separately.')
