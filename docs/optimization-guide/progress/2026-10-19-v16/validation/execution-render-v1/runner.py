import argparse,datetime,hashlib,json,os,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','render']);p.add_argument('attempt');a=p.parse_args()
r=pathlib.Path('/Users/ryanli/code/minivllm');b=r/'bench/m5_ultra_256gb/progress_v16';out=r/'.artifacts/m5-ultra-256gb'/('progress-v16-'+a.stage+'-'+a.attempt);log=r/'.artifacts/m5-ultra-256gb'/('progress-v16-execution-'+a.stage+'-'+a.attempt)
log.mkdir();snapshot=log/'source-before';snapshot.mkdir()
pins={}
for s in sorted(b.glob('*.py')):
 raw=s.read_bytes();(snapshot/s.name).write_bytes(raw);pins[s.name]=hashlib.sha256(raw).hexdigest()
python=r/('.venv/bin/python' if a.stage=='prepare' else '.artifacts/m5-ultra-256gb/progress-render-venv/bin/python')
cmd=[str(python),'-I','-B',str(b/(a.stage+'.py')),'--run','--repo',str(r),'--out',str(out)]
if a.stage=='render':cmd+=['--data',str(r/'.artifacts/m5-ultra-256gb/progress-v16-prepare-v1')]
env=dict(os.environ,MPLCONFIGDIR='/tmp/minivllm-plot-cache')
record={'schema':'v16-offline-command-v1','stage':a.stage,'attempt':a.attempt,'argv':cmd,'cwd':str(r),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_pins_before':pins,'environment':{'MPLCONFIGDIR':env['MPLCONFIGDIR']},'scope':'Closed evidence and static plot only; no model or service action'}
start=time.monotonic()
with (log/'stdout.txt').open('wb') as o,(log/'stderr.txt').open('wb') as e:
 result=subprocess.run(cmd,cwd=r,env=env,stdout=o,stderr=e)
record.update(returncode=result.returncode,elapsed_seconds=time.monotonic()-start,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_pins_after={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sorted(b.glob('*.py'))})
record['sources_unchanged']=record['source_pins_before']==record['source_pins_after'];record['status']='PASS' if result.returncode==0 and record['sources_unchanged'] else 'FAIL'
(log/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');(log/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
print(json.dumps({'status':record['status'],'rc':result.returncode,'output':str(out),'receipt':str(log/'receipt.json'),'stderr':(log/'stderr.txt').read_text()[-3000:]}))
raise SystemExit(result.returncode)
