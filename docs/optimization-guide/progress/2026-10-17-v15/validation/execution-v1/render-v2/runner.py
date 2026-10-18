import argparse, datetime, hashlib, json, os, pathlib, re, subprocess, sys, time
R=pathlib.Path('/Users/ryanli/code/minivllm')
D=R/'docs/optimization-guide/progress/2026-10-17-v15'
O=R/'.artifacts/m5-ultra-256gb/progress-v15-execution-v1'
P=R/'.venv/bin/python'; V=R/'.artifacts/m5-ultra-256gb/progress-render-venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['tests','runtime','prepare','render-v1','render-v2','render-v3']);a=ap.parse_args()
O.mkdir(exist_ok=True);out=O/a.stage;out.mkdir()
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
sources={str(p.relative_to(D)):sha(p) for p in sorted(D.iterdir()) if p.is_file()}
for name in sources:
 p=out/'source-before'/name;p.parent.mkdir(exist_ok=True);p.write_bytes((D/name).read_bytes())
argv={
 'tests':[str(P),'-I','-B',str(D/'test_cpu.py')],
 'runtime':[str(V),'-I','-B','-c','import sys, matplotlib; print(sys.executable); print(matplotlib.__version__); print(matplotlib.__file__)'],
 'prepare':[str(P),'-I','-B',str(D/'prepare.py'),'--run','--repo',str(R),'--out',str(R/'.artifacts/m5-ultra-256gb/progress-v15-prepared-v1')],
}
if a.stage.startswith('render-'):
 argv[a.stage]=[str(V),'-I','-B',str(D/'render.py'),'--run','--repo',str(R),'--data',str(R/'.artifacts/m5-ultra-256gb/progress-v15-prepared-v1'),'--out',str(R/('.artifacts/m5-ultra-256gb/progress-v15-'+a.stage))]
record=dict(schema='v15-offline-command-v1',status='STARTED',stage=a.stage,argv=argv[a.stage],cwd=str(R),
 started_at=now(),source_pins_before=sources,environment={'MPLCONFIGDIR':'/tmp/minivllm-plot-cache'},
 scope='CPU tests/closed metadata/Matplotlib only; no model or service action')
save(out/'receipt.json',record);start=time.monotonic()
try:
 with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
  process=subprocess.run(argv[a.stage],cwd=R,env=dict(os.environ,MPLCONFIGDIR='/tmp/minivllm-plot-cache'),stdout=stdout,stderr=stderr,timeout=120,check=False)
 record.update(returncode=process.returncode,status='PASS' if process.returncode==0 else 'FAIL')
except BaseException as exc:record.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)))
finally:
 after={str(p.relative_to(D)):sha(p) for p in sorted(D.iterdir()) if p.is_file()}
 record.update(ended_at=now(),elapsed_seconds=time.monotonic()-start,source_pins_after=after,sources_unchanged=after==sources)
 if after!=sources:record['status']='FAIL'
 if a.stage=='tests':
  raw=(out/'stderr.txt').read_text();match=re.search(r'Ran (\d+) tests? in ',raw)
  record.update(expected_tests=8,actual_tests=int(match.group(1)) if match else None,
                full_unittest_output_retained=True,unfiltered_module=True)
  if not match or int(match.group(1))!=8:record['status']='FAIL'
 record['outputs']={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in [out/'stdout.txt',out/'stderr.txt'] if p.exists()}
 save(out/'receipt.json',record)
print(json.dumps(record));raise SystemExit(0 if record['status']=='PASS' else 1)
