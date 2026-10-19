"""Owned, bounded numerical smoke. No performance measurement or service action."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm')
A=R/'.artifacts/m5-ultra-256gb'
E=A/'nvfp4-multirow-reuse-device-execution-v1'
O=A/'nvfp4-multirow-reuse-device-v1'
S=R/'bench/m5_ultra_256gb/nvfp4_multirow_reuse/smoke.py'
assert hashlib.sha256(S.read_bytes()).hexdigest()=='cac89f54ce7ded33e09b7b8e6bfa61caa03af1ae80f2c36b4b2e658afa0e8213'
assert json.loads((A/'glm-expert-repack-execution-v1/final.json').read_text())['returncode']==0
assert not O.exists()
E.mkdir();shutil.copyfile(__file__,E/'execute.py');shutil.copyfile(S,E/'smoke-source.py')
cmd=[str(R/'.venv/bin/python'),'-I','-B',str(S),'--run','--out',str(O)]
record=dict(command=cmd,started_at=time.time(),timeout_seconds=150,scope='Synthetic numerical-only alongside restored oMLX; offline GLM copy already finished; no timing claim.')
(E/'invocation.json').write_text(json.dumps(record,indent=2)+'\n')
with (E/'stdout.txt').open('x') as stdout,(E/'stderr.txt').open('x') as stderr:
    try:result=subprocess.run(cmd,cwd=R,stdout=stdout,stderr=stderr,timeout=150);record['returncode']=result.returncode
    except subprocess.TimeoutExpired:record.update(returncode=124,parent_timeout=True,child_killed_and_waited_by_subprocess_run=True)
record['ended_at']=time.time()
result=json.loads((O/'result.json').read_text()) if (O/'result.json').exists() else None
cases=list(O.glob('K*/T*/result.json'))
passed=record['returncode']==0 and result is not None and result.get('status')=='PASS_SYNTHETIC_BYTES_ONLY' and result.get('cases')==36 and len(cases)==36 and all(json.loads(p.read_text())['status']=='PASS' for p in cases)
record.update(status='PASS_SYNTHETIC_BYTES_ONLY' if passed else 'FAIL',case_receipts=len(cases),performance_qualified=False)
(E/'final.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True);print((E/'stdout.txt').read_text(),flush=True);print((E/'stderr.txt').read_text(),flush=True)
raise SystemExit(0 if passed else 1)
