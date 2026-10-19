"""Single explicit offline copy; original weights stay read only, no model runtime."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

R=Path('/Users/ryanli/code/minivllm')
N=R/'bench/m5_ultra_256gb/glm_expert_repack'
E=R/'.artifacts/m5-ultra-256gb/glm-expert-repack-execution-v1'
O=R/'.artifacts/models/GLM-5.3-Flash-4bit-stacked-v1'
PLAN_SHA='323ff8da5d2faf6bb774b1c03f3a52f49198cbb9898f21c91ac25aa546a4c50a'
assert hashlib.sha256((N/'prereg-v1/plan.json').read_bytes()).hexdigest()==PLAN_SHA
assert not O.exists()
E.mkdir()
shutil.copyfile(__file__,E/'execute.py')
command=[str(R/'.venv/bin/python'),'-I','-B',str(N/'candidate.py'),'--run','--plan',str(N/'prereg-v1/plan.json'),'--plan-sha256',PLAN_SHA,'--out',str(O)]
record=dict(command=command,started_at=time.time(),status='RUNNING',output=str(O),source_plan_sha256=PLAN_SHA)
(E/'invocation.json').write_text(json.dumps(record,indent=2)+'\n')
for label,cmd in [('before-memory',['/usr/bin/vm_stat']),('before-swap',['/usr/sbin/sysctl','vm.swapusage'])]:
    p=subprocess.run(cmd,capture_output=True,text=True)
    (E/(label+'.json')).write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
with (E/'stdout.txt').open('x') as stdout,(E/'stderr.txt').open('x') as stderr:
    result=subprocess.run(command,cwd=R,stdout=stdout,stderr=stderr)
record.update(ended_at=time.time(),returncode=result.returncode,status='PASS' if result.returncode==0 else 'FAIL')
(E/'final.json').write_text(json.dumps(record,indent=2)+'\n')
for label,cmd in [('after-memory',['/usr/bin/vm_stat']),('after-swap',['/usr/sbin/sysctl','vm.swapusage'])]:
    p=subprocess.run(cmd,capture_output=True,text=True)
    (E/(label+'.json')).write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
print(json.dumps(record),flush=True)
print((E/'stdout.txt').read_text(),flush=True)
print((E/'stderr.txt').read_text(),flush=True)
raise SystemExit(result.returncode)
