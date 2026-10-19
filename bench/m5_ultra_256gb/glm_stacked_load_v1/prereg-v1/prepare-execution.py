"""Actual read-only controller prepare and source seal before the GLM cycle."""
import json
from pathlib import Path
import shutil
import subprocess
import time
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';B=R/'bench/m5_ultra_256gb/glm_stacked_load_v1'
out=B/'prereg-v1';out.mkdir()
shutil.copyfile(__file__,out/'prepare-execution.py')
reg=A/'glm-stacked-registration-v1';commands=json.loads((reg/'commands.json').read_text())
events=[]
for name,command in [('controller-prepare',commands['controller_prepare']),('seal',[str(R/'.venv/bin/python'),'-I','-B',str(B/'preregister.py'),'--seal-prepared',str(reg)])]:
    event=dict(name=name,command=command,started_at=time.time())
    with (out/(name+'-stdout.txt')).open('x') as stdout,(out/(name+'-stderr.txt')).open('x') as stderr:
        p=subprocess.run(command,cwd=R,stdout=stdout,stderr=stderr)
    event.update(returncode=p.returncode,ended_at=time.time());events.append(event)
    (out/(name+'-command.json')).write_text(json.dumps(event,indent=2)+'\n')
    print(json.dumps(event),flush=True)
    if p.returncode:raise SystemExit(p.returncode)
for name,source in [('cpu',A/'glm-stacked-cpu-v1'),('input-registration',A/'glm-stacked-input-registration-attempt-v1'),('registration',reg),('inputs',A/'glm-stacked-inputs-v1')]:
    shutil.copytree(source,out/name)
shutil.copyfile(A/'glm-stacked-cold-journal-v1/journal.json',out/'prepared-journal.json')
(out/'preparation.json').write_text(json.dumps(dict(status='PREPARED_AND_SEALED',events=events,service_mutated=False),indent=2)+'\n')
