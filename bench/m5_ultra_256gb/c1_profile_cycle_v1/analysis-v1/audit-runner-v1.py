"""Run the two actual saved offline audits after the existing driver returned."""
import datetime
import json
from pathlib import Path
import subprocess
import time

R=Path('/Users/ryanli/code/minivllm')
N=R/'bench/m5_ultra_256gb/c1_profile_cycle_v1'
A=R/'.artifacts/m5-ultra-256gb'
final=json.loads((A/'c1-profile-cycle-v1-execution/final.json').read_text())
journal=json.loads((A/'c1-profile-cycle-v1-journal/journal.json').read_text())
assert final['journal_state']==journal['state']=='RESTORED', 'actual restoration before offline audit'
commands=json.loads((N/'prereg-v1/commands.json').read_text())
out=N/'analysis-v1'
out.mkdir(exist_ok=True)
events=[]
for name in ('restoration_audit','closed_audit'):
    command=commands[name]
    started=time.time()
    with (out/(name+'-stdout.txt')).open('x') as stdout, (out/(name+'-stderr.txt')).open('x') as stderr:
        result=subprocess.run(command,cwd=R,stdout=stdout,stderr=stderr)
    event=dict(name=name,command=command,started_at=started,ended_at=time.time(),returncode=result.returncode)
    events.append(event)
    with (out/(name+'-command.json')).open('x') as stream:json.dump(event,stream,indent=2);stream.write('\n')
    print(json.dumps(event),flush=True)
with (out/'audit-runner-v1.json').open('x') as stream:json.dump(dict(events=events,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),stream,indent=2);stream.write('\n')
raise SystemExit(0 if all(x['returncode']==0 for x in events) else 1)
