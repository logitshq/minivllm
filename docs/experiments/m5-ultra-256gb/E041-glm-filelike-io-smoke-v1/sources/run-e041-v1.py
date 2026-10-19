import hashlib, json, subprocess, time
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm')
base=R/'.artifacts/m5-ultra-256gb'
logs=base/'glm-filelike-io-smoke-execution-v1'
logs.mkdir()
script=R/'bench/m5_ultra_256gb/glm_filelike_io_smoke_v1/smoke.py'
python=R/'.artifacts/omlx-baseline/.venv/bin/python'
commands=[('default',[str(python),'-I','-B',str(script)]),('actual',[str(python),'-I','-B',str(script),'--run','--out',str(base/'glm-filelike-io-smoke-v1')])]
receipts=[]
for name,command in commands:
    started=time.time(); result=subprocess.run(command,cwd=R,capture_output=True)
    (logs/(name+'.stdout')).write_bytes(result.stdout); (logs/(name+'.stderr')).write_bytes(result.stderr)
    receipts.append(dict(name=name,command=command,returncode=result.returncode,started_unix=started,elapsed_seconds=time.time()-started))
    (logs/'execution.json').write_text(json.dumps(dict(attempts=receipts,script_sha256=hashlib.sha256(script.read_bytes()).hexdigest()),indent=2)+'\n')
    print(name,result.returncode,result.stdout.decode(errors='replace'),result.stderr.decode(errors='replace'),flush=True)
print('execution',str(logs),flush=True)
raise SystemExit(0 if all(r['returncode']==0 for r in receipts) else 1)
