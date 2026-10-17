import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
P=Path('/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/lithos_d512_formal/execute.py')
sp=importlib.util.spec_from_file_location('cpu_formal_driver',P);m=importlib.util.module_from_spec(sp)
with patch('subprocess.run',side_effect=AssertionError('no real backend')),patch('subprocess.Popen',side_effect=AssertionError('no real backend')):
 sp.loader.exec_module(m)
 rows=[]
 for mode in ('success','owner-fail','owner-exception','stop-reject','bad-pin'):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);p=r/'prereg';j=r/'journal';d=r/'execution';o=r/'owner';p.mkdir();j.mkdir()
   c=r/'controller.py';owner=r/'owner.py';c.write_bytes(b'CPU controller');owner.write_bytes(b'CPU owner');(p/'owner-plan.json').write_bytes(b'{}\n');(j/'journal.json').write_text('{"state":"PREPARED"}')
   pins={str(x):m.sha(x) for x in (P,c,owner,p/'owner-plan.json')}
   if mode=='bad-pin':pins[str(c)]='0'*64
   receipt=dict(source_pins=pins,owner_plan_sha256=m.sha(p/'owner-plan.json'),prepared_journal_sha256=m.sha(j/'journal.json'))
   (p/'receipt.json').write_text(json.dumps(receipt));calls=[]
   def fake(name,command):
    calls.append((name,command))
    if name=='stop':
     if mode=='stop-reject':return 1
     (j/'journal.json').write_text('{"state":"STOPPED"}');return 0
    if name=='lithos-formal-owner':
     o.mkdir()
     if mode=='owner-exception':raise OSError('CPU simulated driver exception')
     return 1 if mode=='owner-fail' else 0
    assert name=='recover' and command[-2:]==['--watchdog-seconds','3600']
    assert '--no-experiment' not in command
    (j/'journal.json').write_text('{"state":"RESTORED"}');return 0
   kw=dict(R=r,P=p,J=j,D=d,O=o,C=c,OWNER=owner,PY='CPU-python',A=['CPU-bound-contract'],run=fake)
   with patch.multiple(m,**kw),contextlib.redirect_stdout(io.StringIO()):
    try:result=m.main()
    except Exception as exc:result=type(exc).__name__
   expected={'success':0,'owner-fail':1,'owner-exception':'OSError','stop-reject':1,'bad-pin':'ValueError'}[mode]
   assert result==expected,(mode,result)
   assert [x[0] for x in calls]==([] if mode=='bad-pin' else ['stop'] if mode=='stop-reject' else ['stop','lithos-formal-owner','recover'])
   if mode!='bad-pin':
    final=json.loads((d/'final.json').read_bytes())
    assert final['journal_state']==('PREPARED' if mode=='stop-reject' else 'RESTORED')
   rows.append(dict(case=mode,status='PASS',result=result,phases=[x[0] for x in calls]))
out=Path('/tmp/lithos-formal-execute-cpu.json');out.write_text(json.dumps(dict(status='PASS',scope='CPU fake driver only; no subprocess/service/network',source_sha256=m.sha(P),tests=rows),indent=2)+'\n')
print(out)
print(json.dumps(rows))
