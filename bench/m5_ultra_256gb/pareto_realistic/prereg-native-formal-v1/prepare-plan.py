from pathlib import Path
import importlib.util,hashlib,json
R=Path('/Users/ryanli/code/minivllm');P=R/'bench/m5_ultra_256gb/pareto_realistic/prereg-native-formal-v1';A=R/'.artifacts/m5-ultra-256gb'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
owner=R/'bench/m5_ultra_256gb/pareto_realistic/native_formal_owner.py';controller=R/'bench/m5_ultra_256gb/omlx_cold_restart/controller_v4.py'
assert sha(owner)=='e33faa8f211e905f40daabd00941f3fd016041c9705c6a7a4a2fe805fb0aceff'
assert sha(controller)=='836a4c16534521441da8021df3f963210d885da867f79a33d86a3b6dd01101c0'
s=importlib.util.spec_from_file_location('formal_owner',owner);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
original=json.loads((R/'bench/m5_ultra_256gb/pareto_realistic/prereg-http-baseline-v2/owner-plan.json').read_bytes())
prior=R/'docs/experiments/m5-ultra-256gb/E023-http-baseline-v2'
cold=dict(journal_path=str(A/'native-formal-v1-cold-journal'),controller=ref(controller),approved_contract=original['cold_restart']['approved_contract'],proposal_plan_sha256=original['cold_restart']['proposal_plan_sha256'],predecessor=dict(**ref(prior/'restored-journal.json'),audit=ref(prior/'restoration-audit.json')))
qual={}
for C in (1,8):
 child=A/f'http-baseline-v2-owner/phases/c{C}-replay/qualification'
 qual[str(C)]=dict(audit=ref(A/f'http-baseline-v2-analysis/c{C}-independent-audit-v2.json'),replay_plan=ref(child/'plan.json'),child_directory=str(child))
http=dict(transport=original['http']['transport'],environment=original['http']['environment'],deadlines=dict(ready=120,request=600,warm=720,formal=2400,close=10,term=15,kill=15,overall=3300,resource_interval=5))
plan=m.plan_template(original['qualification'],cold,http,qual,trials=['r1-c1','r1-c8'])
P.mkdir(exist_ok=False);write(P/'owner-plan.json',plan)
(P/'prepare-plan.py').write_bytes(Path(__file__).read_bytes())
write(P/'cpu-receipt.json',dict(schema='E023-native-formal-root-cpu-v1',status='PASS',command='.venv/bin/python -I -B -m unittest bench.m5_ultra_256gb.omlx_cold_restart.test_v4_cpu bench.m5_ultra_256gb.pareto_realistic.test_native_formal_owner_cpu -q',tests=41,elapsed_seconds=.364,exit_code=0,tool_chunk='cda08e',source_pins={str(p):sha(p) for p in (owner,controller,R/'bench/m5_ultra_256gb/pareto_realistic/test_native_formal_owner_cpu.py',R/'bench/m5_ultra_256gb/omlx_cold_restart/test_v4_cpu.py')}))
print(json.dumps(dict(plan_sha256=sha(P/'owner-plan.json'),owner=str(A/'native-formal-v1-owner'),journal=cold['journal_path'],trials=plan['trials'],deadline_seconds=plan['deadline_seconds'])))
