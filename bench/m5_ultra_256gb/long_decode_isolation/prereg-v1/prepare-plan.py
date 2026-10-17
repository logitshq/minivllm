"""CPU prepare only. Actual service prepare/run are separate reviewed root actions."""
from pathlib import Path
from types import SimpleNamespace
import hashlib, importlib.util, json, re, time
R=Path('/Users/ryanli/code/minivllm'); A=R/'.artifacts/m5-ultra-256gb'
P=R/'bench/m5_ultra_256gb/long_decode_isolation/prereg-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
owner=R/'bench/m5_ultra_256gb/long_decode_isolation/cold_owner.py'
qualifier=owner.with_name('qualify.py')
controller=R/'bench/m5_ultra_256gb/omlx_cold_restart/controller_v6.py'
auditor=controller.with_name('audit_restore_v6.py')
assert not any('PENDING' in p.read_text() for p in (owner,qualifier,controller,auditor))
spec=importlib.util.spec_from_file_location('e024_isolation_cold_owner',owner)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prior=R/'docs/experiments/m5-ultra-256gb/E024-long-decode-cold-v1'
old=R/'bench/m5_ultra_256gb/long_decode_graph/prereg-v1/owner-plan.json'
assert sha(old)==m.OWNER_PLAN_SHA
old_plan=json.loads(old.read_bytes())
q=dict(schema='E024-dispatch-isolation-plan-v1',qualifier_sha256=sha(qualifier),
       e024_plan=old_plan['qualification_plan'],deadline_s=1800)
P.mkdir(exist_ok=False);write(P/'isolation-plan.json',q)
base=m.load('../pareto_realistic/qualify_absent_owner.py')
cold=dict(journal_path=str(A/'long-decode-isolation-v1-cold-journal'),controller=ref(controller),
 approved_contract=base.CONTRACT,proposal_plan_sha256=base.PROPOSAL_SHA,
 predecessor=dict(**ref(prior/'restored-journal.json'),audit=ref(prior/'restoration-audit.json')))
plan=dict(schema='E024-isolation-cold-owner-plan-v1',execution=dict(owner=ref(owner),pins=m.source_pins()),
 diagnostic_owner_plan=ref(old),isolation_plan=ref(P/'isolation-plan.json'),cold_restart=cold,
 trials=[m.PHASE],deadline_seconds=1830)
write(P/'owner-plan.json',plan);(P/'prepare-plan.py').write_bytes(Path(__file__).read_bytes())
plan_sha=sha(P/'owner-plan.json')
m.validate_plan(SimpleNamespace(plan=P/'owner-plan.json',plan_sha256=plan_sha))
cpu=A/'long-decode-isolation-v1-cpu-prepare';cpu.mkdir(exist_ok=False);start=time.monotonic()
def check():
 if time.monotonic()-start>120:raise TimeoutError('CPU prepare bound')
command,env=m.prepare(plan,cpu/'prepared',check)
write(cpu/'prepared.json',dict(schema='E024-isolation-cpu-prepare-v1',status='PASS',plan_sha256=plan_sha,
 command=command,mlx_environment={k:v for k,v in env.items() if k.startswith('MLX_')},
 elapsed_seconds=time.monotonic()-start,gpu_executed=False,service_changed=False))
source=(R/'bench/m5_ultra_256gb/long_decode_graph/prereg-cold-v1/execute.py').read_text()
source=source.replace('long_decode_graph/prereg-cold-v1','long_decode_isolation/prereg-v1')
source=source.replace('long-decode-cold-v1','long-decode-isolation-v1').replace('controller_v5.py','controller_v6.py')
source=source.replace('long_decode_graph/cold_owner.py','long_decode_isolation/cold_owner.py')
source=re.sub(r"PLAN_SHA = '[0-9a-f]+'", "PLAN_SHA = '"+plan_sha+"'",source)
source=source.replace('unchanged E024 gates-OFF diagnostic: original 32-point matrix, parameter replacement checks and short lifecycle; first numerical difference stops; no performance run','new E024 four-arm diagnostic at B1/used257: eager and compiled at attention lengths257/1536, independent identical input state; finite numerical differences retained across all four arms; execution/resource/input mutation failures stop; no performance run')
(P/'execute.py').write_text(source)
pins={**plan['execution']['pins'],str(controller):sha(controller),str(auditor):sha(auditor),
 str(P/'execute.py'):sha(P/'execute.py'),str(P/'prepare-plan.py'):sha(P/'prepare-plan.py')}
for p in [qualifier,qualifier.with_name('test_cpu.py'),controller.with_name('test_v6_cpu.py'),controller.with_name('test_audit_restore_v6_cpu.py')]:pins[str(p)]=sha(p)
write(P/'preparation.json',dict(schema='E024-isolation-prereg-preparation-v1',status='CPU_PREPARED',
 owner_plan_sha256=plan_sha,isolation_plan_sha256=sha(P/'isolation-plan.json'),cpu_prepare=ref(cpu/'prepared.json'),
 source_pins=pins,numerical_qualified=False,performance_qualified=False))
print(json.dumps(dict(plan_sha256=plan_sha,cpu_prepare='PASS',journal=cold['journal_path'],owner_directory=str(A/'long-decode-isolation-v1-owner'))))
