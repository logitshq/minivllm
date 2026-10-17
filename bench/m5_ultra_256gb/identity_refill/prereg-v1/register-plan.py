from pathlib import Path
import importlib.util, json, hashlib, sys
ROOT=Path('/Users/ryanli/code/minivllm'); HERE=ROOT/'bench/m5_ultra_256gb/identity_refill'
BASE=HERE.parent; ART=ROOT/'.artifacts/m5-ultra-256gb'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
screen=load('e028_screen_for_registration',HERE/'screen.py');owner=load('e028_owner_for_registration',HERE/'owner.py')
old=json.loads((BASE/'prefill_nax_same_split/prereg-v2/plan.json').read_text())
pins={p:h for p,h in old['execution']['pins'].items() if '/site-packages/' in p or p==str(Path(sys.executable).resolve())}
for p,h in pins.items(): assert sha(p)==h, p
pins.update(screen.source_pins())
required=list(HERE.glob('*.py'))+[HERE/'README.md']+[BASE/p for p in ('resources_v2.py','resources.py','resource_base.py','suspended_service.py','conv_tail_owned_copy/layer_owner.py','conv_tail_owned_copy/layer_screen.py','shared_w_m1/screen.py')]
for p in required:pins[str(p)]=sha(p)
inputs=ART/'identity-refill-v1-inputs'; destination=ART/'identity-refill-v1-owner'; lease=ART/'identity-refill-v1-service-lease'
assert not any(p.exists() for p in (inputs,destination,lease))
inputs.mkdir()
plan=dict(schema='E028-identity-refill-owner-plan-v1',execution=dict(python=str(ROOT/'.venv/bin/python'),screen=str(HERE/'screen.py'),pins=pins,environment=owner.checks.ENV),resource_policy=owner.checks.POLICY,contract=screen.fixed_plan(),scope='Two actual-shape layers with synthetic finite states only; complete original BF16/F32 comparison, source/output mutation and fallback checks. Same separately bounded numerical policy admits stable existing pressure2, never full-model or performance work. Default OFF, no production hook and no new Pareto point.')
planpath=inputs/'plan.json';planpath.write_text(json.dumps(plan,indent=2)+'\n')
jobs=[[str(ROOT/'.venv/bin/python'),'-I','-B',str(HERE/'owner.py'),'--plan',str(planpath),'--plan-sha256',sha(planpath),'--out',str(destination)]]
jobpath=inputs/'jobs.json';jobpath.write_text(json.dumps(jobs,indent=2)+'\n')
receipt=dict(plan_sha256=sha(planpath),jobs_sha256=sha(jobpath),pins=len(pins),contract=plan['contract'],registered_paths=dict(inputs=str(inputs),owner=str(destination),lease=str(lease)),device_run=False,performance_qualified=False,full_model_qualified=False)
(inputs/'registration.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='contract'}))
