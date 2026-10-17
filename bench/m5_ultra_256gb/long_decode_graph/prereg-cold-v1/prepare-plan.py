"""Prepare the unchanged E024 diagnostic against the audited current cold predecessor."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import re
import time

R=Path('/Users/ryanli/code/minivllm')
A=R/'.artifacts/m5-ultra-256gb'
P=R/'bench/m5_ultra_256gb/long_decode_graph/prereg-cold-v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(path),sha256=sha(path))
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
owner=R/'bench/m5_ultra_256gb/long_decode_graph/cold_owner.py'
controller=R/'bench/m5_ultra_256gb/omlx_cold_restart/controller_v5.py'
auditor=R/'bench/m5_ultra_256gb/omlx_cold_restart/audit_restore_v5.py'
assert sha(owner)=='49e4ef7396a81c12c167b8a33075725c0d2af679d91d652b204fb408ddfda07b'
assert sha(controller)=='6f61d28450d9c44401c2e5a4e6f8945ce82ec9430f56456476c7eda0cd4cd578'
assert sha(auditor)=='9039af520fc095982d85d07cf1712010cec62f672786dc2b04f871db08891ee3'
spec=importlib.util.spec_from_file_location('e024_cold_owner',owner)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
prior=R/'docs/experiments/m5-ultra-256gb/E023-native-formal-v1'
old=R/'bench/m5_ultra_256gb/long_decode_graph/prereg-v1/owner-plan.json'
assert sha(old)==module.OWNER_PLAN_SHA
base=module.load('../pareto_realistic/qualify_absent_owner.py')
cold=dict(journal_path=str(A/'long-decode-cold-v1-cold-journal'),controller=ref(controller),
    approved_contract=base.CONTRACT,proposal_plan_sha256=base.PROPOSAL_SHA,
    predecessor=dict(**ref(prior/'restored-journal.json'),audit=ref(prior/'restoration-audit.json')))
plan=dict(schema='E024-cold-owner-plan-v1',execution=dict(owner=ref(owner),pins=module.source_pins()),
    diagnostic_owner_plan=ref(old),cold_restart=cold,trials=[module.PHASE],deadline_seconds=1830)
P.mkdir(exist_ok=False);write(P/'owner-plan.json',plan)
(P/'prepare-plan.py').write_bytes(Path(__file__).read_bytes())
plan_sha=sha(P/'owner-plan.json')
module.validate_plan(SimpleNamespace(plan=P/'owner-plan.json',plan_sha256=plan_sha))
cpu=A/'long-decode-cold-v1-cpu-prepare';cpu.mkdir(exist_ok=False)
started=time.monotonic()
def check():
    if time.monotonic()-started>120:raise TimeoutError('CPU prepare bound')
command,env=module.prepare(plan,cpu/'prepared',check)
write(cpu/'prepared.json',dict(schema='E024-cold-cpu-prepare-v1',status='PASS',plan_sha256=plan_sha,
    command=command,mlx_environment={k:v for k,v in env.items() if k.startswith('MLX_')},
    elapsed_seconds=time.monotonic()-started,gpu_executed=False,service_changed=False))
source=(R/'bench/m5_ultra_256gb/pareto_realistic/prereg-native-formal-v1/execute.py').read_text()
source=source.replace('frozen authorized formal timing cycle','frozen authorized E024 diagnostic cycle')
source=source.replace('pareto_realistic/prereg-native-formal-v1','long_decode_graph/prereg-cold-v1')
source=source.replace('native-formal-v1','long-decode-cold-v1')
source=source.replace('controller_v4.py','controller_v5.py')
source=re.sub(r"PLAN_SHA = '[0-9a-f]+'", "PLAN_SHA = '"+plan_sha+"'",source)
source=source.replace('pareto_realistic/native_formal_owner.py','long_decode_graph/cold_owner.py')
source=source.replace("run('formal-owner'", "run('diagnostic-owner'")
source=source.replace('fixed C1/C8 observer-OFF formal timing, eight warm and 64 formal requests each',
    'unchanged E024 gates-OFF diagnostic: original 32-point matrix, parameter replacement checks and short lifecycle; first numerical difference stops; no performance run')
(P/'execute.py').write_text(source)
write(P/'preparation.json',dict(schema='E024-cold-prereg-preparation-v1',status='CPU_PREPARED',
    owner_plan_sha256=plan_sha,cpu_prepare=ref(cpu/'prepared.json'),
    source_pins={**plan['execution']['pins'],str(controller):sha(controller),str(auditor):sha(auditor),
                 str(P/'execute.py'):sha(P/'execute.py'),str(P/'prepare-plan.py'):sha(P/'prepare-plan.py')},
    original_candidate_plan_sha256=module.QUALIFICATION_SHA,
    numerical_qualified=False,performance_qualified=False))
print(json.dumps(dict(plan_sha256=plan_sha,cpu_prepare='PASS',journal=cold['journal_path'],
    owner_directory=str(A/'long-decode-cold-v1-owner'),trials=plan['trials'])))
