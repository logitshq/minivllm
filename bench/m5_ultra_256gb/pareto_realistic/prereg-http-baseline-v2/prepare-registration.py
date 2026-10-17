from pathlib import Path
import hashlib, json, importlib.util
R=Path('/Users/ryanli/code/minivllm'); B=R/'bench/m5_ultra_256gb/pareto_realistic'
P=B/'prereg-http-baseline-v2'; P.mkdir(exist_ok=True)
A=R/'docs/experiments/m5-ultra-256gb/E023-http-baseline-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
spec=importlib.util.spec_from_file_location('http_owner_v2',B/'http_owner_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=json.loads((B/'prereg-http-baseline-v1/owner-plan.json').read_bytes())
cold=old['cold_restart'];cold['journal_path']=str(R/'.artifacts/m5-ultra-256gb/http-baseline-v2-cold-journal')
cold['predecessor']={'path':str(A/'restored-journal.json'),'sha256':sha(A/'restored-journal.json'),'audit':{'path':str(A/'restoration-audit.json'),'sha256':sha(A/'restoration-audit.json')}}
plan=m.plan_template(old['qualification'],cold,old['http']);write(P/'owner-plan.json',plan)
plan_sha=sha(P/'owner-plan.json')
wrapper=(B/'prereg-http-baseline-v1/execute.py').read_text().replace('http-baseline-v1','http-baseline-v2').replace('http_owner.py','http_owner_v2.py').replace('d641a95b7a454ce45ea01c3f3b3b0bc59f08ec71de0f0d5c79973315f7861429',plan_sha)
with (P/'execute.py').open('x') as f:f.write(wrapper)
text=(B/'prereg-http-baseline-v1/README.md').read_text().replace('replay v1','replay v2').replace('PID74456','PID81337').replace('immutable committed E028','immutable archived failed HTTP v1').replace('http-baseline-v1-*','http-baseline-v2-*')
text+='\n\n## Narrow retry of the failed readiness check\n\nThe first capture stopped before any HTTP request because macOS lsof -Fpn also\nreturns the always-selected file descriptor field. Its exact p81230/f4/localhost\nreceipt is retained as a CPU fixture. V2 permits one optional ASCII decimal fd\nfield between the exact PID and endpoint, retaining every ownership and cleanup\ncheck. The owner changes only the capture source path and SHA. No model, scheduler,\nworkload, environment, resource limit or numerical contract changes. All v1\nsources, raw failure and independently audited restoration are preserved.\nRoot 28 CPU tests PASS; independent source review found no remaining blocker.\nThe new predecessor is the v1 RESTORED PID81337 plus its separate restoration\naudit, which does not turn the failed experiment into numerical success.\n'
with (P/'README.md').open('x') as f:f.write(text)
write(P/'root-cpu-tests.json',{'command':'.venv/bin/python -B -m unittest bench.m5_ultra_256gb.pareto_realistic.test_http_capture_v2_cpu bench.m5_ultra_256gb.pareto_realistic.test_http_owner_v2_cpu -v','tool_chunk_id':'d8b9ad','exit_code':0,'tests':28,'unittest_seconds':0.287,'scope':'Root actual CPU tests; no GPU/service execution','source_pins':{str(B/n):sha(B/n) for n in ['http_capture_v2.py','http_owner_v2.py','test_http_capture_v2_cpu.py','test_http_owner_v2_cpu.py','http_capture_v2_listener_fixture.json']}})
print(json.dumps({'plan_sha256':plan_sha,'predecessor':cold['predecessor']}))
