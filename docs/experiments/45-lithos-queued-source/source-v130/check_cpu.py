"""V130 minimal CPU source/parser assertions only; no HTTP/model imports."""
import ast,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent
for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
t=ast.parse((D/'client.py').read_text());ns={}
const=next(x for x in t.body if isinstance(x,ast.Assign));f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='body')
exec(compile(ast.Module(body=[const,f],type_ignores=[]),'actual_body','exec'),ns)
a,b=ns['body']('model','openai'),ns['body']('model','ollama')
assert a['messages']==b['messages'] and a['temperature']==b['options']['temperature']==0 and a['top_p']==b['options']['top_p']==1
assert a['max_completion_tokens']==b['options']['num_predict']==32
assert a['stream'] and b['stream'] and not b['think'] and 'keep_alive' not in b
assert b['options']['repeat_penalty']==1 and b['options']['seed']==b['options']['top_k']==0
s=(D/'client.py').read_text();assert 'barrier.wait();start=time.perf_counter()' in s
assert "value.get('done') is True" in s and "payload=='[DONE]'" in s and "'retry_count':0" in s
assert "status!=200" in s and "'latency_p50_ms':1000*statistics.median" in s and "if all_ok else None" in s
q=(D/'qualify.py').read_text();assert "choices=['AC','Battery'],required=True" in q and "official.official_command('lithos',1" in q
assert "control.cleanup_owned(proc,birth,O/'cleanup-signal-identities.json')" in q
assert "control.wait_port_absence(port)" in q and "metrics)==expected" in q and "x['cached_prompt_tokens']==0" in q
assert "namespace" not in s
assert not any(x=='mlx' or x.startswith('mlx.') for x in sys.modules)
print(json.dumps({'status':'PASS_CPU_V130_SOURCE_AND_WIRE_PLAN','MLX_imported':False,'GPU_used':False,'qualification_HTTP_requests':8,'qualification_outputs':256,'fixed_power_required':True,'automatic_retries':0,'new_custom_queue':False}))
