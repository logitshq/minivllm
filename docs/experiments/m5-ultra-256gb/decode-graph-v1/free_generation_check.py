import json,sys
from pathlib import Path
sys.path.insert(0,'/Users/ryanli/code/minivllm')
import mlx.core as mx
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.engine.compiled_decode import CompiledDecode
from tutorial_code.miniengine.sampling_params import SamplingParams
b=Path('.artifacts/m5-ultra-256gb');w=json.loads((b/'fullqual-build2/workload.json').read_text())
out=b/'graph-free-generation-v1';out.mkdir()
engine=LLMEngine(w['model_dir'],max_seqs=4,max_len=2048)
for l in engine.model.layers:
 if not l.is_attn:l.linear_attn.set_gates_compile(True)
fast=CompiledDecode(engine.model);results=[]
for B in (1,2,4):
 ids=[]
 for implementation in (engine.model,fast):
  engine.runner.decode_model=implementation
  ids.append(engine.generate([w['prompt_ids']]*B,SamplingParams(temperature=0,max_tokens=32)))
 assert ids[0]==ids[1]
 observed={tuple(r['output_ids_when_available']) for p in (b/'graph-frontier-v1').glob(f'C{B}-pair*-graph/result.json') for phase in json.loads(p.read_text())['groups'].values() for g in phase for r in g['requests']}
 assert observed==set(tuple(row) for row in ids[0])
 results.append({'B':B,'reference_ids':ids[0],'candidate_ids':ids[1],'matches_all_native_HTTP_outputs_at_this_C':True})
 print('PASS',B,flush=True)
mx.synchronize()
(out/'complete.json').write_text(json.dumps({'status':'PASS','results':results,'scope':'Compare raw outputs to eager at the same batch. MLX QMM reduction differs between B1 and B2/B4, so cross-batch token identity is not assumed.'},indent=2))
