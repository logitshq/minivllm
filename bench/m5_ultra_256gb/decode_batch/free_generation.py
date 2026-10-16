"""Same-batch eager/optimized generation, including the larger eager fallback."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mlx.core as mx
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.engine.compiled_decode import CompiledDecode
from tutorial_code.miniengine.sampling_params import SamplingParams

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir()
w=json.loads(Path('.artifacts/m5-ultra-256gb/fullqual-build2/workload.json').read_text())
e=LLMEngine(w['model_dir'],max_seqs=64,max_len=2048,cache_limit_gb=2)
for layer in e.model.layers:
    if not layer.is_attn:layer.linear_attn.set_gates_compile(True)
fast=CompiledDecode(e.model,fuse_decode=True);results=[]
for B in (1,2,3,4,5,6,7,8,16,32,64):
    ids=[]
    for f in (e.model,fast):
        e.runner.decode_model=f
        ids.append(e.generate([w['prompt_ids']]*B,SamplingParams(temperature=0,max_tokens=32)))
    assert ids[0]==ids[1],B
    assert all(len(row)==32 for row in ids[0]),(B,'normal EOS shorter than benchmark token budget')
    results.append({'B':B,'reference_ids':ids[0],'candidate_ids':ids[1],'exact':True})
    (a.out/'progress.json').write_text(json.dumps(results,indent=2));print('PASS free generation',B,flush=True)
(a.out/'complete.json').write_text(json.dumps({'status':'PASS','results':results,
    'scope':'Raw IDs match eager at the same batch/schedule, greedy normal EOS; batch sizes are not assumed numerically equivalent.'},indent=2))
