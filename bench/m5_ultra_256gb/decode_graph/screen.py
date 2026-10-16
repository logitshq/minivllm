"""Exploratory factorial screen, not a fresh-process adoption test."""
import argparse, json, os, statistics, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mlx.core as mx
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.engine.compiled_decode import CompiledDecode
from tutorial_code.miniengine.sampling_params import SamplingParams

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir()
    w=json.loads(Path('.artifacts/m5-ultra-256gb/fullqual-build2/workload.json').read_text())
    engine=LLMEngine(w['model_dir'],max_seqs=4,max_len=2048)
    compiled=CompiledDecode(engine.model)
    configurations=[('old_v19',False,True,0),('eager_pool0',False,False,0),('graph_pool0',True,False,0),
                    ('eager_pool2',False,False,2),('graph_pool2',True,False,2),('graph_v19_pool2',True,True,2)]
    results=[]
    for name,graph,norm,pool in configurations+list(reversed(configurations)):
        mx.synchronize();mx.clear_cache();mx.set_cache_limit(pool<<30)
        engine.runner.decode_model=compiled if graph else engine.model
        for layer in engine.model.layers:
            if not layer.is_attn:
                layer.linear_attn.set_gates_compile(True);layer.linear_attn.set_norm_reuse(norm)
        runs=[]
        for index in range(6):
            mx.synchronize();start=time.perf_counter()
            ids=engine.generate([w['prompt_ids']],SamplingParams(temperature=0,max_tokens=32))[0]
            mx.synchronize();elapsed=time.perf_counter()-start
            assert ids==w['continuation_ids']
            assert not engine.runner.batch.size and not engine.runner.staging
            runs.append({'elapsed_s':elapsed,'tokens':len(ids),'tok_s':len(ids)/elapsed,'output_ids':ids})
        row={'name':name,'graph':graph,'norm_reuse':norm,'allocator_pool_gib':pool,'warm':runs[:3],'measure':runs[3:],'median':statistics.median(r['tok_s'] for r in runs[3:])}
        results.append(row);print(name,round(row['median'],3),flush=True)
        (a.out/'screen.json').write_text(json.dumps({'environment':{k:v for k,v in os.environ.items() if k.startswith('MLX_')},'rows':results},indent=2))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
