"""Paired full-model logits/state checks and real scheduler transitions."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mlx.core as mx
import numpy as np
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.engine.compiled_decode import CompiledDecode
from tutorial_code.miniengine.cache import BatchKV,BatchState,DecodeBatch,make_cache
from tutorial_code.miniengine.context import ForwardContext
from tutorial_code.miniengine.sampling_params import SamplingParams

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir()
    w=json.loads(Path('.artifacts/m5-ultra-256gb/fullqual-build2/workload.json').read_text())
    e=LLMEngine(w['model_dir'],max_seqs=4,max_len=2048,compile_decode=True)
    model=e.model;fast=e.runner.decode_model
    for l in model.layers:
        if not l.is_attn:l.linear_attn.set_gates_compile(True)
    def arrays(layers):return [x for c in layers for x in ((c.K,c.V) if isinstance(c,BatchKV) else (c.conv,c.S))]
    def copies(layers):return [BatchKV(c.K[:],c.V[:]) if isinstance(c,BatchKV) else BatchState(c.conv,c.S) for c in layers]
    def digest(x):return hashlib.sha256(np.asarray(x.astype(mx.float32)).tobytes()).hexdigest()
    caches=make_cache(model);prefill=model(mx.array(w['prompt_ids']),ForwardContext('prefill',caches),last_only=True);mx.eval(prefill)
    rows=[]
    def check(label,ref,cand,positions,ids,extra=False):
        kwargs={'positions':mx.array(positions),'lens':mx.array([p+1 for p in positions]),'max_len':max(positions)+1}
        c1=ForwardContext('decode',ref,**kwargs);c2=ForwardContext('decode',cand,**kwargs)
        if extra:c1.record=c2.record=True
        tokens=mx.array(ids)
        l1=model(tokens,c1);mx.eval(l1,arrays(ref));l2=fast(tokens,c2);mx.eval(l2,arrays(cand))
        hashes1=[digest(l1)]+[digest(x) for x in arrays(ref)]
        hashes2=[digest(l2)]+[digest(x) for x in arrays(cand)]
        row={'case':label,'positions':positions,'reference_sha256':hashes1,'candidate_sha256':hashes2,'exact':hashes1==hashes2}
        rows.append(row);(a.out/'checks.json').write_text(json.dumps(rows,indent=2));assert row['exact'],label
    for B in [1,2,4]:
        batch=DecodeBatch(model);batch.rebuild([],[(str(i),caches,2048) for i in range(B)]);mx.eval(batch.arrays())
        ref=copies(batch.layers);cand=copies(batch.layers)
        for step,tok in enumerate(w['continuation_ids']):
            check(f'B{B}-step{step}',ref,cand,[19+step]*B,[tok]*B)
        print('PASS',B,'32 steps',flush=True)
    # Non-uniform histories, last optimized position, and eager fallback across
    # the boundary. Nonzero synthetic past KV detects a padded-mask mistake.
    mx.random.seed(781)
    batch=DecodeBatch(model);batch.rebuild([],[(str(i),caches,2048) for i in range(4)])
    for c in batch.layers:
        if isinstance(c,BatchKV):
            c.K=mx.random.normal(c.K.shape).astype(c.K.dtype)*0.1
            c.V=mx.random.normal(c.V.shape).astype(c.V.dtype)*0.1
    mx.eval(batch.arrays())
    ref=copies(batch.layers);cand=copies(batch.layers)
    for positions in [[19,63,127,254],[20,64,128,255],[21,65,129,256]]:
        check('mixed-boundary',ref,cand,positions,w['continuation_ids'][:4])
    check('metadata-fallback',copies(batch.layers),copies(batch.layers),[19]*4,[79852]*4,True)
    small=[BatchKV(c.K[:3],c.V[:3]) if isinstance(c,BatchKV) else BatchState(c.conv[:3],c.S[:3]) for c in batch.layers]
    check('B3-fallback',copies(small),copies(small),[19]*3,[79852]*3)
    saved=model.norm.weight;model.norm.weight=saved+mx.array(0.125,saved.dtype)
    check('replaced-weight',copies(batch.layers),copies(batch.layers),[19]*4,[79852]*4)
    model.norm.weight=saved
    check('restored-weight',copies(batch.layers),copies(batch.layers),[19]*4,[79852]*4)
    # Step-level committed state hashes include the existing asynchronous
    # lookahead; request insertions and removals must not reuse stale state.
    def schedule(graph):
        e.runner.decode_model=fast if graph else model
        e.add_request(w['prompt_ids'],SamplingParams(temperature=0,max_tokens=12),req_id='a')
        trace=[];step=0
        while e.has_work():
            if step==3:e.add_request(w['prompt_ids'],SamplingParams(temperature=0,max_tokens=5),req_id='b')
            out=e.step();mx.synchronize()
            trace.append({'outputs':out,'requests':list(e.requests),'batch':list(e.runner.batch.req_ids),
                          'states':[digest(x) for x in e.runner.batch.arrays()]})
            step+=1
        assert not e.runner.staging and not e.runner.batch.size and e.runner._ahead is None
        # Explicit stop ID, normal max-token stopping, and request reuse.
        stop=e.generate([w['prompt_ids']],SamplingParams(temperature=0,max_tokens=32,stop_token_ids=[w['continuation_ids'][3]]))
        assert len(stop[0])==4
        return {'trace':trace,'stop_output':stop}
    before=schedule(False);after=schedule(True)
    (a.out/'scheduler.json').write_text(json.dumps({'reference':before,'candidate':after},indent=2));assert before==after
    (a.out/'complete.json').write_text(json.dumps({'status':'PASS','paired_forwards':len(rows),'state_arrays_per_forward':128,'scheduler_exact':True,'domain':'M5 Ultra MLX0.32.3 27B BF16 NVFP4, B1/2/4, used length<=256; other cases eager'},indent=2))
    print('PASS all',len(rows),flush=True)
if __name__=='__main__':main()
