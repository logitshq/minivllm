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
    e=LLMEngine(w['model_dir'],max_seqs=32,max_len=2048,compile_decode=True,fuse_decode=True)
    model=e.model;fast=e.runner.decode_model
    for l in model.layers:
        if not l.is_attn:l.linear_attn.set_gates_compile(True)
    def arrays(layers):return [x for c in layers for x in ((c.K,c.V) if isinstance(c,BatchKV) else (c.conv,c.S))]
    def copies(layers):return [BatchKV(c.K[:],c.V[:]) if isinstance(c,BatchKV) else BatchState(c.conv,c.S) for c in layers]
    def digest(x):return hashlib.sha256(np.asarray(x.astype(mx.float32)).tobytes()).hexdigest()
    caches=make_cache(model);prefill=model(mx.array(w['prompt_ids']),ForwardContext('prefill',caches),last_only=True);mx.eval(prefill)
    from mlx.utils import tree_flatten
    def parameter_signatures():
        return [(k, v.shape, str(v.dtype), hashlib.sha256(np.asarray(v.astype(mx.float32) if v.dtype==mx.bfloat16 else v).tobytes()).hexdigest()) for k,v in tree_flatten(model.parameters())]
    parameters_before = parameter_signatures()
    (a.out/'parameters-before.json').write_text(json.dumps(parameters_before,indent=2))
    rows=[]
    def check(label,ref,cand,positions,ids,extra=False):
        kwargs={'positions':mx.array(positions),'lens':mx.array([p+1 for p in positions]),'max_len':max(positions)+1}
        c1=ForwardContext('decode',ref,**kwargs);c2=ForwardContext('decode',cand,**kwargs)
        if extra:c1.record=c2.record=True
        tokens=mx.array(ids)
        l1=model(tokens,c1);mx.eval(l1,arrays(ref));l2=fast(tokens,c2);mx.eval(l2,arrays(cand))
        same=[mx.array_equal(x.view(mx.uint16 if x.dtype == mx.bfloat16 else mx.uint32), y.view(mx.uint16 if y.dtype == mx.bfloat16 else mx.uint32)) for x,y in zip([l1]+arrays(ref),[l2]+arrays(cand))]
        mx.eval(same)
        hashes1=[True]*len(same)
        hashes2=[bool(x.item()) for x in same]
        row={'case':label,'positions':positions,'reference_expected_bit_equal':hashes1,'bit_equal':hashes2,'exact':hashes1==hashes2}
        rows.append(row);(a.out/'checks.json').write_text(json.dumps(rows,indent=2));assert row['exact'],label
    for B in range(1,33):
        batch=DecodeBatch(model);batch.rebuild([],[(str(i),caches,2048) for i in range(B)]);mx.eval(batch.arrays())
        ref=copies(batch.layers);cand=copies(batch.layers)
        for step,tok in enumerate(w['continuation_ids'][:2]):
            check(f'B{B}-step{step}',ref,cand,[19+step]*B,[tok]*B)
        print('PASS',B,'2 paired steps',flush=True)
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
    check('B3-insertion-shape',copies(small),copies(small),[19]*3,[79852]*3)
    saved=model.norm.weight;model.norm.weight=saved+mx.array(0.125,saved.dtype)
    check('replaced-weight',copies(batch.layers),copies(batch.layers),[19]*4,[79852]*4)
    model.norm.weight=saved
    check('restored-weight',copies(batch.layers),copies(batch.layers),[19]*4,[79852]*4)
    # B1 packing refreshes each changed original array, including scales.
    one=DecodeBatch(model);one.rebuild([],[('one',caches,2048)]);mx.eval(one.arrays())
    for label,op in [('mlp',model.layers[0].mlp.gate_proj),
                     ('gdn',model.layers[0].linear_attn.in_proj_b),
                     ('attention',next(l.self_attn.q_proj for l in model.layers if l.is_attn))]:
        for field in ['weight','scales']:
            saved=getattr(op,field)
            changed=mx.roll(saved,1,axis=0);mx.eval(changed);setattr(op,field,changed)
            check(label+'-'+field+'-replaced',copies(one.layers),copies(one.layers),[19],[79852])
            setattr(op,field,saved)
            check(label+'-'+field+'-restored',copies(one.layers),copies(one.layers),[19],[79852])
    # Changed original weights refresh even after tracing another packed batch.
    two=DecodeBatch(model);two.rebuild([],[(str(i),caches,2048) for i in range(2)]);mx.eval(two.arrays())
    op=model.layers[0].mlp.gate_proj;saved=op.weight
    for label,weight in [('replaced',mx.roll(saved,1,axis=0)),('restored',saved)]:
        op.weight=weight;mx.eval(weight)
        check('B2-packed-weight-'+label,copies(two.layers),copies(two.layers),[19]*2,[79852]*2)
        assert fast._packed.clean_groups() == fast._packed.groups
        check('B1-after-B2-'+label,copies(one.layers),copies(one.layers),[19],[79852])
        assert fast._packed.clean_groups() == fast._packed.groups
    # Check the B1 optimized limit and its first eager position, not just B4.
    for positions in ([254],[255],[256]):
        check('B1-boundary',copies(one.layers),copies(one.layers),positions,[79852])
    # Different row positions and boundary transitions at the largest supported batch.
    large=DecodeBatch(model);large.rebuild([],[(str(i),caches,2048) for i in range(32)]);mx.eval(large.arrays())
    for positions in ([19+i*7 for i in range(32)], [255]*32, [256]*32):
        check('B32-mixed-boundary',copies(large.layers),copies(large.layers),positions,[79852+i for i in range(32)])
    over=DecodeBatch(model);over.rebuild([],[(str(i),caches,2048) for i in range(33)]);mx.eval(over.arrays())
    check('B33-fallback',copies(over.layers),copies(over.layers),[19]*33,[79852]*33)
    del large,over
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(w['model_dir'],local_files_only=True)
    for label,text in [('math','Compute 17 * 23 and explain briefly.'),
                       ('code','Write a Python function that reverses a list.'),
                       ('chinese','请解释为什么月亮会有阴晴圆缺。')]:
        prompt=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        cs=make_cache(model);logits=model(mx.array(prompt),ForwardContext('prefill',cs),last_only=True);mx.eval(logits)
        tok=int(mx.argmax(logits[0]).item())
        batch=DecodeBatch(model);batch.rebuild([],[(label,cs,2048)]);mx.eval(batch.arrays())
        ref=copies(batch.layers);cand=copies(batch.layers)
        for step in range(32):
            check(label+'-step'+str(step),ref,cand,[len(prompt)+step],[tok])
            # Diverse valid vocabulary IDs keep inputs independent of the optimization.
            tok=w['continuation_ids'][step]
        print('PASS context',label,flush=True)
    parameters_after = parameter_signatures()
    (a.out/'parameters-after.json').write_text(json.dumps(parameters_after,indent=2))
    assert parameters_after == parameters_before
    (a.out/'parameters.json').write_text(json.dumps({'unchanged':True,'count':len(parameters_after),
        'derived_bytes':sum(x.nbytes for p in fast._packed.arrays for x in p)},indent=2))
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
    def staggered(graph):
        e.runner.decode_model=fast if graph else model
        for i in range(8):
            e.add_request(w['prompt_ids'],SamplingParams(temperature=0,max_tokens=5+i),req_id='row-'+str(i))
        trace=[];step=0
        while e.has_work():
            if step==7:
                e.add_request(w['prompt_ids'],SamplingParams(temperature=0,max_tokens=5),req_id='late')
            out=e.step();mx.synchronize()
            trace.append({'outputs':out,'batch':list(e.runner.batch.req_ids),
                          'states':[digest(x) for x in e.runner.batch.arrays()]})
            step+=1
        assert not e.runner.staging and not e.runner.batch.size and e.runner._ahead is None
        return trace
    before=staggered(False);after=staggered(True)
    (a.out/'staggered.json').write_text(json.dumps({'reference':before,'candidate':after},indent=2));assert before==after
    (a.out/'complete.json').write_text(json.dumps({'status':'PASS','paired_forwards':len(rows),'state_arrays_per_forward':128,'scheduler_exact':True,'parameters_unchanged':True,'domain':'M5 Ultra MLX0.32.3 27B BF16 NVFP4, fusion B1..8 length<=256; other cases eager'},indent=2))
    print('PASS all',len(rows),flush=True)
if __name__=='__main__':main()
