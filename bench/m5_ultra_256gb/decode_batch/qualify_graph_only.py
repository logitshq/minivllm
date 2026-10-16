"""Cover compile_decode=True without fusion at the newly supported batch sizes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mlx.core as mx
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.engine.compiled_decode import CompiledDecode
from tutorial_code.miniengine.cache import BatchKV,BatchState,DecodeBatch,make_cache
from tutorial_code.miniengine.context import ForwardContext

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir()
w=json.loads(Path('.artifacts/m5-ultra-256gb/fullqual-build2/workload.json').read_text())
e=LLMEngine(w['model_dir'],max_seqs=8,max_len=2048,cache_limit_gb=2);m=e.model
for l in m.layers:
    if not l.is_attn:l.linear_attn.set_gates_compile(True)
fast=CompiledDecode(m,fuse_decode=False)
def arrays(cs):return [v for c in cs for v in ((c.K,c.V) if isinstance(c,BatchKV) else (c.conv,c.S))]
def copy(cs):return [BatchKV(c.K[:],c.V[:]) if isinstance(c,BatchKV) else BatchState(c.conv,c.S) for c in cs]
cs=make_cache(m);y=m(mx.array(w['prompt_ids']),ForwardContext('prefill',cs),last_only=True);mx.eval(y);rows=[]
for B in range(1,9):
    batch=DecodeBatch(m);batch.rebuild([],[(str(i),cs,2048) for i in range(B)]);mx.eval(batch.arrays());r=copy(batch.layers);c=copy(batch.layers)
    for pos in (19,255,256):
        hs=[]
        for f,state in ((m,r),(fast,c)):
            ctx=ForwardContext('decode',state,positions=mx.array([pos]*B),lens=mx.array([pos+1]*B),max_len=pos+1)
            y=f(mx.array([79852]*B),ctx);mx.eval(y,arrays(state));hs.append([y]+arrays(state))
        eq=[mx.array_equal(x.view(mx.uint16 if x.dtype==mx.bfloat16 else mx.uint32),y.view(mx.uint16 if y.dtype==mx.bfloat16 else mx.uint32)) for x,y in zip(*hs)]
        mx.eval(eq);same=[bool(x.item()) for x in eq];assert all(same),(B,pos)
        rows.append({'B':B,'position':pos,'bit_equal':same});(a.out/'checks.json').write_text(json.dumps(rows,indent=2))
    print('PASS graph without fusion',B,flush=True)
assert fast._packed is None
(a.out/'complete.json').write_text(json.dumps({'status':'PASS','paired_forwards':len(rows),'scope':'All B1..8, last optimized length and first eager length; compile enabled, fusion disabled.'},indent=2))
