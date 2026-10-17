"""E013: synchronized component costs, plus uninstrumented full draft.

Named costs.py to avoid shadowing the Python standard-library profile module.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench/m5_ultra_256gb/draft_selector'))
from screen import bits_equal, file_hash, write
import mlx.core as mx
from transformers import AutoTokenizer
from tutorial_code.miniengine.cache import make_cache
from tutorial_code.miniengine.context import ForwardContext
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.spec.dflash2 import DFlash2Drafter, load_drafter


def measure(fn):
    mx.synchronize()
    t0 = time.perf_counter_ns()
    result = fn()
    t1 = time.perf_counter_ns()
    mx.eval(result)
    mx.synchronize()
    t2 = time.perf_counter_ns()
    return {'call_ms': (t1-t0)/1e6, 'eval_wait_ms': (t2-t1)/1e6, 'total_ms': (t2-t0)/1e6}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workload', type=Path, required=True)
    p.add_argument('--drafter', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args(); a.out.mkdir()
    rows=[]
    try:
        sources=sorted((ROOT/'tutorial_code/miniengine').rglob('*.py'))+[Path(__file__),ROOT/'bench/m5_ultra_256gb/draft_selector/screen.py',ROOT/'bench/m5_ultra_256gb/draft_selector/candidate.py']
        pins={str(f.relative_to(ROOT)):file_hash(f) for f in sources}
        w=json.loads(a.workload.read_text())
        write(a.out/'plan.json',{'experiment':'E013','head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_sha256':pins,'workload_sha256':file_hash(a.workload),'drafter_weights_sha256':file_hash(a.drafter/'model.safetensors'),
            'drafter_config_sha256':file_hash(a.drafter/'config.json'),'history_lengths':[32,256,1024],'blocks':[3,5,8],
            'seed':'Original 19-token benchmark prefix, followed by repeated fixed prose; target prefill supplies historical features only',
            'selector':'E012 ON in all calls','timing':'One first timed call, two warm rounds, seven alternating forward/reverse component sweeps',
            'decision':'Rank module wall costs; head >=40% of full selects head path investigation; otherwise largest component. Component times are not additive and not HTTP.',
            'scope':'Synchronized component profile plus original uninstrumented full draft; no generation/acceptance claim'})
        mx.set_cache_limit(2*1024**3)
        engine=LLMEngine(w['model_dir'],max_seqs=1,max_len=2048,cache_limit_gb=2)
        model=engine.model; draft=load_drafter(str(a.drafter),device_selection=True)
        tok=AutoTokenizer.from_pretrained(w['model_dir'],local_files_only=True)
        filler=tok.encode(' Apple silicon inference measures model computation and memory traffic. ',add_special_tokens=False)
        original=DFlash2Drafter._select
        for length in (32,256,1024):
            ids=(w['prompt_ids']+filler*((length//len(filler))+1))[:length]
            caches=make_cache(model)
            logits,feats=model(mx.array(ids),ForwardContext('prefill',caches),last_only=True,capture=tuple(draft.cfg.target_layer_ids))
            mx.eval(logits,feats); anchor=int(mx.argmax(logits[0]).item())
            cut=length-4
            draft.reset(); draft.draft(feats[:cut],ids[cut],cut,model.embed_tokens,model.lm_head_fn,3)
            seed=[(l.self_attn.ctx_k,l.self_attn.ctx_v,l.self_attn.ctx_len) for l in draft.layers]
            mx.eval([v for k,v,n in seed]+[k for k,v,n in seed])
            def restore():
                for l,(k,v,n) in zip(draft.layers,seed):
                    l.self_attn.ctx_k,l.self_attn.ctx_v,l.self_attn.ctx_len=k,v,n
            def kv():
                return [x for l in draft.layers for x in (l.self_attn.ctx_k,l.self_attn.ctx_v)]
            for block in (3,5,8):
                captured=[]
                def observe(self,h,logits,anchor):
                    captured.append((h,logits))
                    return original(self,h,logits,anchor)
                restore()
                with patch.object(DFlash2Drafter,'_select',observe):
                    old=draft.draft(feats[cut:],anchor,length,model.embed_tokens,model.lm_head_fn,block)
                oldkv=kv();mx.eval(oldkv)
                # Build the same operations once to freeze materialized stage inputs.
                restore(); ctx=draft.hidden_norm(draft.fc(feats[cut:]))
                embedded=model.embed_tokens(mx.array([anchor]+[draft.cfg.mask_token_id]*(block-1)))
                def backbone():
                    restore();x=embedded
                    for layer in draft.layers:x=layer(x,ctx,length)
                    return x,kv()
                hidden,newkv=backbone(); h=draft.norm(hidden)[1:]; output=model.lm_head_fn(h).astype(mx.float32)
                selected=original(draft,h,output,anchor)
                same=bits_equal([captured[0][0],captured[0][1]]+oldkv,[h,output]+newkv)
                assert all(same) and old==selected
                mx.eval(ctx,embedded,hidden,h,output,newkv)
                def full():
                    restore()
                    return draft.draft(feats[cut:],anchor,length,model.embed_tokens,model.lm_head_fn,block)
                components={
                    'context_projection':lambda:draft.hidden_norm(draft.fc(feats[cut:])),
                    'embedding':lambda:model.embed_tokens(mx.array([anchor]+[draft.cfg.mask_token_id]*(block-1))),
                    'backbone':backbone,'final_norm':lambda:draft.norm(hidden)[1:],
                    'lm_head':lambda:model.lm_head_fn(h).astype(mx.float32),
                    'selector':lambda:original(draft,h,output,anchor),'full_draft':full}
                first={name:measure(fn) for name,fn in components.items()}
                for _ in range(2):
                    for fn in components.values():measure(fn)
                samples=[]
                for repeat in range(7):
                    names=list(components)
                    if repeat%2:names.reverse()
                    for name in names:samples.append({'repeat':repeat,'component':name,**measure(components[name])})
                med={name:statistics.median(s['total_ms'] for s in samples if s['component']==name) for name in components}
                rows.append({'history_length':length,'block':block,'prompt_ids':ids,'ids':old,'hidden_logits_kv_bit_equal':same,
                    'first_timed':first,'samples':samples,'median_ms':med,'head_fraction_of_full':med['lm_head']/med['full_draft']})
                write(a.out/'profile.json',rows)
                print('PASS',length,block,json.dumps(med),flush=True)
        assert all(file_hash(ROOT/f)==h for f,h in pins.items())
        write(a.out/'complete.json',{'status':'PASS','cases':len(rows),'scope':'Profile, not optimization or end-to-end speedup',
            'peak_active_gib':mx.get_peak_memory()/1024**3})
    except BaseException:
        write(a.out/'failure.json',{'status':'STOP','completed_cases':len(rows),'error':traceback.format_exc()})
        raise


if __name__=='__main__':main()
