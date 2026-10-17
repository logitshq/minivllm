"""Explicit root-only V110 leaf/interface diagnostic, not engine/spec qualification."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import sys
import traceback

D=Path(__file__).resolve().parent
P=D.parent
ROOT=P/'candidate-v110-external-mtp-build1-root'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def identity(expected):
    assert sha(D/'source-pins.json')==expected
    pins=json.loads((D/'source-pins.json').read_text())
    for path,digest in pins.items():assert sha(path)==digest,path
    schema=json.loads((D/'checkpoint-schema.json').read_text())
    f=Path(schema['draft_dir'])/'model.safetensors'
    with f.open('rb') as stream:
        prefix=stream.read(8);n=struct.unpack('<Q',prefix)[0];header=stream.read(n)
    assert f.stat().st_size==schema['model_file_bytes']
    assert hashlib.sha256(prefix+header).hexdigest()==schema['prefix_header_sha256']
    assert n==schema['header_bytes']
    return pins,schema


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    switches=ap.add_mutually_exclusive_group(required=True)
    switches.add_argument('--inert',action='store_true');switches.add_argument('--run',action='store_true')
    ap.add_argument('--mode',choices=('draft-leaf','target-interface'),required=True)
    ap.add_argument('--pins-sha256',required=True);ap.add_argument('--out',type=Path)
    args=ap.parse_args();pins,schema=identity(args.pins_sha256)
    assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
    if args.inert:
        print(json.dumps({'status':'CPU_INERT_ONLY','mode':args.mode,'GPU_used':False,'payload_read':False,'source_count':len(pins)}));return
    assert args.out and args.out.is_absolute();args.out.mkdir()
    write(args.out/'started.json',{'pid':os.getpid(),'ppid':os.getppid(),'mode':args.mode,'pins':args.pins_sha256})
    controls={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
    assert {k:os.environ.get(k) for k in controls}==controls
    assert not any(k.startswith(('DYLD_','OMLX_','METAL_')) for k in os.environ)
    assert {name:importlib.metadata.version(name) for name in ('mlx','mlx-metal')}=={'mlx':'0.32.3','mlx-metal':'0.32.3'}
    sys.path.insert(0,str(P/'m4-final-native15-official-attempt2'));import common as guard
    samples=[]
    def resource(label):
        guard.probe(samples,args.out/'resources.json',label,'AC',os.getpid())
        assert samples[-1]['processinfo_thermal_metadata']['code']==0
    resource('before-MLX')
    try:
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        assert mx.default_device()==mx.gpu
        device=mx.device_info();assert device['device_name']=='Apple M4 Max' and device['architecture']=='applegpu_g16s'
        mx.set_memory_limit((2 if args.mode=='draft-leaf' else 28)<<30)
        mx.set_cache_limit((64<<20) if args.mode=='draft-leaf' else (2<<30))
        sys.path.insert(0,str(ROOT))
        from tutorial_code.miniengine.config import ModelConfig
        from tutorial_code.miniengine.spec.external_mtp import load_draft,ExternalMTPAdapter
        from tutorial_code.miniengine.cache import make_cache,AttnCache,GDNCache
        from tutorial_code.miniengine.context import ForwardContext
        model_dir=Path(json.loads((P/'workload.json').read_text())['model_dir'])
        cfg=ModelConfig.from_dir(model_dir)
        def sig(array,save=None):
            mx.eval(array)
            host=np.ascontiguousarray(np.asarray(array.view(mx.uint16) if array.dtype==mx.bfloat16 else array))
            if array.dtype==mx.bfloat16:assert not np.any((host&0x7f80)==0x7f80)
            elif array.dtype in (mx.float16,mx.float32):assert np.isfinite(host).all()
            view=memoryview(host).cast('B')
            if save is not None:
                with (args.out/save).open('xb') as stream:stream.write(view)
            return {'shape':list(array.shape),'dtype':str(array.dtype),'bytes':view.nbytes,'sha256':hashlib.sha256(view).hexdigest()}
        def params(module):return {name:id(value) for name,value in tree_flatten(module.parameters())}
        def signatures(module):return {name:sig(value) for name,value in tree_flatten(module.parameters())}
        def cache_sig(caches):
            return [{'offset':c.offset,'arrays':[None if a is None else sig(a) for a in ((c.keys,c.values) if isinstance(c,AttnCache) else (c.conv_state,c.S))]} for c in caches]
        def clone(caches):
            result=[]
            for c in caches:
                copied=AttnCache() if isinstance(c,AttnCache) else GDNCache()
                if isinstance(c,AttnCache):
                    copied.keys=None if c.keys is None else c.keys[:]
                    copied.values=None if c.values is None else c.values[:]
                else:
                    copied.conv_state=None if c.conv_state is None else c.conv_state[:]
                    copied.S=None if c.S is None else c.S[:]
                copied.offset=c.offset;result.append(copied)
            return result
        checks=[];rounds=[]
        if args.mode=='draft-leaf':
            draft=load_draft(schema['draft_dir'],cfg)
            assert len(params(draft))==23
            before=signatures(draft)
            # Nonzero absolute RoPE with empty own KV proves the two frontiers
            # are distinct. This does not supply a synthetic target/head.
            draft.set_shared_kv({},17)
            x=mx.sin(mx.arange(3*5120,dtype=mx.float32)*0.003).reshape(3,5120).astype(mx.bfloat16)
            h=mx.cos(mx.arange(3*5120,dtype=mx.float32)*0.002).reshape(3,5120).astype(mx.bfloat16)
            inputs=[sig(x),sig(h)]
            out=draft.forward_embedded(x,h);checks.append({'output':sig(out,'leaf-output.raw'),'stage':'absolute17/store0->absolute20/store3'})
            assert out.shape==(3,5120) and out.dtype==mx.bfloat16 and draft._cache[0].offset==3 and draft._cache[0].position==20
            draft._cache[0].trim(1)
            assert draft._cache[0].offset==2 and draft._cache[0].position==19
            second=draft.forward_embedded(x[:1],h[:1]);checks.append({'output':sig(second,'leaf-after-trim.raw')})
            assert draft._cache[0].offset==3 and draft._cache[0].position==20
            assert inputs==[sig(x),sig(h)] and before==signatures(draft)
            result={'status':'COMPLETED_REAL_DRAFT_LEAF_FINITE_SMOKE_NO_TARGET_QUALIFICATION','draft':draft.metadata(),
                    'checks':checks,'draft23weights_and_inputs_readonly':True,'target_loaded':False}
        else:
            from tutorial_code.miniengine.model.loader import load_model
            target,_=load_model(model_dir)
            gdns=[layer.linear_attn for layer in target.layers if not layer.is_attn]
            for g in gdns:g.set_gates_compile(True);g.set_conv_norm_reuse_cache(True)
            assert len(gdns)==48
            target_owners=params(target);assert len(target_owners)==1349
            adapter=ExternalMTPAdapter()
            assert adapter.enabled is False and adapter.draft is None
            assert adapter.prepare(target,schema['draft_dir']).try_draft(None,None) is None
            assert params(target)==target_owners
            adapter.prepare(target,schema['draft_dir'],enabled=True)
            draft=adapter.draft
            assert params(target)==target_owners and len(params(draft))==23
            assert not set(params(draft).values()) & set(target_owners.values())
            assert draft._borrowed[0] is target.embed_tokens and draft._borrowed[1] is target.lm_head
            draft_before=signatures(draft)
            prompt=json.loads((P/'workload.json').read_text())['prompt_ids']
            ids=mx.array(prompt,dtype=mx.int32);ids_before=sig(ids)
            A,B=make_cache(target),make_cache(target)
            original=target(ids,ForwardContext('prefill',A))
            logits,hidden=target.forward_with_hidden(ids,ForwardContext('prefill',B))
            assert sig(original,'target-original-prefill.raw')==sig(logits,'target-hidden-seam-logits.raw')
            assert cache_sig(A)==cache_sig(B)
            assert sig(target.lm_head_fn(hidden))==sig(logits)
            prompt_hidden=hidden
            hidden_before=sig(prompt_hidden,'target-normalized-hidden.raw')
            bonus=int(mx.argmax(logits[-1]).item())
            draft.reset(target)
            draft.prefill_from_target_hidden(ids[None],hidden[None],bonus)
            draft.set_shared_kv({},len(prompt))
            assert draft._cache[0].offset==draft._cache[0].position==len(prompt)
            anchor=bonus;generated=[bonus]
            for round_index in range(2):
                proposed=adapter.try_draft(anchor,hidden[-1:][None],block_size=3)
                assert proposed.shape==(1,2) and proposed.dtype==mx.int32
                proposals=proposed[0].tolist()
                starting=cache_sig(B)
                trial=clone(B);histories=[];hs=[];posterior=[]
                for token in [anchor]+proposals:
                    score,h=target.forward_with_hidden(mx.array([token],dtype=mx.int32),ForwardContext('prefill',trial))
                    posterior.append(int(mx.argmax(score[-1]).item()));hs.append(h)
                    histories.append(clone(trial))
                assert cache_sig(B)==starting,'provisional target clone changed committed inputs'
                accepted=0
                while accepted<len(proposals) and proposals[accepted]==posterior[accepted]:accepted+=1
                role='actual-greedy-prefix'
                if round_index==0:
                    accepted=0;role='predeclared-decline-all-proposals-even-if-correct'
                produced=proposals[:accepted]+[posterior[accepted]]
                B=histories[accepted]
                committed_hidden=mx.concatenate(hs[:accepted+1],axis=0)[None]
                committed_before=sig(committed_hidden)
                draft.accept_verified_tokens(committed_hidden,proposed,accepted,produced)
                # Independent ordinary target advances its own current anchor;
                # no saved continuation or future A token is supplied to B.
                for delivered in produced:
                    ordinary=target(mx.array([anchor],dtype=mx.int32),ForwardContext('prefill',A))
                    assert int(mx.argmax(ordinary[-1]).item())==delivered
                    anchor=delivered;generated.append(delivered)
                assert cache_sig(A)==cache_sig(B)
                assert committed_before==sig(committed_hidden)
                assert draft._cache[0].position==B[0].offset
                mx.eval(*draft.eval_arrays())
                rounds.append({'role':role,'drafts':proposals,'posterior':posterior,'accepted':accepted,'delivered':produced,
                               'committed_hidden_rows':accepted+1,'draft_state':draft.metadata(),
                               'target_prefix_cache_same_as_independent_serial':True})
                hidden=committed_hidden[0]
            # Original prompt/hidden owners were not used as mutable cache.
            assert ids_before==sig(ids) and hidden_before==sig(prompt_hidden)
            assert draft_before==signatures(draft) and params(target)==target_owners
            before_reset=cache_sig(B)
            draft.reset(target)
            assert draft._cache[0].offset==draft._cache[0].position==0 and not draft._pending_round
            assert cache_sig(B)==before_reset
            adapter.prepare(target,schema['draft_dir'],enabled=False)
            assert adapter.try_draft(None,None) is None and params(target)==target_owners
            result={'status':'COMPLETED_EXTERNAL_MTP_HIDDEN_INTERFACE_DIAGNOSTIC_NO_ENGINE_OR_TARGET_VERIFY_QUALIFICATION',
                    'prefill_logits_and_128cache_original_call_strict':True,'parameter1349_identity_unchanged':True,
                    'draft23weights_readonly':True,'real_embedding_head_borrowed_originals':True,
                    'target_input_ids_readonly':True,'rounds':rounds,'actual_generated_diagnostic_ids':generated,
                    'fixed32_old_gold_or_vendor_draft_ID_equivalence_claim':False,'target_loaded':True}
        mx.synchronize()
        result.update(device=device,mode=args.mode,source_pins_sha256=args.pins_sha256,
                      memory={name:getattr(mx,name)() for name in ('get_active_memory','get_peak_memory','get_cache_memory')},
                      old_engine_default_unchanged=True,performance_qualified=False,V108_transaction_integrated=False,
                      target_revision=model_dir.name,external_draft_revision=Path(schema['draft_dir']).name,
                      actual_draft_parameter_arrays=len(params(draft)),author_loaded_payload=False)
        imported={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.startswith('tutorial_code.miniengine') and getattr(module,'__file__',None)}
        assert all(Path(path).is_relative_to(ROOT) and path in pins for path in imported.values())
        result['actual_candidate_imports']=imported
        resource('after-numeric-before-exit')
        identity(args.pins_sha256)
        write(args.out/'receipt.json',result)
        print(json.dumps({'status':result['status'],'mode':args.mode,'rounds':len(rounds)}),flush=True)
    except BaseException as exc:
        write(args.out/'failure.json',{'status':'FAIL_STOP','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
