"""V108 isolated numerical transaction qualification; CPU inert default."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from runtime_support import D,P,ROOT,identity,memory,runtime,signature,write,Observer,provenance
sys.dont_write_bytecode=True

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group();mode.add_argument('--inert',action='store_true');mode.add_argument('--execute-gpu',action='store_true');ap.add_argument('--out',type=Path);args=ap.parse_args(argv)
    pins=identity()
    if not args.execute_gpu:
        assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
        print(json.dumps({'status':'CPU_READY_TINY_ONLY_GPU_UNRUN','pins':len(pins),'GPU_used':False,'fullmodel_qualified':False}));return 0
    assert args.out and args.out.is_absolute();args.out.mkdir(exist_ok=False)
    r={'status':'STARTED','GPU_used':False,'model_loads':0,'pins':pins,'transactions':[],'scope':'one GDN cache + one attention cache, actual44 conv/state helpers; no target/draft weights loaded','timing_run':False,'fullmodel_qualified':False}
    save=lambda:write(args.out/'receipt.json',r);save()
    try:
        import mlx.core as mx
        import numpy as np
        r['runtime']=runtime(mx);mx.set_memory_limit(2<<30);mx.set_cache_limit(64<<20)
        assert not any(n.startswith('tutorial_code.miniengine') for n in sys.modules)
        sys.path.insert(0,str(ROOT))
        from tutorial_code.miniengine.cache import AttnCache,GDNCache
        from tutorial_code.miniengine.spec.transaction import VerifyTransaction,fused_recurrent_history,select_delivery
        AttnCache.step=64
        bank=P/'v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1/synthetic-random-PCG64-18001'
        meta=json.loads((bank/'bank.json').read_text())
        def load(key):
            from runtime_support import sha
            a=meta['arrays'][key];assert sha(bank/a['file'])==a['sha256']
            host=np.fromfile(bank/a['file'],dtype='<u2' if a['dtype']=='bfloat16' else '<f4').reshape(a['shape'])
            v=mx.array(host);return v.view(mx.bfloat16) if a['dtype']=='bfloat16' else v
        u,g,beta,S=map(load,('u','g','beta','S'))
        weights=mx.ones((10240,4),mx.bfloat16)*0.25
        source_before=[signature(mx,np,value) for value in (u,g,beta,S,weights)]
        observer=Observer(mx,np,r['transactions'],save)
        r['GPU_used']=True
        for length in (1,3,5,8):
            for keep in range(length+1):
                ac=AttnCache();ac.keys=mx.zeros((1,4,64,256),mx.bfloat16);ac.values=mx.zeros_like(ac.keys)
                ac.keys[:,:,:2,:]=0.25;ac.values[:,:,:2,:]=-0.5
                # Abort keeps the old tail; committed nonzero prefixes zero it.
                ac.keys[:,:,20:21,:]=0.75;ac.values[:,:,20:21,:]=0.75;ac.offset=2
                gc=GDNCache();gc.S=S[0];gc.conv_state=mx.zeros((3,10240),mx.bfloat16);gc.offset=2
                raw=mx.concatenate([mx.roll(u,j,axis=-1) for j in range(length)],axis=0)
                gs=mx.concatenate([g+j*0.001 for j in range(length)],axis=0);bs=mx.concatenate([beta]*length,axis=0)
                k=mx.arange(4*length*256,dtype=mx.float32).reshape(1,4,length,256).astype(mx.bfloat16)*0.0001
                v=-k
                with VerifyTransaction([ac,gc],length) as t:
                    info={'case':f'T{length}-keep{keep}','committed_inputs':keep,'verify_inputs':length,'scripted_prefix_not_model_prediction':True}
                    observer('begin',t,info)
                    ac.update(k,v)
                    out=fused_recurrent_history(raw,gs,bs,weights,gc,mx)
                    mx.eval(out,*t.eval_arrays())
                    observer('before_commit',t,info);t.commit(keep);observer('after_commit',t,info)
                    assert ac.offset==gc.offset==2+keep
                    if keep:
                        for a in (ac.keys,ac.values):assert np.all(np.asarray(a.view(mx.uint16))[:,:,2+keep:,:]==0)
                del ac,gc,raw,gs,bs,k,v,out,t
        # CPU-selection logic at a real transaction boundary is separately
        # scripted; it does not masquerade as predictions from a loaded LLM.
        r['delivery_cases']=[{'drafts':d,'posterior':p,'budget':b,'eos':list(e),'result':select_delivery(d,p,b,e)} for d,p,b,e in [([10,11],[10,11,12],1,()),([10,11],[10,11,12],3,(10,)),([10,11],[10,11,12],3,(11,)),([10,11],[10,11,12],3,(12,)),([10,11],[90,11,12],3,())]]
        ac=AttnCache();ac.keys=mx.zeros((1,4,64,256),mx.bfloat16);ac.values=mx.zeros_like(ac.keys);ac.offset=2
        gc=GDNCache();gc.S=S[0];gc.conv_state=mx.zeros((3,10240),mx.bfloat16);gc.offset=2
        initial=[signature(mx,np,a) for a in (ac.keys,ac.values,gc.conv_state,gc.S)]
        try:
            with VerifyTransaction([ac,gc],3):raise RuntimeError('intentional partial producer abort')
        except RuntimeError as error:assert str(error)=='intentional partial producer abort'
        assert initial==[signature(mx,np,a) for a in (ac.keys,ac.values,gc.conv_state,gc.S)]
        assert ac.offset==gc.offset==2 and ac._verify108 is gc._verify108 is None
        assert len(r['transactions'])==21 and all(v['same_invocation_prefix_strict'] and v['literal_history_readonly'] and v['invocation_inputs_readonly'] for v in r['transactions'])
        source_after=[signature(mx,np,value) for value in (u,g,beta,S,weights)];assert source_before==source_after
        r.update(status='PASS_V108_TINY_LITERAL_PREFIX_TRANSACTION_21CASES',strict_transactions=21,abort_restored=True,source_inputs_readonly=True,source_before=source_before,source_after=source_after,memory=memory(mx),actual_imported_sources=provenance(pins));identity();return 0
    except BaseException as error:
        r.update(status='FAIL_STOP',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        r['completed_strict_transactions']=sum(row.get('same_invocation_prefix_strict') is True and row.get('literal_history_readonly') is True and row.get('invocation_inputs_readonly') is True for row in r['transactions'])
        save()

if __name__=='__main__':raise SystemExit(main())
