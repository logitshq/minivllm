"""Root-only target-only V127 local same-input numeric qualification. Inert by default."""
import argparse,collections,gc,hashlib,importlib.metadata,importlib.util,json,os,struct,sys,traceback
from pathlib import Path
from capture import Capture
from router import ProjectionRouter
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;P=D.parent
ROOT=P/'candidate-v116-bounded-draftloader-build1-root'
CONTROLS={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
LIMIT=34<<30;CACHE=64<<20
PYTHON='/Users/ryanli/code/minivllm/.venv/bin/python'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def identity():
    assert not any(n=='mlx' or n.startswith(('mlx.','numpy','tutorial_code.miniengine')) for n in sys.modules),'inference preimport forbidden'
    pins=read(D/'source-pins.json')
    for p,h in pins.items():assert sha(p)==h,('source/pinned evidence changed',p)
    b=read(D/'prerequisites.json')
    q=read(b['V124_actual']);assert q['status']=='PASS_LOCAL_V124_SHARED_WEIGHT_M1_STRICT_SYNTHETIC' and q['strict_matrix_checks']==80 and q['readonly_all_pass']
    a=read(b['V124_review']);assert a['status']=='PASS_SAVED_V124_SYNTHETIC_NUMERIC_REVIEW' and a['actual_receipt_sha256']==sha(b['V124_actual'])
    old=read(b['V113_actual']);assert old['status']=='PASS_V113_ALL497_SAME_INPUT_BROADCAST_M1_STRICT_NUMERIC_ONLY' and old['broadcastM1_vs_rowM1_strict_equal_count']==497 and old['keep0_restored_all128'] and old['target1349_identity_readonly']
    for prefix in ('V124','V113'):
        release=read(b[prefix+'_release']);assert release['GPU_released'] is True
        assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
    for row in read(D/'header-pins.json'):
        p=Path(row['path']);assert p.stat().st_size==row['file_bytes']
        with p.open('rb') as f:
            prefix=f.read(8);n=struct.unpack('<Q',prefix)[0];assert 0<n<32*1024*1024;body=f.read(n)
        assert hashlib.sha256(prefix+body).hexdigest()==row['prefix_header_SHA256']
    return pins

def run(out,pins):
    r={'status':'STARTED_LOCAL_QUALIFICATION','GPU_used':False,'model_loads':0,'draft_loads':0,'live_drafting':False,'future_A_output_used_as_proposal':False,'timing_run':False,'fullmodel_qualified':False,'adopted':False,'comparisons':[],'routes':[],'layouts':[],'readonly':[],'original_hash_checks':[],'resources':[],'source_pins_sha256':sha(D/'source-pins.json')}
    save=lambda:write(out/'receipt.json',r)
    save();capture=None;mx=None;parameters=None;before=None;error=None;final_errors=[]
    try:
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        sys.path.insert(0,str(P))
        from resource_snapshot_power_v1 import snapshot,deltas,guard
        assert sys.executable==PYTHON and importlib.metadata.version('mlx')==importlib.metadata.version('mlx-metal')=='0.32.3'
        selected={k:v for k,v in os.environ.items() if k.startswith(('MLX_','METAL_','DYLD_','OMLX_','MONOLITH_','LITHOS_')) or k in ('PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE')}
        assert selected==CONTROLS,'exact50/50/20/0 environment without DYLD required'
        device=mx.device_info();assert mx.default_device()==mx.gpu and device['device_name']=='Apple M4 Max' and device['architecture']=='applegpu_g16s'
        mx.set_memory_limit(LIMIT);mx.set_cache_limit(CACHE)
        r['runtime']={'versions':{'mlx':'0.32.3','mlx-metal':'0.32.3'},'device':device,'controls':CONTROLS,'memory_limit_bytes':LIMIT,'cache_limit_bytes':CACHE}
        def memory():return {key:getattr(mx,key)() for key in ('get_active_memory','get_cache_memory','get_peak_memory')}
        def probe(label):
            mx.synchronize();raw=snapshot(label,os.getpid());write(out/('resource-%03d-raw.json'%len(r['resources'])),raw)
            row={**raw,'mlx_memory':memory()};r['resources'].append(row);deltas(r['resources']);save();guard(r['resources'])
            assert row['power_metadata']['status']=='known' and row['power_metadata']['source']=='AC Power'
            assert row['processinfo_thermal_metadata']['status']=='known' and row['processinfo_thermal_metadata']['code'] in (0,1)
            assert row['pressure_level_value']==1 and row['exact_owned_pid_metadata']['status']=='known'
            assert all(x['owned'] for x in row['known_inference_processes'])
            assert row['swap_used_mb']<=r['resources'][0]['swap_used_mb'] and row['vm_counters']['Swapouts']==r['resources'][0]['vm_counters']['Swapouts']
            assert row['mlx_memory']['get_active_memory']<=LIMIT and row['mlx_memory']['get_peak_memory']<=LIMIT
        def host(value):
            view=value.view(mx.uint16) if value.dtype==mx.bfloat16 else value;mx.eval(view)
            b=memoryview(view);assert b.c_contiguous and b.nbytes==value.nbytes,'raw export requires canonical contiguous view; no copying operands'
            return b.cast('B'),view
        def signature(value):
            b,owner=host(value)
            if value.dtype==mx.bfloat16:
                words=np.frombuffer(b,dtype='<u2');assert not np.any((words&0x7f80)==0x7f80),'nonfinite BF16'
            elif value.dtype in (mx.float32,mx.float16):assert np.isfinite(np.frombuffer(b,dtype=np.float32 if value.dtype==mx.float32 else np.float16)).all()
            return {'shape':list(value.shape),'dtype':str(value.dtype),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        def parameter_signatures():
            assert [(n,id(v)) for n,v in tree_flatten(model.parameters())]==identities,'parameter identity changed'
            return {n:signature(v) for n,v in parameters}
        probe('before-target-load')
        sys.path.insert(0,str(ROOT))
        from tutorial_code.miniengine.engine.llm_engine import LLMEngine
        from tutorial_code.miniengine.cache import AttnCache,DecodeBatch,make_cache
        from tutorial_code.miniengine.context import ForwardContext
        from tutorial_code.miniengine.model import layers
        from tutorial_code.miniengine.spec.transaction import VerifyTransaction
        from tutorial_code.miniengine.sampling_params import SamplingParams
        protocol=read(D/'protocol.json');block=read(P/'v113-broadcast-m1-diagnostic-build1/actual-block.json')
        AttnCache.step=DecodeBatch.step=64
        engine=LLMEngine(protocol['target_model_dir'],max_seqs=1,max_len=64,max_batched_tokens=512,dtype=mx.bfloat16,cache_limit_gb=0)
        mx.set_cache_limit(CACHE);model=engine.model;r.update(GPU_used=True,model_loads=1)
        gdns=[layer.linear_attn for layer in model.layers if not layer.is_attn];assert len(gdns)==48
        for g in gdns:g.set_gates_compile(True);g.set_conv_norm_reuse_cache(True)
        parameters=list(tree_flatten(model.parameters()));assert len(parameters)==1349
        identities=[(n,id(v)) for n,v in parameters];before=parameter_signatures();r['parameters_before']=before;r['parameter_identity_before']=identities
        assert engine.runner.spec is None
        assert all(getattr(op,'_verify_m1_scope114',None) is None for _,op in model.named_modules())
        probe('after-target-load-before-any-token')
        write(out/'model-ready.json',{'pid':os.getpid(),'target_loads':1,'draft_loads':0,'token_executed':False,'physical_capacity':64});save()
        # Exact frozen V123 one-request warm: no draft, future IDs assert-only.
        startup=module('v127_source_v123_adapter',P/'v123-startup-ordinary-warm-build1/adapter.py')
        r['ordinary_warm']=startup.ordinary_startup_warm(engine,SamplingParams,read(P/'workload.json'))
        probe('after-ordinary-warm-runner-empty')
        expected_names={name for name,op in model.named_modules() if type(op) is layers.Linear}
        assert len(expected_names)==497
        caches=make_cache(model)
        logits,features=model(mx.array(block['prompt_ids']),ForwardContext('prefill',caches),last_only=True,capture=(5,19,33,47,61))
        anchor_array=mx.argmax(logits,axis=-1);mx.eval(anchor_array,features);anchor=int(anchor_array.item());r['live_anchor']=anchor;assert anchor==block['assert_live_anchor']
        def cache_signatures():
            return [{'kind':'attention' if hasattr(c,'keys') else 'gdn','offset':c.offset,'values':[signature(v) for v in ((c.keys,c.values) if hasattr(c,'keys') else (c.conv_state,c.S))]} for c in caches]
        initial=cache_signatures();assert initial==block['initial_cache'];r['initial_cache_matches_actual108_all128']=True
        capture=Capture(model,layers,block['verify_inputs']);capture.install()
        with VerifyTransaction(caches,block['verify_inputs']) as transaction:
            capture.active=True
            try:
                # Saved real B proposals only. No alternate output/ordinary future IDs enter this graph.
                logits,new_features=model(mx.array([anchor]+block['drafts']),ForwardContext('prefill',caches),capture=(5,19,33,47,61))
            finally:capture.active=False
            posterior=mx.argmax(logits,axis=-1)
            mx.eval(posterior,new_features,*transaction.eval_arrays());mx.synchronize()
            r['observed_posterior']=posterior.tolist();assert r['observed_posterior']==block['assert_posterior']
            records=capture.records;capture.close();capture=None
            assert len(records)==497 and {x['name'] for x in records}==expected_names and len({x['name'] for x in records})==497
            assert records[-1]['name']=='lm_head';r['captured_actual_T5_operator_count']=497
            probe('original-batch-capture-complete-hooks-removed')
            old_gold=read(P/'v113-broadcast-m1-diagnostic-build1/actual-linear-gold.json')['operators']
            old_actual=read(read(D/'prerequisites.json')['V113_actual'])
            old_inputs={x['name']:x for x in old_actual['comparisons'] if x['variant']=='batch_vs_rowM1'}
            # Verify every original input and completed batch output BEFORE building any alternative.
            for rec in records:
                x,y=rec['x'],rec['y'];sx,sy=signature(x),signature(y);gold=old_gold[rec['name']]
                assert [sx]==gold['input_signatures']==old_inputs[rec['name']]['input_signatures']
                assert sy['sha256']==gold['batch_sha256']==old_inputs[rec['name']]['left_sha256']
                rec['before']={'x':sx,'y':sy,'weight':signature(rec['op'].weight),'scales':signature(rec['op'].scales)}
                rec['owner_ids']=[id(v) for v in (x,y,rec['op'].weight,rec['op'].scales)]
                r['original_hash_checks'].append({'name':rec['name'],'input':sx,'batch_output':sy,'old112_and113_equal':True})
            r['all497_original_input_and_batch_output_hashes_reproduced']=True;save()
            shared=module('v127_source_shared124',P/'v124-shared-weight-m1-source-build1/shared_m1.py')
            fallback=module('v127_source_fallback113',P/'v113-broadcast-m1-diagnostic-build1/broadcast_m1.py').broadcast_m1
            router=ProjectionRouter(mx,shared,fallback,{'runtime_version':'0.32.3','device_name':device['device_name'],'architecture':device['architecture']});assert router.enabled is False;router.set_enabled(True)
            layout_kernel=mx.fast.metal_kernel(name='v127_observe_original_operand_layout',input_names=['value'],output_names=['descriptor'],source=(D/'layout_probe.metal').read_text(),ensure_row_contiguous=False)
            def layout(value,label):
                assert value.ndim==2
                output=layout_kernel(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(4,)],output_dtypes=[mx.uint64])[0];mx.eval(output)
                v=output.tolist();assert v[:2]==list(value.shape)
                r['layouts'].append({'label':label,'shape':v[:2],'actual_strides_elements':v[2:],'internal_MLX_flags_observed':False,'ensure_row_contiguous':False})
                return tuple(v[2:])
            for rec in records:
                op,x,y=rec['op'],rec['x'],rec['y'];N,K=op.weight.shape[0],x.shape[-1]
                ws,ss=layout(op.weight,rec['name']+'.weight'),layout(op.scales,rec['name']+'.scales')
                for T in protocol['T_cases']:
                    actual_x=x if T==5 else x[:T];xs=layout(actual_x,rec['name']+f'.X.T{T}');x_before=signature(actual_x)
                    native_rows=[mx.quantized_matmul(actual_x[i:i+1],op.weight,op.scales,None,transpose=True,**shared.Q) for i in range(T)]
                    native=mx.concatenate(native_rows,axis=0);mx.eval(native)
                    sn=signature(native)
                    if T==5:
                        assert sn['sha256']==old_gold[rec['name']]['row_M1_sha256']==old_inputs[rec['name']]['right_sha256']
                    actual,route=router.project(op,actual_x,weight_strides=ws,scale_strides=ss,x_strides=xs);mx.eval(actual)
                    sa=signature(actual);ab,ao=host(actual);nb,no=host(native)
                    same=actual.shape==native.shape==(T,N) and actual.dtype==native.dtype==mx.bfloat16 and sa==sn and ab==nb
                    item={'name':rec['name'],'order':rec['order'],'T':T,'N':N,'K':K,'input_scope':'actual_saved_B_verify_T5' if T==5 else 'derived_prefix_of_actual_T5_X','input_signature':x_before,'A_native_M1':sn,'B_shared_or_fallback':sa,'shape_dtype_finite_bytes_strict':same,**route}
                    r['comparisons'].append(item);r['routes'].append({'name':rec['name'],'T':T,'N':N,'K':K,**route})
                    if not same:
                        save();write(out/'first-failure.json',item);(out/'first-failure-A.bf16').write_bytes(nb);(out/'first-failure-B.bf16').write_bytes(ab);raise AssertionError('first complete BF16 output mismatch STOP')
                    assert signature(actual_x)==x_before,'prefix input bytes changed'
                    # Drop each alternative pair and its lazy owners before the next case.
                    del ab,ao,nb,no,actual,native,native_rows,actual_x;gc.collect()
                after={'x':signature(x),'y':signature(y),'weight':signature(op.weight),'scales':signature(op.scales)}
                ids=[id(v) for v in (x,y,op.weight,op.scales)];same=after==rec['before'] and ids==rec['owner_ids']
                r['readonly'].append({'name':rec['name'],'before':rec['before'],'after':after,'owner_ids_before':rec['owner_ids'],'owner_ids_after':ids,'unchanged':same});assert same
                if (rec['order']+1)%32==0:probe('compared-through-%03d'%(rec['order']+1))
            expected=collections.Counter({(T,*map(int,key.replace('N','').split(':K'))):count for T in (5,2,4) for key,count in protocol['expected_geometry_operator_counts'].items()})
            observed=collections.Counter((row['T'],row['N'],row['K']) for row in r['comparisons'] if row['route']=='shared124')
            assert observed==expected and len(r['comparisons'])==1491 and all(row['shape_dtype_finite_bytes_strict'] for row in r['comparisons'])
            assert len(r['readonly'])==497 and all(x['unchanged'] for x in r['readonly'])
            r['all497_native_M1_output_hashes_reproduced']=True
            r['shared_route_counts']=[{'T':T,'N':N,'K':K,'calls':count} for (T,N,K),count in sorted(observed.items())]
            r['fallback_calls']=sum(row['route']!='shared124' for row in r['routes']);assert r['fallback_calls']==0
            transaction.commit(0)
        assert cache_signatures()==initial and all(getattr(c,'_verify108',None) is None for c in caches)
        r['keep0_restored_all128']=True
        after=parameter_signatures();r['parameters_after']=after;assert after==before;r['target1349_identity_readonly']=True
        imported={}
        for name,mod in tuple(sys.modules.items()):
            if name.startswith('tutorial_code.miniengine') and getattr(mod,'__file__',None):
                path=str(Path(mod.__file__).resolve());assert Path(path).is_relative_to(ROOT) and pins.get(path)==sha(path);imported[name]=path
        r['actual_imported_candidate_sources']=imported
        probe('after-local-comparisons-keep0-parameter-readonly')
        r['status']='PASS_V127_497_REAL_T5_AND_DERIVED_T2_T4_SHARED_W_M1_LOCAL_ONLY'
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};r.update(status='FAIL_STOP',error=error)
    finally:
        if capture is not None:capture.close()
        if mx is not None and 'probe' in locals():
            try:probe('finally')
            except BaseException as exc:final_errors.append({'stage':'final_resource','error':repr(exc)})
        r['final_errors']=final_errors
        if final_errors:r['status']='FAIL_STOP'
        r['completed_matrix_checks']=len(r['comparisons']);r['strict_equal_matrix_checks']=sum(x['shape_dtype_finite_bytes_strict'] for x in r['comparisons']);save()
    return 1 if error is not None or final_errors else 0

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inert',action='store_true');ap.add_argument('--execute-gpu',action='store_true');ap.add_argument('--out',type=Path);a=ap.parse_args(argv)
    pins=identity()
    if not a.execute_gpu:
        print(json.dumps({'status':'CPU_INERT_V127_SOURCE_PINS_PASS_GPU_UNRUN','pins':len(pins),'GPU_used':False,'target_loads':0,'draft_loads':0,'planned_actual_T5_operators':497,'planned_prefix_matrix_checks':1491}));return 0
    assert a.out is not None and a.out.is_absolute() and not a.out.exists(),'fresh absolute output required';a.out.mkdir();write(a.out/'preimport-pins.json',pins)
    return run(a.out,pins)
if __name__=='__main__':raise SystemExit(main())
