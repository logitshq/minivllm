"""V117 root-only full27B numerical qualification. CPU inert default; no timing."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from runtime_support import D,P,ROOT,identity,memory,runtime,signature,write,Observer,provenance
sys.dont_write_bytecode=True

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group();mode.add_argument('--inert',action='store_true');mode.add_argument('--execute-gpu',action='store_true');ap.add_argument('--out',type=Path);ap.add_argument('--tiny-receipt',type=Path);ap.add_argument('--tiny-sha256');args=ap.parse_args(argv)
    pins=identity();gold=json.loads((D/'assertion-gold.json').read_text());protocol=json.loads((D/'protocol.json').read_text())
    if not args.execute_gpu:
        assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
        print(json.dumps({'status':'CPU_READY_FULL_NUMERIC_GPU_UNRUN','pins':len(pins),'required_prior_tiny_actual':True,'V113_binding':json.loads((D/'v113-binding.json').read_text())['status'],'V116_binding':json.loads((D/'v116-binding.json').read_text())['status'],'requires_bound113_PASS_and_independent_review':True,'expected_old_IDs_each_arm':96,'GPU_used':False,'performance_prepared':False}));return 0
    from qualification_gate import require_v113
    prerequisite=require_v113()  # hard HOLD checked before importing MLX
    from loader_gate import require_v116
    loader_prerequisite=require_v116()
    from runtime_support import sha
    assert args.out and args.out.is_absolute();assert args.tiny_receipt and args.tiny_receipt.is_absolute() and args.tiny_sha256
    assert sha(args.tiny_receipt)==args.tiny_sha256
    tiny=json.loads(args.tiny_receipt.read_text());assert tiny['status']=='PASS_V108_TINY_LITERAL_PREFIX_TRANSACTION_21CASES' and tiny['strict_transactions']==21 and tiny['abort_restored']
    assert len(tiny['transactions'])==21 and all(row['same_invocation_prefix_strict'] and row['literal_history_readonly'] and row['invocation_inputs_readonly'] for row in tiny['transactions'])
    assert tiny['source_inputs_readonly']
    assert all(pins[p]==h for p,h in tiny['pins'].items()),'tiny source pins changed'
    args.out.mkdir(exist_ok=False)
    r={'status':'STARTED','model_loads':0,'GPU_used':False,'pins':pins,'tiny_receipt':str(args.tiny_receipt),'tiny_receipt_sha256':args.tiny_sha256,'profile':protocol['profile'],'timing_run':False,'performance_qualified':False,'adopted':False,'requests':[],'transactions':[],
       'old96_ID_hard_gate_each_arm':True,'V113_prerequisite':prerequisite,'V116_loader_prerequisite':loader_prerequisite,'bounded_loader_stages':[],'ordinary_M1_reference_requests':[],'same_frontier_initial_checks':[],'same_frontier_commit_checks':[],'all_same_frontier_128_and_last_logits_are_HARD_GATES':True,'old_tensor_gold_not_relaxed':True,'old128_state_hash_diagnostics':[],
       'draft_source_file_bytes':3848817896,'draft_memory_scope':'raw checkpoint size; loaded/quantization/history/KV allocator counters separately observed','weight_loads_expected':1,'native_or_shader_changed':False}
    save=lambda:write(args.out/'receipt.json',r);save()
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten
        r['runtime']=runtime(mx);mx.set_memory_limit(34<<30);mx.set_cache_limit(2<<30)
        assert not any(n.startswith('tutorial_code.miniengine') for n in sys.modules)
        sys.path.insert(0,str(ROOT))
        from tutorial_code.miniengine.engine.llm_engine import LLMEngine
        from tutorial_code.miniengine.engine.model_runner import ModelRunner
        from tutorial_code.miniengine.cache import AttnCache,DecodeBatch
        from tutorial_code.miniengine.model.gdn import GatedDeltaNet
        from tutorial_code.miniengine.sampling_params import SamplingParams
        from tutorial_code.miniengine.spec.dflash2 import load_drafter
        AttnCache.step=DecodeBatch.step=64
        engine=LLMEngine(protocol['target_model_dir'],max_seqs=1,max_len=64,max_batched_tokens=512)
        r['model_loads']=1;r['GPU_used']=True
        gdns=[layer.linear_attn for layer in engine.model.layers if not layer.is_attn]
        assert len(gdns)==48 and all(type(g) is GatedDeltaNet for g in gdns)
        for gdn in gdns:gdn.set_gates_compile(True);gdn.set_conv_norm_reuse_cache(True)
        parameters=list(tree_flatten(engine.model.parameters()));assert len(parameters)==1349
        identities=[(name,id(value)) for name,value in parameters]
        r['parameter_identity_before']=identities
        def parameter_signatures():
            assert identities==[(name,id(value)) for name,value in tree_flatten(engine.model.parameters())]
            return {name:signature(mx,np,value) for name,value in parameters}
        before=parameter_signatures();r['parameter_before']=before;r['target1349_identity_before']=True
        r['target_loaded_memory']=memory(mx)
        r['actual_imported_sources_before_token']=provenance(pins)
        write(args.out/'model-ready.json',{'pid':__import__('os').getpid(),'model_loads':1,'token_executed':False,'draft_loaded':False})
        print(json.dumps({'event':'MODEL_READY','pid':__import__('os').getpid()}),flush=True);save()
        from ordinary_reference import OrdinaryM1Reference
        reference_by_request={}
        def generate(arm,name,prompt,budget,expected):
            # Only prompt IDs enter add_request. Expected continuation is read
            # solely after completed live generation, never by draft/feedback.
            rid=arm+'.'+name
            reference=None
            if arm in ('B','E'):
                reference=OrdinaryM1Reference(mx,np,engine.model,prompt,64,signature)
                reference_by_request[rid]=reference
            request=engine.add_request(prompt,SamplingParams(temperature=0,max_tokens=budget),req_id=rid)
            emitted=[];deliveries=[]
            while engine.has_work():
                values=engine.step();batch=[token for current,token,finished in values if current==rid]
                if batch:deliveries.append(batch);emitted.extend(batch)
            row={'arm':arm,'name':name,'prompt_ids':prompt,'budget':budget,'IDs':emitted,'deliveries':deliveries,'strict_old_ID_equal':emitted==expected,'input_future_gold_used':False}
            r['requests'].append(row);save()
            assert emitted==expected,('old38/44/89 greedy IDs differ',arm,name,emitted,expected)
            assert not engine.requests and not engine.runner.staging and engine.runner.batch.size==0 and not engine.runner.feats and engine.runner._ahead is None
            row['runner_empty_after_finish']=True
            if reference is not None:
                assert reference.steps==budget-1 and reference.frontier==len(prompt)+budget-1
                assert reference.tokens==emitted,'independent ordinary M1 IDs differ from live B outputs'
                r['ordinary_M1_reference_requests'].append({'request':rid,'prompt_ids':list(prompt),'actual_IDs':list(reference.tokens),'computed_frontier':reference.frontier,'ordinary_decode_steps':reference.steps,'prefill_logits':reference.prefill_logits_signature,'IDs_fed_to_B':False})
                del reference_by_request[rid]
            if engine.runner.spec is not None:
                assert not engine.runner.spec.state
                assert all(layer.self_attn.ctx_len==0 and layer.self_attn.ctx_k is None and layer.self_attn.ctx_v is None for layer in engine.runner.spec.drafter.layers)
                row['draft_reset_after_finish']=True
            current=parameter_signatures();row['target1349_readonly']=current==before;assert current==before,'target parameter bytes changed'
            row['memory_after_finish']=memory(mx);save();return emitted
        for name,row in gold['requests'].items():generate('A',name,row['prompt_ids'],32,row['expected_IDs'])
        r['A_old96_IDs_passed']=True;save()
        def bounded_loader_stage(label, detail):
            r['bounded_loader_stages'].append({'label':label,'detail':detail,'allocator':memory(mx)})
            save()
        drafter=load_drafter(protocol['draft_snapshot'],bounded_load=True,stage_observer=bounded_loader_stage)
        r['draft_parameter_arrays_observed']=len(tree_flatten(drafter.parameters()))
        r['after_draft_load_memory']=memory(mx)
        assert parameter_signatures()==before,'draft load changed target parameters'
        engine.runner=ModelRunner(engine.model,64,drafter,spec_block=5)
        observer=Observer(mx,np,r['transactions'],save)
        old44=json.loads(Path(protocol['old_receipts'][1]).read_text());old_by_name={row['name']:row for row in old44['requests']}
        def audit(phase,transaction,info):
            if phase=='begin':
                initial=reference_by_request[info['request']].compare_begin(transaction)
                r['same_frontier_initial_checks'].append({'request':info['request'],**initial});save()
                assert initial['all128_strict_equal'] and initial['all32_attention_tails_zero'],'new114 initial A/B same-frontier cache hard gate failed'
            observer(phase,transaction,info)
            if phase!='after_commit' or info['remaining_before']!=info['delivered'] or not info['request'].startswith('B.'):return
            name=info['request'][2:]
            if name not in old_by_name:return
            old={row['label']:row for row in old_by_name[name]['steps'][-1]['outputs']}
            diagnostics=[]
            for layer,trace in enumerate(transaction.traces):
                values=(trace.cache.keys,trace.cache.values) if trace.kind=='attention' else (trace.cache.conv_state,trace.cache.S)
                for component,value in enumerate(values):
                    label=f'state.{2*layer+component}';sig=signature(mx,np,value);digest=sig['sha256']
                    if trace.kind=='attention':
                        host=np.ascontiguousarray(np.asarray(value.view(mx.uint16)),dtype='<u2');raw=host.tobytes();stride=64*256*2
                        assert value.shape==(1,4,64,256)
                        assert np.all(host[:,:,trace.cache.offset:,:]==0)
                        canonical=b''.join(raw[h*stride:(h+1)*stride]+bytes((256-64)*256*2) for h in range(4))
                        digest=__import__('hashlib').sha256(canonical).hexdigest()
                    diagnostics.append({'label':label,'new_sha256':digest,'old44_sha256':old[label]['A_sha256'],'strict_equal':digest==old[label]['A_sha256']})
            r['old128_state_hash_diagnostics'].append({'name':name,'scope':'128 committed cache states only, CPU KV64=>256 zero padding; diagnostics never toleranced or labelled old132 PASS','equal':sum(d['strict_equal'] for d in diagnostics),'different':sum(not d['strict_equal'] for d in diagnostics),'states':diagnostics});save()
        engine.runner.spec.set_verify_transaction(True,eos_ids=engine.eos_ids,capacity=64,observer=audit)
        def compare_commit114(logits,transaction,info):
            comparison=reference_by_request[info['request']].compare_commit(logits,transaction,info)
            r['same_frontier_commit_checks'].append(comparison);save()
            assert comparison['hard_gate_pass'],'new114 ordinaryM1 vs propagated B committed128/lastlogits strict hard gate failed'
        engine.runner.spec.set_verify_broadcast_m1(True,logits_observer=compare_commit114)
        assert len(engine.runner.spec._verify_m1_ops114)==497
        for name,row in gold['requests'].items():generate('B',name,row['prompt_ids'],32,row['expected_IDs'])
        r['B_old96_IDs_passed']=True
        for budget in (1,2,3,5):
            row=gold['requests']['base'];generate('E',f'budget{budget}',row['prompt_ids'],budget,row['expected_IDs'][:budget])
        rounds=engine.runner.spec.transaction_rounds
        assert len(rounds)==len(r['transactions']) and not observer.pending
        assert all(row['same_invocation_prefix_strict'] and row['literal_history_readonly'] and row['invocation_inputs_readonly'] for row in r['transactions'])
        assert all(row['GDN_helper_calls']==48*row['verify_inputs'] for row in rounds)
        assert all(row['committed_inputs']==row['delivered'] and row['next_frontier']==row['start']+row['committed_inputs'] for row in rounds)
        assert len(r['same_frontier_commit_checks'])==len(rounds) and all(row['hard_gate_pass'] for row in r['same_frontier_commit_checks'])
        assert not reference_by_request
        assert all(row['every_Linear_once'] and row['verify_lm_head_calls']==1 and row['flags_restored_before_eval_or_draft'] for row in engine.runner.spec.verify_m1_rounds114)
        assert all(getattr(op,'_verify_m1_scope114',None) is None for name,op in engine.runner.spec._verify_m1_ops114)
        r['actual_verify_broadcast_rounds']=engine.runner.spec.verify_m1_rounds114
        after=parameter_signatures();assert after==before
        r['parameter_identity_after']=[(name,id(value)) for name,value in tree_flatten(engine.model.parameters())]
        r['parameter_after']=after
        r['actual_imported_sources_final']=provenance(pins)
        assert len(r['requests'])==10 and all(row['target1349_readonly'] and row['runner_empty_after_finish'] for row in r['requests'])
        r.update(status='PASS_V117_BOUNDED_LOAD_VERIFY_BROADCAST_M1_OLD96IDS_LITERAL_AND_ORDINARY_M1_COMMIT128_LASTLOGITS_STRICT',old_ID_checks_each_arm=96,target_parameter_identity_count=1349,target1349_readonly=True,strict_transactions=len(rounds),same_invocation_cache_array_checks=128*len(rounds),draft_calls=sum(row['draft_calls'] for row in rounds),drafted_tokens=sum(row['drafted_tokens'] for row in rounds),delivered_decode_tokens=sum(row['delivered'] for row in rounds),ordinary_fresh_prefill_not_prefix_bitgold=True,full_old132_tensor_equivalence_claim=False,memory_final=memory(mx))
        identity();return 0
    except BaseException as error:
        r.update(status='FAIL_STOP',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        r['completed_old_ID_checks']={arm:sum(len(row['IDs']) for row in r['requests'] if row['arm']==arm and row.get('strict_old_ID_equal')) for arm in ('A','B')}
        r['completed_strict_transactions']=sum(row.get('same_invocation_prefix_strict') is True for row in r['transactions'])
        save()

if __name__=='__main__':raise SystemExit(main())
