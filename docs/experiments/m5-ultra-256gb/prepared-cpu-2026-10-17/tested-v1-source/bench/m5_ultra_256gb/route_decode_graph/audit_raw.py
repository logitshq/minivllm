"""Independent CPU-only original-byte audit of the B1/used257 route canary.

An audited numerical rejection is a readable PASS audit with canary_qualified
false. Neither outcome grants the remaining matrix/lifecycle or performance gates.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'long_decode_isolation/audit_raw.py'
BASE_SHA='b3bffc7bd261b9b347b5d299d965273ff2406a92c27576747867aa515be62261'
QUALIFIER_SHA='fe8e1c044e97885ec7c97fddce04a6e34f5b9067be4d982c5f6ad1071c59c801'
CANDIDATE_SHA='6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3'


def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def base():
    need(sha(BASE)==BASE_SHA,'frozen original-byte auditor source')
    spec=importlib.util.spec_from_file_location('_route_canary_raw_reader',BASE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def record(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def compare(b,child,left,right,claimed):
    """Recompute all 129 original tensors; verify reports and saved first failure."""
    actual=b.compare_snapshots(left,right)
    wanted={row['name']:row for row in actual['tensors']}
    need(claimed['label']=='A-C' and len(claimed['arrays'])==129
         and len({r['name'] for r in claimed['arrays']})==129
         and {r['name'] for r in claimed['arrays']}==set(wanted),'complete129 contrast rows')
    for row in claimed['arrays']:
        x=wanted[row['name']]
        need(row==dict(check='A-C',name=x['name'],shape=x['shape'],dtype=b.DTYPE[x['encoding']],
             reference_finite=True,candidate_finite=True,bit_equal=x['bit_equal']),
             'reported comparison differs from original bytes')
    need(claimed['numerical_equal'] is actual['bit_equal']
         and claimed['status']==('PASS' if actual['bit_equal'] else 'DIFFERENT'),'contrast summary')
    if actual['bit_equal']:need(claimed['first_difference'] is None,'spurious first difference')
    else:
        first=next(r for r in claimed['arrays'] if not wanted[r['name']]['bit_equal'])
        failure=claimed['first_difference'];name=first['name']
        need(failure['label']=='A-C/'+name and failure['details']=={k:v for k,v in first.items() if k!='check'},
             'first differing tensor record')
        pair=dict(path='comparisons/A-C/'+failure['array_artifact']['path'],
            sha256=failure['array_artifact']['sha256'],finite=True,tensors={
                'reference':{k:v for k,v in left['tensors'][name].items() if k!='offset'},
                'candidate':{k:v for k,v in right['tensors'][name].items() if k!='offset'}})
        saved=b.snapshot_index(child,pair)
        need(b.tensor_bytes(saved,'reference')==b.tensor_bytes(left,name)
             and b.tensor_bytes(saved,'candidate')==b.tensor_bytes(right,name),'first difference raw payload')
    return actual


def prefill_evidence(b,child,complete,parameters):
    raw=(child/'events.jsonl').read_bytes();need(raw.endswith(b'\n'),'closed event lines')
    events=[json.loads(line) for line in raw.splitlines()]
    weights=[x for x in events if x.get('event')=='loaded-weights']
    need(len(weights)==1 and len(weights[0]['checks'])==len(parameters),'independent initial weights receipt')
    names=set()
    for row in weights[0]['checks']:
        need(row['check']=='independent-loaded-weights' and row['name'].startswith('root/'),'weight comparison label')
        name=row['name'][5:];need(name in parameters and name not in names,'complete original parameter names');names.add(name)
        p=parameters[name]
        need(row['shape']==p['shape'] and row['dtype']==p['dtype'] and all(row[k] is True for k in
             ('reference_finite','candidate_finite','bit_equal')),'weight shape/finite/equal receipt')
    expected=[dict(check='verified-prefill-logits',name='root',shape=[1,248320],dtype='mlx.core.bfloat16')]
    for layer in range(64):
        items=(('keys',[1,4,256,256],'BF16'),('values',[1,4,256,256],'BF16')) if layer%4==3 else (
            ('conv_state',[3,10240],'BF16'),('S',[48,128,128],'F32'))
        expected.extend(dict(check='verified-prefill-state',name=f'root/{layer}/{key}',shape=shape,dtype=b.DTYPE[dtype])
                        for key,shape,dtype in items)
    need(complete['prefill_checks']==[dict(**x,reference_finite=True,candidate_finite=True,bit_equal=True) for x in expected],
         'full64layer independently loaded prefill receipt')
    return events,dict(initial_weight_checks=len(parameters),prefill_array_checks=129,
        scope='Device-recorded equality/finite flags; successful initial weight/prefill payloads were not saved and are not independently recomputed.')


def actual_graph(events,route,token):
    rows=[x for x in events if x.get('event')=='graph-call'];need(len(rows)==1,'one actual candidate invocation')
    row=rows[0]
    need(row['phase']=='C-compiled1023' and row['batch']==1 and row['used']==257 and row['positions']==[256]
         and row['lens']==[257] and row['tokens']==[token] and row['KV_backing_lengths']==[[1536,1536]]*16
         and row['status']=='SUBMITTED' and row['invocation']==row['compiled_function_count']==1
         and row['actual_compile_objects']==route['actual_compile_objects'],'actual compiled1023 graph witness')
    objects=route['actual_compile_objects'];need(len(objects)==1,'one cached candidate callable')
    obj=objects[0]
    need(obj['key']==repr((1023,((False,False),)*48)) and obj['type']=='gc_func'
         and type(obj['id']) is int and obj['id']>0,'compiled1023 callable key/type')
    return row


def terminal(complete,equal):
    need(complete['numerical_equal'] is equal and complete['canary_qualified'] is equal,'numerical result matches full raw bytes')
    need(complete['status']==('PASS' if equal else 'FAIL'),'canary hard numerical gate')
    error=complete.get('error')
    if equal:need(not error,'qualified canary has producer error')
    else:need(error and error['type']=='ValueError'
              and error['message']=='route canary original-dtype bit gate rejected','complete numerical rejection, not an execution failure')


def audit(child,plan_sha256):
    child=Path(child).resolve()
    result=dict(schema='E024-route-canary-raw-audit-v1',status='INCOMPLETE',diagnostic_complete=False,
        canary_qualified=False,full_model_qualified=False,performance_qualified=False,
        audit_source=record(__file__),raw_reader_source=record(BASE))
    try:
        b=base();complete=read(child/'complete.json');plan=read(child/'plan.json')
        need(sha(child/'plan.json')==plan_sha256==complete['plan_sha256'],'actual canary plan SHA')
        need(complete['schema']=='E024-route-canary-result-v1','fixed canary result schema')
        result.update(producer_status=complete['status'],producer_error=complete.get('error'),
                      complete_receipt=record(child/'complete.json'),plan_sha256=plan_sha256)
        if complete.get('diagnostic_complete') is not True:
            result.update(reason='Canary execution/invariants incomplete; no numerical qualification.',
                completed_arms=[k for k,v in complete.get('arms',{}).items() if v.get('status')=='COMPLETE'])
            return result
        need(complete['source_model_pins_rechecked'] is True and complete['full_model_qualified'] is False
             and complete['performance_qualified'] is False and not complete.get('cleanup_error'),'complete postchecks and bounded scope')
        need(complete['device_run'] is True and complete['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),
             'actual fixed device/runtime')
        # The producer plan binds exact new source; the immutable shared reader is pinned above.
        need(set(plan)=={'schema','qualifier_sha256','candidate','e024_plan','deadline_s'}
             and plan['schema']=='E024-route-canary-plan-v1' and plan['qualifier_sha256']==complete['qualifier_sha256']==QUALIFIER_SHA==sha(HERE/'qualify.py'),
             'actual canary qualifier source')
        entry=plan['candidate']
        need(set(entry)=={'path','sha256'} and entry['path']==str(HERE/'candidate.py')
             and sha(entry['path'])==entry['sha256']==CANDIDATE_SHA,'actual adjacent route candidate source')
        original_ref=plan['e024_plan'];need(original_ref['sha256']==b.E024_PLAN_SHA==sha(original_ref['path']),'original E024 plan')
        original=read(original_ref['path']);need(read(child/'inherited-e024-plan.json')==original,'saved original plan')
        inherited_ref=original['e023_plan'];need(inherited_ref['sha256']==b.E023_PLAN_SHA==sha(inherited_ref['path']),'inherited E023 plan')
        inherited=read(inherited_ref['path'])
        need(read(child/'inherited-e023-plan.json')==inherited
             and complete['mlx_environment']==inherited['execution']['environment'],'clean inherited MLX environment')
        expected_sources={str(HERE/'qualify.py'):plan['qualifier_sha256'],entry['path']:entry['sha256'],
            str(HERE.parent/'long_decode_isolation/qualify.py'):b.QUALIFIER_SHA,
            str(HERE.parent/'long_decode_graph/qualify.py'):'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
            original['candidate']['path']:original['candidate']['sha256'],
            original['e023_qualifier']['path']:original['e023_qualifier']['sha256'],
            str(HERE.parent/'long_decode_graph/model-source-pins.json'):'7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5'}
        need(complete['sources']==expected_sources and all(sha(path)==digest for path,digest in expected_sources.items()),'exact small executable source receipts')
        model_pins=HERE.parent/'long_decode_graph/model-source-pins.json'
        need(complete['model_source_receipt']==dict(root=str(Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'),
             manifest_sha256=sha(model_pins),files=read(model_pins)['files']),'actual loaded model source receipt')
        contract=complete['contract']
        need(contract['schema']=='E024-route-canary-contract-v1'
             and [contract[k] for k in ('batch','used','prefill_tokens','KV_backing','states','layers')]==[1,257,256,1536,128,64]
             and contract['arms']==[dict(name='A',mode='eager',bound=257),dict(name='C',mode='compiled',bound=1023)]
             and contract['comparisons']==['A-C'] and contract['configuration']==dict(max_len=2048,max_seqs=8,max_batched_tokens=16384,
                 cache_limit_gib=2,compile_decode=False,fuse_decode=False,gates_compile=False,norm_reuse=False)
             and contract['no_fuse_used'] is False and contract['full_matrix_qualified'] is False,'fixed single canary intervention')
        need(set(complete['arms'])=={'A','C'},'exact two canary arms')
        def saved(receipt,kind):
            need(read((child/receipt['path']).with_suffix('.json'))==receipt,'snapshot sidecar binding')
            index=b.snapshot_index(child,receipt);b.layout(index,kind);return index
        canonical=saved(complete['canonical'],'input')
        values={key:struct.unpack('<i',b.tensor_bytes(canonical,'root/'+key))[0] for key in ('tokens','positions','lens')}
        need(values['positions']==256 and values['lens']==257 and 0<=values['tokens']<248320,'actual decode inputs')
        before=b.pinned_json(child,complete['parameters_before']);b.parameter_receipt(before)
        after=b.pinned_json(child,complete['parameters_after_prefill']);b.parameter_receipt(after)
        need(b.parameter_values(before)==b.parameter_values(after),'parameter values changed during prefill')
        outputs={};inputs={};union=0
        for name,mode,bound in (('A','eager',257),('C','compiled',1023)):
            row=complete['arms'][name]
            need(row['status']=='COMPLETE' and row['mode']==mode and row['bound']==bound,'canary intervention')
            first,last=saved(row['input_before'],'input'),saved(row['input_after'],'input')
            need(row['input_before']['metadata']==row['input_after']['metadata']==complete['canonical']['metadata'],'input metadata unchanged')
            need(b.compare_snapshots(canonical,first)['bit_equal'] and b.compare_snapshots(first,last)['bit_equal'],'complete input bits changed')
            need(type(row['input_buffer_bytes']) is int and row['input_buffer_bytes']>0,'input storage receipt')
            union+=row['input_buffer_bytes'];need(row['input_union_bytes']==union,'independent evaluated input buffer receipt')
            after=b.pinned_json(child,row['parameters_after']);b.parameter_receipt(after)
            need(b.parameter_values(before)==b.parameter_values(after),'parameter value receipts changed')
            outputs[name]=saved(row['output'],'output')
            route=row['route'];need(route['max_len']==bound and route['compiled_invocations']==(0 if name=='A' else 1),'route bound/invocation')
            if name=='A':need(route['kind']=='eager','actual eager reference')
            else:need(route['kind']=='route-preserving-E024-candidate','actual new route candidate')
            raw=b.tensor_bytes(outputs[name],'root/logits');words=struct.unpack('<248320H',raw)
            greedy=max(range(len(words)),key=lambda i:struct.unpack('<f',struct.pack('<I',words[i]<<16))[0])
            need(row['greedy_ids']==[greedy],'greedy from raw BF16; not the equality gate')
            inputs[name]=dict(tensors=131,canonical_bit_equal=True,after_bit_equal=True,greedy_ids=[greedy])
        events,prefix=prefill_evidence(b,child,complete,before)
        witness=actual_graph(events,complete['arms']['C']['route'],values['tokens'])
        need(complete['arms']['A']['output']['metadata']==complete['arms']['C']['output']['metadata'],'pair output metadata')
        need(read(child/'comparisons.json')==complete['comparisons'] and len(complete['comparisons'])==1,'one saved complete contrast')
        contrast=compare(b,child,outputs['A'],outputs['C'],complete['comparisons'][0]);equal=contrast['bit_equal']
        terminal(complete,equal)
        result.update(status='PASS',diagnostic_complete=True,numerical_equal=equal,canary_qualified=equal,
            inputs=inputs,contrast=contrast,prefix_evidence=prefix,graph_witness=witness,
            parameter_count=len(before),parameter_evidence_scope='Original-dtype SHA/finite/shape value receipts unchanged before/after prefill and each arm; handle IDs recorded only. Weight payloads were not saved and are not independently rehashed.',
            input_evidence_scope='All131 original tensors per arm equal canonical and their after snapshots. Buffer disjointness is device-reported evidence bound to source, not inferred from safetensors.',
            scope='Only B1/used257 eager257 versus new compiled1023 canary. No40point matrix, boundary/batch transitions, parameter replacement, ahead lifecycle, gates-ON HTTP, SDPA runtime trace or performance qualification.')
    except FileNotFoundError as error:result.update(reason=str(error))
    except Exception as error:result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child',required=True,type=Path);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();need(not args.out.exists(),'fresh audit output')
    result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
