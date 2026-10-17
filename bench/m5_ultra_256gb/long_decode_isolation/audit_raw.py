"""CPU-only byte audit for the fixed E024 four-arm dispatch diagnostic.

The saved arrays, not reported equality flags, determine each contrast. Floating
point comparison is original unsigned bits with a separate finite check. Array
shape/storage checks follow the archived E028 reader; only the three explicit
I32 context arrays extend its BF16/F32 domain. This module never imports MLX.
"""
import argparse
from array import array
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[3]
READER=ROOT/'docs/experiments/m5-ultra-256gb/E028-identity-refill-v3/verify.py'
READER_SHA='afeaa81256fd607ad17d0c4fa91a4d952fc447b3f5291233ae34dc391e99470f'
QUALIFIER_SHA='9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e'
E024_PLAN_SHA='5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
E023_PLAN_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
ARMS=('A','B','C','D')
CONTRASTS=(('A','B'),('A','C'),('B','D'),('A','D'))
WIDTH={'BF16':2,'F32':4,'I32':4}
DTYPE={'BF16':'mlx.core.bfloat16','F32':'mlx.core.float32','I32':'mlx.core.int32'}


def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def reader():
    need(sha(READER)==READER_SHA,'frozen original-dtype reader changed')
    spec=importlib.util.spec_from_file_location('_e024_original_dtype_reader',READER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def record(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def snapshot_index(root,receipt):
    """One bounded tensor at a time; validate every byte, no model-size retention."""
    root=Path(root).resolve();relative=Path(receipt['path']);path=(root/relative).resolve()
    need(not relative.is_absolute() and '..' not in relative.parts and path.is_relative_to(root),'snapshot path escapes run')
    need(receipt['finite'] is True and sha(path)==receipt['sha256'],'snapshot SHA/finite declaration')
    base=reader();result={};spans=[]
    with path.open('rb') as stream:
        prefix=stream.read(8);need(len(prefix)==8,'safetensors prefix')
        size=struct.unpack('<Q',prefix)[0]
        need(0<size<=1<<20 and 8+size<=path.stat().st_size,'safetensors header bounds')
        header=json.loads(stream.read(size),object_pairs_hook=base.unique)
        payload=8+size;body_size=path.stat().st_size-payload
        for key,item in header.items():
            if key=='__metadata__':
                need(item is None or isinstance(item,dict),'safetensors metadata');continue
            need(set(item)=={'dtype','shape','data_offsets'},'tensor header fields')
            dtype,shape,offsets=item['dtype'],item['shape'],item['data_offsets']
            need(dtype in WIDTH and isinstance(shape,list) and all(type(n) is int and n>0 for n in shape),'original dtype/shape')
            need(isinstance(offsets,list) and len(offsets)==2 and all(type(n) is int for n in offsets),'tensor offset types')
            lo,hi=offsets;need(0<=lo<=hi<=body_size and hi-lo==WIDTH[dtype]*math.prod(shape),'tensor payload bounds/shape')
            stream.seek(payload+lo);raw=stream.read(hi-lo);need(len(raw)==hi-lo,'truncated tensor')
            finite=True if dtype=='I32' else base.finite(raw,dtype)
            meta=dict(shape=shape,dtype=DTYPE[dtype],encoding=dtype,bytes=hi-lo,sha256=hashlib.sha256(raw).hexdigest(),finite=finite)
            need(finite and receipt['tensors'].get(key)==meta,'original tensor SHA/finite/metadata: '+key)
            result[key]=dict(**meta,offset=payload+lo);spans.append((lo,hi))
        cursor=0
        for lo,hi in sorted(spans):need(lo==cursor,'tensor gap/overlap');cursor=hi
        need(cursor==body_size and set(result)==set(receipt['tensors']),'trailing bytes/missing tensor')
    return dict(path=str(path),sha256=receipt['sha256'],tensors=result)


def expected_layout(kind):
    result={}
    for i in range(64):
        items=(('K',[1,4,1536,256],'BF16'),('V',[1,4,1536,256],'BF16')) if i%4==3 else (
            ('conv',[1,3,10240],'BF16'),('S',[1,48,128,128],'F32'))
        result.update({f'root/state/{i}/{name}':(shape,dtype) for name,shape,dtype in items})
    if kind=='output':result['root/logits']=([1,248320],'BF16')
    elif kind=='input':result.update({f'root/{key}':([1],'I32') for key in ('tokens','positions','lens')})
    else:raise ValueError('fixed input/output kind')
    return result


def layout(index,kind):
    expected=expected_layout(kind);need(set(index['tensors'])==set(expected),'fixed complete tensor key set')
    for key,(shape,dtype) in expected.items():
        item=index['tensors'][key]
        need(item['shape']==shape and item['encoding']==dtype,'fixed tensor shape/dtype: '+key)


def tensor_bytes(index,key):
    meta=index['tensors'][key]
    with Path(index['path']).open('rb') as stream:
        stream.seek(meta['offset']);raw=stream.read(meta['bytes'])
    need(len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256'],'tensor changed after indexing')
    return raw


def compare_words(left,right,dtype,shape):
    """All words, first exact difference, with original float storage preserved."""
    width=WIDTH[dtype];need(len(left)==len(right)==width*math.prod(shape),'pair bytes/shape')
    if left==right:return dict(bit_equal=True,elements=math.prod(shape),differing_words=0,differing_bytes=0,first_difference=None)
    a,b=array('H' if width==2 else 'I'),array('H' if width==2 else 'I')
    a.frombytes(left);b.frombytes(right)
    if sys.byteorder!='little':a.byteswap();b.byteswap()
    different=sum(x!=y for x,y in zip(a,b));first=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
    first_byte=next(i for i,(x,y) in enumerate(zip(left,right)) if x!=y)
    coordinates=[];n=first
    for dim in reversed(shape):coordinates.insert(0,n%dim);n//=dim
    evidence=dict(element_index=first,coordinates=coordinates,byte_offset=first_byte,
        left_word_hex=f'{a[first]:0{width*2}x}',right_word_hex=f'{b[first]:0{width*2}x}',
        xor_word_hex=f'{a[first]^b[first]:0{width*2}x}')
    return dict(bit_equal=False,elements=len(a),differing_words=different,
        differing_bytes=sum(x!=y for x,y in zip(left,right)),first_difference=evidence)


def compare_snapshots(left,right):
    need(set(left['tensors'])==set(right['tensors']),'pair tensor keys')
    result=[]
    for key,a in left['tensors'].items():
        b=right['tensors'][key]
        need(a['shape']==b['shape'] and a['encoding']==b['encoding'],'pair original shape/dtype')
        result.append(dict(name=key,shape=a['shape'],encoding=a['encoding'],
            **compare_words(tensor_bytes(left,key),tensor_bytes(right,key),a['encoding'],a['shape'])))
    return dict(bit_equal=all(row['bit_equal'] for row in result),tensor_count=len(result),
        differing_tensors=sum(not row['bit_equal'] for row in result),
        differing_words=sum(row['differing_words'] for row in result),
        differing_bytes=sum(row['differing_bytes'] for row in result),tensors=result)


def pinned_json(root,item):
    relative=Path(item['path']);path=(root/relative).resolve()
    need(not relative.is_absolute() and '..' not in relative.parts and path.is_relative_to(root),'metadata path')
    need(set(item)=={'path','sha256'} and sha(path)==item['sha256'],'metadata SHA')
    return read(path)


def parameter_receipt(data):
    need(isinstance(data,dict) and len(data)==1349,'complete original parameter set')
    for key,row in data.items():
        need(isinstance(key,str) and set(row)=={'shape','dtype','bytes','sha256','finite','handle_id'},'parameter metadata')
        widths={'mlx.core.bfloat16':2,'mlx.core.float32':4,'mlx.core.uint32':4,'mlx.core.uint8':1}
        need(row['dtype'] in widths and isinstance(row['shape'],list)
             and all(type(x) is int and x>0 for x in row['shape'])
             and type(row['bytes']) is int and row['bytes']==math.prod(row['shape'])*widths[row['dtype']]
             and row['finite'] is True and type(row['handle_id']) is int and row['handle_id']>0
             and isinstance(row['sha256'],str) and len(row['sha256'])==64,'parameter shape/finite/hash')
        need(len(bytes.fromhex(row['sha256']))==32,'parameter SHA encoding')


def parameter_values(data):
    return {name:{k:v for k,v in row.items() if k!='handle_id'} for name,row in data.items()}


def prefix_evidence(child,complete,parameters,input_values):
    raw=(child/'events.jsonl').read_bytes();need(raw.endswith(b'\n'),'closed event lines')
    events=[json.loads(line) for line in raw.splitlines()]
    weights=[x for x in events if x.get('event')=='loaded-weights']
    need(len(weights)==1 and len(weights[0]['checks'])==len(parameters),'independent initial weights receipt')
    names=set()
    for row in weights[0]['checks']:
        need(row['check']=='independent-loaded-weights' and row['name'].startswith('root/'),'initial weight labels')
        name=row['name'][5:];need(name in parameters and name not in names,'exact initial parameter names');names.add(name)
        p=parameters[name]
        need(row['shape']==p['shape'] and row['dtype']==p['dtype'] and all(row[k] is True for k in
             ('reference_finite','candidate_finite','bit_equal')),'initial parameter shape/finite/equal flags')
    expected=[dict(check='verified-prefill-logits',name='root',shape=[1,248320],dtype='mlx.core.bfloat16')]
    for layer in range(64):
        items=(('keys',[1,4,256,256],'BF16'),('values',[1,4,256,256],'BF16')) if layer%4==3 else (
            ('conv_state',[3,10240],'BF16'),('S',[48,128,128],'F32'))
        expected.extend(dict(check='verified-prefill-state',name=f'root/{layer}/{key}',shape=shape,dtype=DTYPE[dtype]) for key,shape,dtype in items)
    need(complete['prefill_checks']==[dict(**x,reference_finite=True,candidate_finite=True,bit_equal=True) for x in expected],
         'all64layers original independent prefill flags')
    graph=[x for x in events if x.get('event')=='graph-call'];need(len(graph)==1,'one actual original candidate invocation')
    row=graph[0]
    need(row['phase']=='D-compiled1536' and row['batch']==1 and row['used']==257 and row['positions']==[256]
         and row['lens']==[257] and row['tokens']==[input_values['tokens']] and row['KV_backing_lengths']==[[1536,1536]]*16
         and row['status']=='SUBMITTED' and row['invocation']==row['compiled_function_count']==1
         and row['actual_compile_objects']==complete['arms']['D']['route']['actual_compile_objects'],'original graph witness')
    return dict(initial_weight_checks=len(parameters),prefill_array_checks=129,graph_witness=row,
        scope='Recorded device equality/finite checks and actual compiled-call witness; successful prefill/weight payloads not independently recomputed.')


def audit(child,plan_sha256):
    child=Path(child).resolve()
    result=dict(schema='E024-dispatch-isolation-raw-audit-v1',status='INCOMPLETE',diagnostic_complete=False,
        candidate_qualified=False,full_model_qualified=False,performance_qualified=False,
        audit_source=record(Path(__file__)),reader_source=record(READER),contrasts={})
    try:
        complete=read(child/'complete.json');plan=read(child/'plan.json')
        need(sha(child/'plan.json')==plan_sha256==complete['plan_sha256'],'actual isolation plan SHA')
        need(complete['schema']=='E024-dispatch-isolation-result-v1','fixed result schema')
        result['producer_status']=complete['status']
        if complete['status']!='PASS':
            result['reason']='Producer did not complete all four finite arms and invariants.'
            result['completed_arms']=[k for k,v in complete.get('arms',{}).items() if v.get('status')=='COMPLETE']
            return result
        need(plan['schema']=='E024-dispatch-isolation-plan-v1'
             and complete['qualifier_sha256']==plan['qualifier_sha256']==QUALIFIER_SHA==sha(Path(__file__).with_name('qualify.py')),
             'actual diagnostic source pin')
        original_ref=plan['e024_plan']
        need(original_ref['sha256']==E024_PLAN_SHA==sha(original_ref['path']),'original E024 plan source')
        original=read(original_ref['path'])
        need(read(child/'inherited-e024-plan.json')==original,'saved original E024 plan')
        inherited_ref=original['e023_plan']
        need(inherited_ref['sha256']==E023_PLAN_SHA==sha(inherited_ref['path']),'inherited E023 plan source')
        inherited=read(inherited_ref['path'])
        need(read(child/'inherited-e023-plan.json')==inherited
             and complete['mlx_environment']==inherited['execution']['environment'],
             'saved inherited plan and clean MLX environment receipt')
        need(complete['device_run'] is True and complete['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),
             'fixed device/runtime receipt')
        need(complete['diagnostic_complete'] is True and complete['source_model_pins_rechecked'] is True
             and complete['full_model_qualified'] is False and complete['performance_qualified'] is False
             and not complete.get('error') and not complete.get('cleanup_error'),'complete bounded diagnostic only')
        contract=complete['contract']
        need(contract['schema']=='E024-dispatch-isolation-contract-v1'
             and [contract[k] for k in ('batch','used','prefill_tokens','KV_backing','states','layers')]==[1,257,256,1536,128,64]
             and contract['arms']==[dict(name=n,mode=m,bound=b) for n,m,b in
                 (('A','eager',257),('B','eager',1536),('C','compiled',257),('D','compiled',1536))]
             and contract['comparisons']==['A-B','A-C','B-D','A-D']
             and contract['configuration']==dict(max_len=2048,max_seqs=8,max_batched_tokens=16384,
                 cache_limit_gib=2,compile_decode=False,fuse_decode=False,gates_compile=False,norm_reuse=False)
             and contract['no_fuse_used'] is False,'fixed four-arm intervention contract')
        need(set(complete['arms'])==set(ARMS),'complete four arms')
        def saved(receipt,kind):
            need(read((child/receipt['path']).with_suffix('.json'))==receipt,'snapshot sidecar binding')
            index=snapshot_index(child,receipt);layout(index,kind);return index
        canonical=saved(complete['canonical'],'input')
        input_values={key:struct.unpack('<i',tensor_bytes(canonical,'root/'+key))[0] for key in ('tokens','positions','lens')}
        need(input_values['positions']==256 and input_values['lens']==257 and 0<=input_values['tokens']<248320,'actual decode input')
        before=pinned_json(child,complete['parameters_before']);parameter_receipt(before)
        after_prefill=pinned_json(child,complete['parameters_after_prefill']);parameter_receipt(after_prefill)
        need(parameter_values(after_prefill)==parameter_values(before),'original parameter bytes changed during prefill')
        outputs={};inputs={};union=0
        for name in ARMS:
            row=complete['arms'][name];mode='eager' if name in ('A','B') else 'compiled';bound=257 if name in ('A','C') else 1536
            need(row['status']=='COMPLETE' and row['mode']==mode and row['bound']==bound,'arm intervention')
            first,last=saved(row['input_before'],'input'),saved(row['input_after'],'input')
            need(row['input_before']['metadata']==row['input_after']['metadata']==complete['canonical']['metadata'],'input metadata unchanged')
            canonical_equal=compare_snapshots(canonical,first);unchanged=compare_snapshots(first,last)
            need(canonical_equal['bit_equal'] and unchanged['bit_equal'],'actual complete input bits changed')
            need(type(row['input_buffer_bytes']) is int and row['input_buffer_bytes']>0,'input storage receipt')
            union+=row['input_buffer_bytes'];need(row['input_union_bytes']==union,'four retained arm input buffers disjoint receipt')
            after=pinned_json(child,row['parameters_after']);parameter_receipt(after)
            need(parameter_values(after)==parameter_values(before),'original parameter bytes changed')
            output=saved(row['output'],'output');outputs[name]=output
            route=row['route']
            need(route['max_len']==bound and route['compiled_invocations']==(0 if mode=='eager' else 1),'actual route count/bound')
            if mode=='eager':need(route['kind']=='eager','eager route')
            elif name=='C':need(route['kind']=='explicit-state-mx.compile' and route['callable_type']=='gc_func','actual compiled257 callable')
            else:
                need(route['kind']=='original-E024-candidate' and len(route['actual_compile_objects'])==1,'original candidate actual callable')
                obj=route['actual_compile_objects'][0]
                need(obj['key']==repr((1536,((False,False),)*48)) and obj['type']=='gc_func'
                     and type(obj['id']) is int and obj['id']>0,'compiled1536 witness')
            words=array('H');words.frombytes(tensor_bytes(output,'root/logits'))
            if sys.byteorder!='little':words.byteswap()
            greedy=max(range(len(words)),key=lambda i:struct.unpack('<f',struct.pack('<I',words[i]<<16))[0])
            need(row['greedy_ids']==[greedy],'greedy ID independently decoded from original BF16 logits')
            inputs[name]=dict(canonical_bit_equal=True,after_bit_equal=True,tensors=131,greedy_ids=[greedy],route=route)
        prefix=prefix_evidence(child,complete,before,input_values)
        need(read(child/'comparisons.json')==complete['comparisons'] and [r['label'] for r in complete['comparisons']]==['A-B','A-C','B-D','A-D'],
             'all four reported comparisons')
        for (left,right),claimed in zip(CONTRASTS,complete['comparisons']):
            label=left+'-'+right;actual=compare_snapshots(outputs[left],outputs[right]);result['contrasts'][label]=actual
            need(complete['arms'][left]['output']['metadata']==complete['arms'][right]['output']['metadata'],'output metadata pair')
            wanted={r['name']:r for r in actual['tensors']}
            need(len(claimed['arrays'])==129 and len({r['name'] for r in claimed['arrays']})==129
                 and {r['name'] for r in claimed['arrays']}==set(wanted),'all reported output tensors')
            for r in claimed['arrays']:
                x=wanted[r['name']]
                need(r==dict(check=label,name=x['name'],shape=x['shape'],dtype=DTYPE[x['encoding']],
                     reference_finite=True,candidate_finite=True,bit_equal=x['bit_equal']),'reported original-dtype comparison')
            need(claimed['numerical_equal'] is actual['bit_equal'] and claimed['status']==('PASS' if actual['bit_equal'] else 'DIFFERENT'),
                 'reported contrast agrees with raw bytes')
            if actual['bit_equal']:need(claimed['first_difference'] is None,'spurious first difference')
            else:
                first=next(r for r in claimed['arrays'] if not wanted[r['name']]['bit_equal']);failure=claimed['first_difference']
                need(failure['label']==label+'/'+first['name'] and failure['details']=={k:v for k,v in first.items() if k!='check'},'first differing tensor record')
                name=first['name'];pair= dict(path='comparisons/'+label+'/'+failure['array_artifact']['path'],
                    sha256=failure['array_artifact']['sha256'],finite=True,tensors={
                        'reference':{k:v for k,v in outputs[left]['tensors'][name].items() if k!='offset'},
                        'candidate':{k:v for k,v in outputs[right]['tensors'][name].items() if k!='offset'}})
                index=snapshot_index(child,pair)
                need(tensor_bytes(index,'reference')==tensor_bytes(outputs[left],name)
                     and tensor_bytes(index,'candidate')==tensor_bytes(outputs[right],name),'saved first-difference payload matches full output')
        equal={k:v['bit_equal'] for k,v in result['contrasts'].items()}
        need(complete['numerical_equal'] is all(equal.values()),'all-pair summary matches independent raw audit')
        result.update(status='PASS',diagnostic_complete=True,numerical_equal=all(equal.values()),inputs=inputs,prefix_evidence=prefix,
            complete_receipt=record(child/'complete.json'),plan_sha256=plan_sha256,
            inherited_plan_sha256=E023_PLAN_SHA,mlx_environment=complete['mlx_environment'],
            parameters_after_prefill=complete['parameters_after_prefill'],
            parameter_count=len(before),parameter_evidence_scope='All original-dtype SHA/finite/shape receipts unchanged; handles are recorded and may change. Parameter payloads were not saved and are not independently rehashed here.',
            input_evidence_scope='All131 saved input tensors per arm independently equal canonical and own after snapshot. Buffer disjointness is a source-pinned device receipt, not inferred from packed safetensors.',
            factor_findings=dict(bound_changes_bits=not equal['A-B'],compile_at257_changes_bits=not equal['A-C'],
                compile_at1536_changes_bits=not equal['B-D'],original_rejection_reproduced=not equal['A-D'],
                bound_alone_sufficient_at_this_point=not equal['A-B'] and equal['B-D']),
            scope='Only fixed B1/used257 four-arm numerical separation. No full matrix, parameter replacement, lifecycle, SDPA runtime kernel trace, owner/resource/restoration or performance qualification.')
    except FileNotFoundError as error:result.update(reason=str(error))
    except Exception as error:result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();need(not args.out.exists(),'fresh audit output')
    result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
