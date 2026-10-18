"""Synthetic receipt/corruption tests only; never model or device qualification."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('matrix_audit_cpu', HERE/'audit_matrix.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def descriptor(shape, dtype, digest='a'*64):
    return dict(shape=shape, dtype=a.DTYPES[dtype], bytes=a.math.prod(shape)*a.WIDTH[dtype], sha256=digest, finite=True)


def snapshot(b, kind):
    return dict(metadata=a.metadata(kind), arrays={name: descriptor(shape, dtype) for name, (shape, dtype) in a.layout(b, kind).items()})


def row_for(spec, previous=None):
    before = snapshot(spec['B'], 'input'); output = snapshot(spec['B'], 'output')
    for field, values in (('tokens', [5]*spec['B']), ('positions', [x-1 for x in spec['lens']]), ('lens', spec['lens'])):
        before['arrays']['root/'+field]['sha256'] = a.int_hash(values)
    continuation = previous is not None and previous['kind'] == spec['kind'] and spec['kind'].endswith('-chain')
    if continuation:
        before['arrays']['root/tokens']['sha256'] = a.int_hash(previous['greedy_ids'][0])
        before['arrays'].update({k: copy.deepcopy(v) for k,v in previous['outputs']['A']['arrays'].items() if k.startswith('root/state/')})
    raw = {key: {} for key in ([] if spec['name'] not in a.RAW_POINTS else ['canonical'] if a.RAW_POINTS[spec['name']] == 'canonical' else ['canonical', 'A-output', 'C-output'])}
    return dict(**spec, status='PASS', canonical_provenance='reference-input-view-after-prior-exact-step' if continuation else 'independent-prefill-rebuild-storage',
        candidate_bound=1023 if max(spec['lens']) < 1024 else 1536, input_buffer_bytes=[100,100],
        input_before={'A': before, 'C': copy.deepcopy(before)}, input_after={'A': copy.deepcopy(before), 'C': copy.deepcopy(before)},
        outputs={'A': output, 'C': copy.deepcopy(output)}, greedy_ids=[[5]*spec['B'], [5]*spec['B']], raw=raw,
        checks=[dict(name=k, check=spec['name']+'/outputs', shape=s, dtype=a.DTYPES[d], reference_finite=True,
                     candidate_finite=True, bit_equal=True) for k,(s,d) in a.layout(spec['B'],'output').items()])


def graph(spec, invocation, objects):
    bound = 1023 if max(spec['lens']) < 1024 else 1536
    objects = copy.deepcopy(objects)
    key = repr((bound, ((False,False),)*48))
    if not any(x['key'] == key for x in objects): objects.append(dict(key=key,type='gc_func',id=len(objects)+100))
    return dict(event='graph-call', phase=spec['name'], batch=spec['B'], used=max(spec['lens']),
        positions=[n-1 for n in spec['lens']], lens=spec['lens'], tokens=[5]*spec['B'],
        KV_backing_lengths=[[1536,1536]]*16, status='SUBMITTED', invocation=invocation,
        compiled_function_count=len(objects),actual_compile_objects=objects)


def prefill(stage, number, begin, end):
    label = f'prefill-{end}-{number}' if stage == 'matrix' else f'mixed-prefill-{number}'
    checks = [dict(check=label+'/logits', name='root', shape=[1,248320], dtype=a.DTYPES['BF16'])]
    for layer in range(64):
        entries = [('keys',[1,4,((end+255)//256)*256,256],'BF16'), ('values',[1,4,((end+255)//256)*256,256],'BF16')] if layer%4 == 3 else [('conv_state',[3,10240],'BF16'),('S',[48,128,128],'F32')]
        checks.extend(dict(check=label+'/state',name=f'root/{layer}/{name}',shape=shape,dtype=a.DTYPES[dtype]) for name,shape,dtype in entries)
    for row in checks: row.update(reference_finite=True,candidate_finite=True,bit_equal=True)
    return dict(event='prefill',stage=stage,row=number,begin=begin,end=end,checks=checks)


def ledger():
    result=[]; previous=None; objects=[]; invocation=0; recorded=[]
    for event,value in a.event_plan():
        if event == 'memory': row=dict(event=event,phase=value,active_bytes=0,cache_bytes=0,peak_bytes=0,observation_not_perf=True)
        elif event == 'loaded-weights': row=dict(event=event,checks=[])
        elif event == 'prefill': row=prefill(*value)
        elif event == 'graph-call':
            invocation+=1; row=graph(value,invocation,objects); objects=row['actual_compile_objects']
        else:
            row=dict(event='point',**row_for(value,previous)); previous=row; recorded.append({k:row[k] for k in ('name','status','raw')})
        row['elapsed_wall_s']=float(len(result)); result.append(row)
    complete=dict(status='PASS',points=recorded,actual_graph_calls=52,compiled_objects=objects)
    return result,complete


class Tests(unittest.TestCase):
    def test_all52_exact_order_scopes_and_actual_callable_families(self):
        points=a.points(); self.assertEqual(len(points),52)
        self.assertEqual(sum(p['B'] for p in points),227)
        self.assertEqual(sum(p['kind']=='matrix' for p in points),40)
        objects=[]
        for i,p in enumerate(points,1):
            r=graph(p,i,objects); objects=a.graph_call(r,p,i,objects)
        self.assertEqual([a.ast.literal_eval(o['key'])[0] for o in objects],[1023,1536])
        self.assertEqual(len([x for x in a.event_plan() if x[0]=='prefill']),48)

    def test_actual_b8_shapes_and_all_original_flags_are_required(self):
        p=a.points()[7]; row=row_for(p); a.verify_point(row,p,None)
        for mutate in (lambda r:r['outputs']['C']['arrays'].pop('root/state/63/V'),
                       lambda r:r['checks'][1].update(bit_equal=1),
                       lambda r:r['outputs']['C']['arrays']['root/state/0/S'].update(shape=[1,48,128,128]),
                       lambda r:r['input_before']['C']['arrays']['root/positions'].update(sha256='0'*64)):
            bad=copy.deepcopy(row); mutate(bad)
            with self.assertRaises(ValueError): a.verify_point(bad,p,None)

    def test_chain_own_state_and_tokens_not_only_cross_arm_equality(self):
        specs=[p for p in a.points() if p['kind']=='mixed-chain']; first=row_for(specs[0]); second=row_for(specs[1],first)
        a.verify_point(second,specs[1],first)
        for field in ('root/tokens','root/state/63/V'):
            bad=copy.deepcopy(second)
            for section in ('input_before','input_after'):
                for arm in ('A','C'): bad[section][arm]['arrays'][field]['sha256']='b'*64
            with self.assertRaisesRegex(ValueError,'chain'): a.verify_point(bad,specs[1],first)
        self.assertEqual(first['outputs']['A']['arrays']['root/state/63/V']['sha256'],'a'*64)

    def test_graph_fallback_flags_wrong_bound_or_handle_replacement_rejected(self):
        p=a.points()[0]; row=graph(p,1,[]); objects=a.graph_call(row,p,1,[])
        nextrow=graph(a.points()[1],2,objects)
        for mutate in (lambda r:r.update(status='RUNNING'),lambda r:r.update(invocation=1),
                       lambda r:r['KV_backing_lengths'].__setitem__(0,[256,256]),
                       lambda r:r['actual_compile_objects'][0].update(id=900),
                       lambda r:r['actual_compile_objects'][0].update(key=repr((1023,((True,False),)*48)))):
            bad=copy.deepcopy(nextrow);mutate(bad)
            with self.assertRaises(ValueError):a.graph_call(bad,a.points()[1],2,objects)

    def test_prefill_real_single_cache_shapes_and_checkpoint_roundup(self):
        row=prefill('matrix',2,1022,1023);a.prefill_checks(row,'matrix',2,1022,1023)
        bad=copy.deepcopy(row);next(x for x in bad['checks'] if x['name']=='root/3/keys')['shape'][2]=1536
        with self.assertRaises(ValueError):a.prefill_checks(bad,'matrix',2,1022,1023)
        row['begin']=0
        with self.assertRaises(ValueError):a.prefill_checks(row,'matrix',2,1022,1023)

    def test_stream_full_synthetic_contract_and_extra_event_rejection(self):
        events,complete=ledger()
        def fake_raw(base,child,row):
            n=len(row['raw']);return dict(snapshots=n,bytes=0,paired_output_tensors=129 if n==3 else 0)
        with tempfile.TemporaryDirectory() as d, patch.object(a,'raw_point',side_effect=fake_raw):
            path=Path(d)/'events.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in events))
            result={};self.assertTrue(a.scan(Path(d),complete,{},None,result))
            self.assertEqual((result['point_count'],result['device_output_checks'],result['prefill_checks'],result['raw_snapshots']), (52,6708,6192,10))
            with path.open('a') as f:f.write(json.dumps(events[-1])+'\n')
            with self.assertRaises(ValueError):a.scan(Path(d),complete,{},None,{})

    def test_reuse_corruption_rejected_even_when_both_arms_match(self):
        events,complete=ledger()
        row=next(r for r in events if r['event']=='point' and r['name']=='reuse-used1023-0-B1')
        for section in ('input_before','input_after'):
            for arm in ('A','C'):row[section][arm]['arrays']['root/state/0/S']['sha256']='b'*64
        with tempfile.TemporaryDirectory() as d, patch.object(a,'raw_point',return_value=dict(snapshots=0,bytes=0,paired_output_tensors=0)):
            path=Path(d)/'events.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in events))
            with self.assertRaisesRegex(ValueError,'reuse must use original'):a.scan(Path(d),complete,{},None,{})

    def test_actual_pinned_small_source_and_plan_identity_producer_layout(self):
        # Read only immutable small manifests/core sources; never model payloads.
        oldref=dict(path=str(HERE.parent/'long_decode_graph/prereg-v1/qualification-plan.json'),sha256=a.E024_SHA)
        old=a.read(oldref['path']); inherited=a.read(old['e023_plan']['path'])
        fixture=a.read(inherited['fixtures']['path']); source=a.read(inherited['source_manifest']['path'])
        paths=[HERE/'qualifier_matrix.py',HERE/'candidate.py',HERE/'qualify.py',HERE.parent/'long_decode_isolation/qualify.py',
               HERE.parent/'long_decode_graph/qualify.py',HERE.parent/'long_decode_graph/candidate.py',
               HERE.parent/'pareto_realistic/qualify_native.py',HERE.parent/'long_decode_graph/model-source-pins.json']
        plan=dict(schema='E024-route-matrix-plan-v1',qualifier_sha256=a.QUALIFIER_SHA,
            candidate=dict(path=str(HERE/'candidate.py'),sha256=a.CANDIDATE_SHA),e024_plan=oldref,deadline_s=1800)
        complete=dict(qualifier_sha256=a.QUALIFIER_SHA,sources={str(p):a.sha(p) for p in paths},
            model_source_receipt=dict(root=str((Path(inherited['package_root'])/'tutorial_code/miniengine').resolve()),manifest_sha256=a.PINS_SHA,files=source['files']),
            mlx_environment=inherited['execution']['environment'])
        diagnostic=[]
        for f in fixture['fixtures']:
            ids=f['prompt_ids']; diagnostic.append({**f,'fixture_id':'diagnostic-repeat-'+f['fixture_id'],
                'prompt_ids':(ids*((1535+len(ids)-1)//len(ids)))[:1535], 'source_fixture_id':f['fixture_id'],
                'diagnostic_only':True,'normal_eos':True,'max_output_tokens':4})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,data in (('plan.json',plan),('inherited-e024-plan.json',old),('inherited-e023-plan.json',inherited),('diagnostic-inputs.json',diagnostic)):
                (root/name).write_text(json.dumps(data))
            complete['plan_sha256']=a.sha(root/'plan.json')
            a.identities(root,complete,complete['plan_sha256'])
            complete['sources'][str(HERE/'candidate.py')]='0'*64
            with self.assertRaisesRegex(ValueError,'source pins'):a.identities(root,complete,complete['plan_sha256'])

    def test_partial_first_failure_stays_rejected_and_future_events_forbidden(self):
        first=a.points()[0];fail=dict(event='point',**first,status='FAIL',raw={},checks=[],error={'message':'compile failure'},elapsed_wall_s=0.)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.jsonl';path.write_text(json.dumps(fail)+'\n')
            result={};self.assertFalse(a.scan(Path(d),{'status':'FAIL'},None,None,result));self.assertEqual(result['point_count'],0)
            self.assertEqual(result['first_failed_point']['name'],first['name'])
            path.write_text(json.dumps(fail)+'\n'+json.dumps(fail)+'\n')
            with self.assertRaisesRegex(ValueError,'after first failure'):a.scan(Path(d),{'status':'FAIL'},None,None,{})

    def test_original_raw_bit_reader_finite_and_descriptor_binding(self):
        # Small real binary fixture exercises full-byte reader, not model math.
        base=a.raw_reader()
        with tempfile.TemporaryDirectory() as d:
            child=Path(d);relative='raw/tiny/canonical.safetensors';path=child/relative;path.parent.mkdir(parents=True)
            payload=struct.pack('<HH',0,0x8000)+struct.pack('<I',0x3f800000)+struct.pack('<i',257)
            header={'root/a':dict(dtype='BF16',shape=[2],data_offsets=[0,4]),
                    'root/b':dict(dtype='F32',shape=[1],data_offsets=[4,8]),'root/lens':dict(dtype='I32',shape=[1],data_offsets=[8,12])}
            encoded=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
            receipt=dict(path=relative,sha256=a.sha(path),finite=True,metadata={})
            receipt['tensors']={k:dict(shape=v['shape'],dtype=a.DTYPES[v['dtype']],encoding=v['dtype'],bytes=v['data_offsets'][1]-v['data_offsets'][0],
                sha256=hashlib.sha256(payload[slice(*v['data_offsets'])]).hexdigest(),finite=True) for k,v in header.items()}
            index=base.snapshot_index(child,receipt);self.assertEqual(base.tensor_bytes(index,'root/a'),struct.pack('<HH',0,0x8000))
            bad=copy.deepcopy(receipt);bad['tensors']['root/a']['sha256']='f'*64
            with self.assertRaises(ValueError):base.snapshot_index(child,bad)
            path.write_bytes(struct.pack('<Q',len(encoded))+encoded+struct.pack('<HH',0,0x7fc0)+payload[4:])
            bad=copy.deepcopy(receipt);bad['sha256']=a.sha(path);bad['tensors']['root/a']['sha256']=hashlib.sha256(struct.pack('<HH',0,0x7fc0)).hexdigest()
            with self.assertRaisesRegex(ValueError,'finite'):base.snapshot_index(child,bad)

    def test_final_producer_source_frozen_and_no_gpu_import_on_help(self):
        self.assertEqual(a.sha(HERE/'qualifier_matrix.py'),a.QUALIFIER_SHA)
        self.assertNotIn('mlx.core',sys.modules)
        run=subprocess.run([sys.executable,'-B',str(HERE/'audit_matrix.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_missing_completion_returns_incomplete_never_qualified(self):
        with tempfile.TemporaryDirectory() as d:
            result=a.audit(d,'a'*64)
        self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['matrix_qualified']);self.assertFalse(result['performance_qualified'])


if __name__=='__main__':unittest.main()
