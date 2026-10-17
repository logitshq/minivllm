"""Small synthetic timing-ledger checks, never device qualification."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('timing_audit_cpu',HERE/'audit_timing.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
m=a.support()


def sample(arm,index,warm,b=1,ms=4.):
    return dict(event='sample',name='fixture',arm=arm,index=index,warm=warm,status='PASS',forward_ms=ms,
        teardown_ms=1.,owned_lifecycle_ms=ms+1.,diagnostic_tokens_per_second=b*1000/ms)

def checks(label,kind,b):
    return [dict(name=name,check=label,shape=shape,dtype=m.DTYPES[dtype],reference_finite=True,candidate_finite=True,bit_equal=True)
            for name,(shape,dtype) in m.layout(b,kind).items()]

def snapshot(b,kind,used):
    arrays={name:dict(shape=shape,dtype=m.DTYPES[dtype],bytes=m.math.prod(shape)*m.WIDTH[dtype],sha256='a'*64,finite=True)
            for name,(shape,dtype) in m.layout(b,kind).items()}
    if kind=='input':
        for key,values in (('tokens',[5]*b),('lens',[used]*b),('positions',[used-1]*b)):arrays['root/'+key]['sha256']=m.int_hash(values)
    return dict(metadata=m.metadata(kind),arrays=arrays)

def fixture():
    rows=[];configs=[];samples=[]
    def emit(row):rows.append(dict(row,elapsed_wall_s=float(len(rows))))
    def memory(name):emit(dict(event='memory',phase=name,active=0,cache=0,peak=0))
    memory('before-load-reference');memory('before-load-candidate');emit(dict(event='loaded-weights',checks=[]))
    begin=0
    for used in (257,1023,1024):
        end=used-1
        for i in range(8):
            label=f'prefill-{end}-{i}';c=[dict(check=label+'/logits',name='root',shape=[1,248320],dtype=m.DTYPES['BF16'])]
            for layer in range(64):
                entries=(('keys',[1,4,((end+255)//256)*256,256],'BF16'),('values',[1,4,((end+255)//256)*256,256],'BF16')) if layer%4==3 else (('conv_state',[3,10240],'BF16'),('S',[48,128,128],'F32'))
                c.extend(dict(check=label+'/state',name=f'root/{layer}/{name}',shape=shape,dtype=m.DTYPES[dtype]) for name,shape,dtype in entries)
            for item in c:item.update(reference_finite=True,candidate_finite=True,bit_equal=True)
            emit(dict(event='prefill',row=i,begin=begin,end=end,checks=c))
        begin=end
        if used==257:continue
        for b in (1,8):
            canonical=snapshot(b,'input',used);output=snapshot(b,'output',used);emit(dict(event='canonical',B=b,used=used,hashes=canonical))
            for enabled in (False,True):
                name=f'B{b}-used{used}-gates'+('ON' if enabled else 'OFF');memory('before-'+name)
                for arm in ('A','C'):
                    c=checks(name+'/'+arm+'/canonical','input',b)
                    if enabled or arm=='C':c+=checks(name+'/'+arm+'/OFF-eager-oracle','output',b)
                    route=dict(kind='eager',bound=used) if arm=='A' else dict(invocations=1,key=repr((1023 if used<1024 else 1536,((enabled,False),)*48)),bound=1023 if used<1024 else 1536)
                    emit(dict(event='qualification',name=name,arm=arm,status='PASS',input_before=canonical,input_after=canonical,output=output,route=route,checks=c))
                for warm,order in ((True,'ACCA'),(False,'ACCA'*3)):
                    for index,arm in enumerate(order):
                        row=sample(arm,index,warm,b,4. if arm=='A' else 2.);row['name']=name
                        measured={k:v for k,v in row.items() if k not in ('teardown_ms','owned_lifecycle_ms')}
                        emit(dict(measured,event='sample-measured',status='RUNNING'))
                        emit(dict(row,checks=checks(name+'/sample-output','output',b)+checks(name+'/sample-input-unchanged','input',b)));samples.append(row)
                configs.append(dict(name=name,status='PASS',B=b,used=used,gates=enabled,qualification_output_hashes=dict(A=output,C=output)))
                memory('after-'+name)
    return rows,dict(configs=configs,samples=samples,elapsed_wall_s=len(rows)+1.)


class Tests(unittest.TestCase):
    def test_independent_medians_all_blocks_and_warm_exclusion(self):
        rows=[sample(arm,i,True,ms=100.) for i,arm in enumerate('ACCA')]
        rows += [sample(arm,i,False,ms=8. if arm=='A' else 2.) for i,arm in enumerate('ACCA'*3)]
        result=a.metrics(rows,1)
        self.assertEqual(result['median_forward_speedup'],4.)
        self.assertEqual([r['A_over_C'] for r in result['ACCA_blocks']],[4.]*3)
        self.assertEqual(result['medians']['A']['forward_ms'],8.)
        self.assertEqual(result['medians']['C']['owned_lifecycle_ms'],3.)

    def test_wrong_order_nonfinite_or_forged_derived_metrics_rejected(self):
        rows=[sample(arm,i,warm) for warm,order in ((True,'ACCA'),(False,'ACCA'*3)) for i,arm in enumerate(order)]
        for change in (lambda r:r[4].update(arm='C'),lambda r:r[4].update(warm=1),lambda r:r[4].update(forward_ms=float('nan')),
                       lambda r:r[4].update(owned_lifecycle_ms=10.),lambda r:r[4].update(diagnostic_tokens_per_second=123.)):
            corrupt=copy.deepcopy(rows);change(corrupt)
            with self.assertRaises(ValueError):a.metrics(corrupt,1)

    def test_stream_actual_producer_shapes_and_all128_sample_receipts(self):
        rows,complete=fixture()
        with tempfile.TemporaryDirectory() as d:
            child=Path(d);(child/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            result=a.scan(m,child,complete,{})
        self.assertEqual(len(result['configs']),8)
        self.assertEqual((result['warm_samples'],result['formal_samples'],result['sample_checks']),(32,96,33280))
        self.assertEqual(result['qualification_output_checks'],1548)
        self.assertTrue(all(c['median_forward_speedup']==2. for c in result['configs']))

    def test_on_output_route_bit_flag_and_measured_binding_corruptions(self):
        source,complete=fixture()
        changes=[lambda r:next(x for x in r if x['event']=='qualification' and x['name'].endswith('ON'))['route'].update(bound=1536),
                 lambda r:next(x for x in r if x['event']=='sample')['checks'][0].update(bit_equal=False),
                 lambda r:next(x for x in r if x['event']=='sample-measured').update(forward_ms=999.),
                 lambda r:next(x for x in r if x['event']=='qualification' and x['name'].endswith('ON'))['output']['arrays']['root/logits'].update(sha256='b'*64)]
        for change in changes:
            rows=json.loads(json.dumps(source));change(rows)
            with tempfile.TemporaryDirectory() as d:
                child=Path(d);(child/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                with self.assertRaises(ValueError):a.scan(m,child,complete,{})

    def test_missing_or_rejected_completion_never_yields_performance(self):
        with tempfile.TemporaryDirectory() as d:
            result=a.audit(d,'0'*64);self.assertEqual(result['status'],'INCOMPLETE')
            (Path(d)/'complete.json').write_text(json.dumps(dict(status='FAIL')))
            result=a.audit(d,'0'*64);self.assertEqual(result['status'],'FAIL')
        for key in ('performance_qualified','http_qualified','pareto_eligible','diagnostic_complete'):self.assertIs(result[key],False)

    def test_frozen_sources_and_accepted_matrix_receipt(self):
        self.assertEqual(a.sha(HERE/'qualifier_timing.py'),a.TIMING_SHA)
        self.assertEqual(a.sha(HERE/'audit_matrix.py'),a.SUPPORT_SHA)
        self.assertEqual(a.sha(a.DEFAULT_PROOF),a.MATRIX_PROOF_SHA)
        self.assertNotIn('mlx.core',__import__('sys').modules)


if __name__=='__main__':unittest.main()
