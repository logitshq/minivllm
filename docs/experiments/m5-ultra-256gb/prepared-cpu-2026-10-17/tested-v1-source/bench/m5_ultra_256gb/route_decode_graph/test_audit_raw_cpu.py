"""Tiny original-byte and witness regressions; no MLX, model or services."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.route_decode_graph import audit_raw as a
from bench.m5_ultra_256gb.long_decode_isolation.test_audit_raw_cpu import save


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.b=a.base()

    def pair(self,different=False):
        tensors={key:(dtype,[2],struct.pack('<2H',0x4000,0x3f80) if dtype=='BF16' else struct.pack('<2I',0x3f800000,0))
                 for key,(shape,dtype) in self.b.expected_layout('output').items()}
        other=copy.deepcopy(tensors)
        if different:other['root/state/4/S']=('F32',[2],struct.pack('<2I',0x3f800001,0))
        left=self.b.snapshot_index(self.root,save(self.root,'a.safetensors',tensors))
        right=self.b.snapshot_index(self.root,save(self.root,'c.safetensors',other))
        rows=[dict(check='A-C',name=key,shape=[2],dtype=self.b.DTYPE[dtype],reference_finite=True,
                   candidate_finite=True,bit_equal=tensors[key]==other[key]) for key,(dtype,shape,raw) in tensors.items()]
        first=None
        if different:
            key='root/state/4/S';folder=self.root/'comparisons/A-C';folder.mkdir(parents=True)
            pair=save(folder,'failure-arrays.safetensors',{'reference':tensors[key],'candidate':other[key]})
            row=next(r for r in rows if r['name']==key)
            first=dict(label='A-C/'+key,details={k:v for k,v in row.items() if k!='check'},
                       array_artifact=dict(path=pair['path'],sha256=pair['sha256']))
        claimed=dict(label='A-C',status='DIFFERENT' if different else 'PASS',numerical_equal=not different,
                     arrays=rows,first_difference=first)
        return left,right,claimed

    def test_all129_original_arrays_equal(self):
        result=a.compare(self.b,self.root,*self.pair())
        self.assertTrue(result['bit_equal']);self.assertEqual(result['tensor_count'],129)

    def test_one_f32_bit_rejects_despite_identical_logits_greedy(self):
        left,right,claimed=self.pair(True);result=a.compare(self.b,self.root,left,right,claimed)
        self.assertFalse(result['bit_equal']);self.assertEqual(result['differing_tensors'],1)
        self.assertEqual(result['differing_words'],1)
        self.assertEqual(self.b.tensor_bytes(left,'root/logits'),self.b.tensor_bytes(right,'root/logits'))

    def test_missing_state_or_false_equal_flag_rejected(self):
        left,right,claimed=self.pair(True)
        for mutate in ('missing','equal','dtype','first'):
            bad=copy.deepcopy(claimed)
            if mutate=='missing':bad['arrays'].pop()
            elif mutate=='equal':bad['numerical_equal']=True
            elif mutate=='dtype':bad['arrays'][0]['dtype']='mlx.core.float32'
            else:bad['first_difference']['array_artifact']['sha256']='0'*64
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):a.compare(self.b,self.root,left,right,bad)

    def test_nonfinite_even_updated_digest_rejected(self):
        receipt=save(self.root,'bad.safetensors',{'x':('F32',[1],struct.pack('<I',0x7fc00000))})
        with self.assertRaises(ValueError):self.b.snapshot_index(self.root,receipt)

    def test_actual_graph_cannot_fallback_or_use_old1536_bound(self):
        objects=[dict(key=repr((1023,((False,False),)*48)),type='gc_func',id=4)]
        route=dict(actual_compile_objects=objects)
        event=dict(event='graph-call',phase='C-compiled1023',batch=1,used=257,positions=[256],lens=[257],
            tokens=[524],KV_backing_lengths=[[1536,1536]]*16,status='SUBMITTED',invocation=1,
            compiled_function_count=1,actual_compile_objects=objects)
        self.assertEqual(a.actual_graph([event],route,524),event)
        for change in ('missing','twice','oldbound','token','backing','fallback'):
            bad=copy.deepcopy(event);r=copy.deepcopy(route);events=[bad]
            if change=='missing':events=[]
            elif change=='twice':events=[bad,bad]
            elif change=='oldbound':bad['actual_compile_objects'][0]['key']=repr((1536,((False,False),)*48))
            elif change=='token':bad['tokens']=[525]
            elif change=='backing':bad['KV_backing_lengths'][0][0]=1023
            else:bad['invocation']=0
            with self.subTest(change=change),self.assertRaises(ValueError):a.actual_graph(events,r,524)

    def test_parameter_handle_record_is_not_a_value_gate(self):
        before={'x':dict(shape=[1],dtype='mlx.core.float32',bytes=4,sha256='1'*64,finite=True,handle_id=1)}
        after=copy.deepcopy(before);after['x']['handle_id']=2
        self.assertEqual(self.b.parameter_values(before),self.b.parameter_values(after))
        after['x']['sha256']='2'*64
        self.assertNotEqual(self.b.parameter_values(before),self.b.parameter_values(after))

    def test_complete_numerical_rejection_is_auditable_but_not_qualified(self):
        row=dict(numerical_equal=False,canary_qualified=False,status='FAIL',
                 error=dict(type='ValueError',message='route canary original-dtype bit gate rejected'))
        a.terminal(row,False)
        for change in ('qualified','status','execution_error','missing_error'):
            bad=copy.deepcopy(row)
            if change=='qualified':bad['canary_qualified']=True
            elif change=='status':bad['status']='PASS'
            elif change=='execution_error':bad['error']['type']='TimeoutError'
            else:bad.pop('error')
            with self.subTest(change=change),self.assertRaises(ValueError):a.terminal(bad,False)
        a.terminal(dict(numerical_equal=True,canary_qualified=True,status='PASS'),True)

    def test_incomplete_failure_retains_reason_and_never_reads_tensors(self):
        (self.root/'plan.json').write_text('{}')
        (self.root/'complete.json').write_text(json.dumps(dict(schema='E024-route-canary-result-v1',
            plan_sha256=a.sha(self.root/'plan.json'),status='FAIL',diagnostic_complete=False,
            error=dict(type='TimeoutError',message='stop'),arms={'A':dict(status='COMPLETE')})))
        with patch.object(self.b,'snapshot_index',side_effect=AssertionError('must not read missing raw')),patch.object(a,'base',return_value=self.b):
            result=a.audit(self.root,a.sha(self.root/'plan.json'))
        self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['canary_qualified'])
        self.assertEqual(result['completed_arms'],['A']);self.assertEqual(result['producer_error']['type'],'TimeoutError')


if __name__=='__main__':unittest.main()
