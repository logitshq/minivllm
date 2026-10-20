"""Closed actual metadata regression; no model/raw tensor reads or backend."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bench.m5_ultra_256gb.rowwise_native_verify_cost_analysis_v2 import audit as a

R=Path('/Users/ryanli/code/minivllm')
COMPLETE=R/'.artifacts/m5-ultra-256gb/c1-verifier-cost-v1-numerical/complete.json'
COMPLETE_SHA='b719a1c25257e0b0fc93e80a8d49215e223d02f708ab5f879558ffe94968e9de'
SUPPORT=R/'bench/m5_ultra_256gb/pareto_realistic/qualify_native.py'
SUPPORT_SHA='26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'


def actual():
    if a.base.sha(COMPLETE)!=COMPLETE_SHA:raise AssertionError('actual closed producer metadata changed')
    value=a.base.read(COMPLETE)
    if value['status']!='PASS_LOCAL_COST_ONLY':raise AssertionError('actual producer not completed')
    return {key:value[key] for key in ('checks','post_checks')}


class Tests(unittest.TestCase):
    def test_actual_closed_receipts_fail_v1_and_pass_v2(self):
        value=actual();self.assertEqual((len(value['checks']),len(value['post_checks'])),(2719,655))
        with self.assertRaisesRegex(ValueError,'initial comparisons'):a.base.producer_checks(value)
        a.producer_checks(value)
        self.assertEqual(sum(r['check'].startswith('T') for r in value['checks']),2064)

    def test_original_snapshot_ast_explains_exact_double_prefix(self):
        self.assertEqual(a.base.sha(SUPPORT),SUPPORT_SHA)
        node=next(n for n in ast.parse(SUPPORT.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='snapshot')
        scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(SUPPORT),'exec'),scope)
        class Array:
            dtype='BF16';shape=(1,);ndim=1
            def view(self,*args):return self
            def __getitem__(self,key):return self
        value=Array(); snapshot=scope['snapshot']
        self.assertEqual(set(snapshot({'root/tokens':value,'root/state/62/S':value})['arrays']),{'root/root/tokens','root/root/state/62/S'})
        self.assertEqual(set(snapshot({'logits':value,'state':[{'S':value}]})['arrays']),{'root/logits','root/state/0/S'})

    def test_no_actual_record_rewriting_or_scope_count_change(self):
        value=actual();before=copy.deepcopy(value);a.producer_checks(value);self.assertEqual(value,before)
        for label in ('canonical-preserved','post-timing-canonical'):
            rows=[r for group in value.values() for r in group if r['check'].startswith(label)]
            self.assertEqual(len(rows),655);self.assertTrue(all(r['name'].startswith('root/root/') for r in rows))
        self.assertEqual(a.contract()['tensor_pairs'],2064);self.assertEqual(a.contract()['timing_count'],400)

    def test_single_triple_or_wrong_output_prefix_rejected(self):
        for field,index,mode in (('checks',2064,'single'),('post_checks',0,'triple'),('checks',0,'output')):
            value=actual();row=value[field][index]
            row['name']=row['name'][5:] if mode=='single' else 'root/'+row['name']
            with self.subTest(mode=mode),self.assertRaises(ValueError):a.producer_checks(value)

    def test_missing_duplicate_shape_finite_or_bit_failure_still_rejected(self):
        for mode in ('missing','duplicate','shape','dtype','finite','bits'):
            value=actual();rows=value['post_checks']
            if mode=='missing':rows.pop()
            elif mode=='duplicate':rows[-1]=copy.deepcopy(rows[0])
            elif mode=='shape':rows[-1]['shape']=[1]
            elif mode=='dtype':rows[-1]['dtype']='mlx.core.bfloat16' if rows[-1]['dtype']!='mlx.core.bfloat16' else 'mlx.core.float32'
            elif mode=='finite':rows[-1]['reference_finite']=False
            else:rows[-1]['bit_equal']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):a.producer_checks(value)

    def test_private_rebind_keeps_raw_timing_and_closure_functions_exact(self):
        self.assertIs(a._frozen_audit.__code__,a.base.audit.__code__)
        for key in ('raw_reader','compare_pair','check_layout','qualification','timing_ledger','closure'):
            self.assertIs(a._frozen_audit.__globals__[key],getattr(a.base,key))
        self.assertIs(a._frozen_audit.__globals__['producer_checks'],a.producer_checks)
        self.assertIsNot(a.base.audit.__globals__['producer_checks'],a.producer_checks)
        self.assertEqual(a._frozen_audit.__globals__['__file__'],a.__file__)
        self.assertEqual(a.base.sha(a.PREVIOUS),a.PREVIOUS_SHA)

    def test_default_remains_inert_and_no_backend(self):
        with patch.dict(a._main_scope,audit=lambda *args:(_ for _ in ()).throw(AssertionError('must not audit'))),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(a.main([]),0)
        self.assertEqual(json.loads(output.getvalue())['schema'],'E054-independent-cost-contract-v2')

    def test_fresh_failure_receipt_retains_v2_and_previous_identity(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);out=root/'failed.json'
            self.assertEqual(a.main(['--run','--child',str(root),'--parent-plan',str(root/'missing'),'--parent-plan-sha256','0'*64,'--out',str(out)]),1)
            value=a.base.read(out);self.assertEqual(value['status'],'FAIL');self.assertEqual(value['schema'],'E054-local-cost-independent-v2')
            self.assertEqual(value['previous_auditor']['sha256'],a.PREVIOUS_SHA)
            self.assertEqual(value['audit_source_sha256'],a.base.sha(a.__file__))


if __name__=='__main__':unittest.main()
