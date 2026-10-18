"""Bounded CPU mechanism tests, explicitly not real-engine qualification."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('route_matrix_cpu',HERE/'qualifier_matrix.py')
q=importlib.util.module_from_spec(spec); spec.loader.exec_module(q)


class MX:
    int32=np.int32
    uint16=np.uint16
    uint32=np.uint32
    def __init__(self): self.events=[]
    def arange(self,n,*,dtype): return np.arange(n,dtype=dtype)
    def take(self,value,index,*,axis):
        self.events.append(('take',id(value))); return np.take(value,index,axis=axis)
    def eval(self,*values): self.events.append(('eval',tuple(id(v) for v in values)))
    def get_array_buffer_size(self,*values):
        roots=[]
        for value in values:
            while isinstance(value.base,np.ndarray): value=value.base
            if all(value is not prior for prior in roots): roots.append(value)
        return sum(v.nbytes for v in roots)
    def array(self,value,*,dtype): return np.array(value,dtype=dtype)
    def contiguous(self,value): return np.ascontiguousarray(value)
    def all(self,value): return np.all(value)
    def isfinite(self,value): return np.isfinite(value)


class Tests(unittest.TestCase):
    def test_fixed52_points_order_boundaries_counts_and_storage(self):
        p=q.point_plan(); contract=q.fixed_plan()
        self.assertEqual(len(p),52); self.assertEqual(len({r['name'] for r in p}),52)
        self.assertEqual([(r['B'],r['lens'][0]) for r in p if r['kind']=='matrix'],
                         [(B,n) for n in q.USED for B in range(1,9)])
        self.assertEqual([r['lens'] for r in p if r['kind']=='uniform-chain'],[[1023],[1024],[1025]])
        self.assertEqual([r['B'] for r in p if r['kind']=='batch-reuse'],[1,8,1,1,8,1])
        mixed=[r for r in p if r['kind']=='mixed-chain']
        self.assertEqual([r['lens'] for r in mixed],[[x+i for x in q.MIXED] for i in range(3)])
        self.assertEqual(mixed[1]['name'],'mixed-used1024')
        self.assertEqual(sum(r['B'] for r in p),227)
        self.assertEqual(8*1535+sum(x-1 for x in q.MIXED),18730)
        self.assertEqual(sum(1 if value=='canonical' else 3 for value in q.RAW_POINTS.values()),10)
        self.assertFalse(contract['performance_qualified']); self.assertFalse(contract['scheduler_lifecycle_qualified'])

    def test_identity_gather_exact_bits_independent_and_eager_before_next(self):
        mx=MX()
        source={'a':np.array([0.,-0.,1.,-2.],np.float32).reshape(2,2),
                'b':np.arange(16,dtype=np.uint16).reshape(2,8)[:,::2]}
        result=q.clone_arrays(mx,source)
        for key in source:
            self.assertEqual(result[key].tobytes(),source[key].tobytes())
            self.assertFalse(np.shares_memory(result[key],source[key]))
        self.assertEqual([r[0] for r in mx.events],['take','eval','take','eval'])
        result['a'][0,0]=9
        self.assertEqual(source['a'][0,0],0)

    def test_accidental_alias_copy_rejects(self):
        mx=MX(); mx.take=lambda value,index,axis:value[:]
        with self.assertRaisesRegex(ValueError,'independent storage'): q.clone_arrays(mx,{'a':np.ones((2,2),np.float32)})

    def test_chain_uses_each_arms_own_greedy_not_reference_substitution(self):
        contexts=[NS(caches=object()),NS(caches=object())]; old=[c.caches for c in contexts]
        # Intentionally differing mock IDs prove routing; real step comparison
        # would reject the previous differing logits before reaching this helper.
        ids=q.advance_chain(MX(),contexts,[[7,8],[9,10]],[258,1024])
        self.assertEqual([x.tolist() for x in ids],[[7,8],[9,10]])
        for ctx,cache in zip(contexts,old):
            self.assertIs(ctx.caches,cache); self.assertEqual(ctx.positions.tolist(),[257,1023])
            self.assertEqual(ctx.lens.tolist(),[258,1024]); self.assertEqual(ctx.max_len,1024)
        with self.assertRaises(ValueError): q.advance_chain(MX(),contexts,[[7]],[1024])

    def test_original_dtype_hashes_and_nonfinite_reject(self):
        arrays={'root/f':np.array([0.,-0.,1.],np.float32),'root/i':np.array([256,1023],np.int32)}
        result=q.hashes(MX(),dict(metadata={'exact':True},arrays=arrays),lambda:None)
        for key,value in arrays.items(): self.assertEqual(result['arrays'][key]['sha256'],hashlib.sha256(value.tobytes()).hexdigest())
        self.assertEqual(result['metadata'],{'exact':True})
        with self.assertRaisesRegex(ValueError,'nonfinite'): q.hashes(MX(),dict(metadata={},arrays={'bad':np.array([np.nan],np.float32)}),lambda:None)

    def test_actual_frozen_helpers_validate_cpu_only_and_exact_candidate(self):
        helper=q.helper(); h=helper.helpers()
        plan=dict(schema='E024-route-matrix-plan-v1',qualifier_sha256=h.sha(q.__file__),
            candidate=dict(path=str(HERE/'candidate.py'),sha256=q.CANDIDATE_SHA),
            e024_plan=dict(path=str(HERE.parent/'long_decode_graph/prereg-v1/qualification-plan.json'),sha256=helper.OLD_PLAN_SHA),deadline_s=1800)
        self.assertEqual(len(q.validate(plan)),7)
        with self.assertRaisesRegex(ValueError,'frozen route candidate'):
            q.validate({**plan,'candidate':{**plan['candidate'],'sha256':'0'*64}})
        with self.assertRaisesRegex(ValueError,'exact matrix plan'): q.validate({**plan,'extra':True})

    def test_no_tolerance_or_difference_recorder_and_cancellation_does_not_dump(self):
        source=(HERE/'qualifier_matrix.py').read_text(); tree=ast.parse(source)
        self.assertNotIn('compare_finite_outputs',source)
        self.assertNotIn('allclose',source)
        point=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='point')
        handler=next(n for n in ast.walk(point) if isinstance(n,ast.ExceptHandler))
        branch=next(n for n in handler.body if isinstance(n,ast.If))
        self.assertIn('InterruptedError',ast.unparse(branch.test)); self.assertIn('TimeoutError',ast.unparse(branch.test))
        self.assertNotIn('dump_failure',ast.unparse(branch.body)); self.assertIn('dump_failure',ast.unparse(branch.orelse))
        self.assertTrue(any(isinstance(n,ast.Raise) for n in handler.body))

    def test_point_failure_keeps_available_eager_output_and_stops(self):
        # Execute the actual nested point body with small explicit mocks. This
        # checks exception ordering only, not MLX, model arithmetic or storage.
        tree=ast.parse((HERE/'qualifier_matrix.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='point')
        code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual-point-body>','exec')
        for failure in ('candidate-execution','numeric-bit-difference'):
            with self.subTest(failure=failure):
                emitted=[]; saved=[]; calls=[]; result={'points':[]}
                def need(ok,message):
                    if not ok: raise ValueError(message)
                class Comparator:
                    def __init__(self,*args): self.rows=[]
                    def meta(self,*args): pass
                    def compare(self,label,*args):
                        if label.endswith('/outputs'):
                            self.rows.append({'bit_equal':False})
                            raise ValueError('original-bit-difference')
                def snapshot(value):
                    return {'metadata':{},'arrays':{str(i):None for i in range(129 if 'logits' in value else 131)}}
                def reference(*args): calls.append('A'); return 'eager-logits'
                def candidate(*args):
                    calls.append('C')
                    if failure=='candidate-execution': raise RuntimeError('compile-failed')
                    return 'candidate-logits'
                def retain(name,row,inputs,outputs,canonical): saved.append((list(outputs),len(inputs)))
                namespace=dict(dict=dict,zip=zip,len=len,all=all,max=max,sum=sum,BaseException=BaseException,
                    InterruptedError=InterruptedError,TimeoutError=TimeoutError,type=type,str=str,
                    traceback=__import__('traceback'),deadline=lambda:None,phase=lambda name:None,need=need,
                    support=NS(Comparator=Comparator,snapshot=snapshot),out=HERE,RAW_POINTS={},
                    mx=NS(eval=lambda *a:None,get_array_buffer_size=lambda *a:sum(len(x) for x in a)),
                    hashes=lambda *a:{},witness=candidate,engines=[NS(model=reference)],
                    dump_failure=retain,emit=emitted.append,result=result,write=lambda *a:None)
                exec(code,namespace)
                contexts=[NS(positions=np.array([256]),lens=np.array([257]),max_len=257,caches=[]) for _ in range(2)]
                with self.assertRaises((RuntimeError,ValueError)):
                    namespace['point']({'name':'fixture','lens':[257]},contexts,[object(),object()],{})
                self.assertEqual(calls,['A','C'])
                self.assertEqual(saved,[(['A'] if failure=='candidate-execution' else ['A','C'],2)])
                self.assertEqual(len(emitted),1); self.assertEqual(emitted[0]['status'],'FAIL')
                self.assertEqual(result['points'],[{'name':'fixture','status':'FAIL','raw':{}}])

    def test_failure_payload_labels_do_not_claim_historical_before_bytes(self):
        tree=ast.parse((HERE/'qualifier_matrix.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='dump_failure')
        code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual-failure-body>','exec')
        saved=[]
        def dump(mx,out,path,snapshot,deadline):
            saved.append((path,snapshot)); return {'path':path}
        namespace=dict(h=NS(dump=dump),mx=None,out=HERE,deadline=lambda:None)
        exec(code,namespace)
        row={'canonical_provenance':'reference-input-view-after-prior-exact-step'}
        canonical,source_a,source_c=object(),object(),object()
        namespace['dump_failure']('fixture',row,[source_a,source_c],{'A':object()},canonical)
        self.assertEqual(list(row['failure_raw']),['canonical-handles-at-failure','A-input-handles-at-failure',
                                                 'C-input-handles-at-failure','A-output'])
        self.assertIn('not historical before bytes',row['failure_raw_scope']['input_payload'])
        self.assertIn('before either arm executes',row['failure_raw_scope']['genuine_before_evidence'])
        self.assertEqual(row['failure_raw_scope']['canonical_provenance'],row['canonical_provenance'])
        self.assertFalse(any('input-before' in path or 'input-after' in path for path,_ in saved))
        self.assertEqual([snap for _,snap in saved[:3]],[canonical,source_a,source_c])

    def test_help_blocks_device_imports(self):
        code='''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('mlx','numpy','tutorial_code'):raise RuntimeError('forbidden '+name)
sys.meta_path.insert(0,Block());sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
'''
        p=subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE/'qualifier_matrix.py')],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr); self.assertIn('--plan',p.stdout)


if __name__=='__main__': unittest.main()
