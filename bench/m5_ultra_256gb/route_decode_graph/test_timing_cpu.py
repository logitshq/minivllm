"""CPU timing-boundary/route/cleanup tests; no actual model or device proof."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('route_timing_cpu',HERE/'qualifier_timing.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


class Tests(unittest.TestCase):
    def test_fixed_points_warmups_abba_and_unqualified_scope(self):
        p=q.fixed_plan()
        self.assertEqual(q.POINTS,((1,1023),(8,1023),(1,1024),(8,1024)))
        self.assertEqual(p['gate_order'],['off','on'])
        self.assertEqual(p['formal_arms'],list('ACCA')*3)
        self.assertEqual(p['warm_arms'].count('A'),2);self.assertEqual(p['warm_arms'].count('C'),2)
        self.assertEqual(len(p['formal_arms'])*8,96);self.assertEqual(len(p['warm_arms'])*8,32)
        self.assertEqual(p['prefill_checkpoints'],[256,1022,1023])
        for key in ('performance_qualified','full_model_qualified','http_qualified','pareto_eligible'):
            self.assertIs(p[key],False)

    def test_timer_contains_only_forward_all_products_and_completion(self):
        events=[];logits=object();K,V,conv,S=[object() for _ in range(4)]
        ctx=NS(caches=[NS(K=K,V=V),NS(conv=conv,S=S)])
        def clock():
            events.append('clock');return 100 if len(events)==1 else 2_000_100
        def forward(ids,actual):
            self.assertIs(actual,ctx);events.append('forward');return logits
        def evaluate(*arrays):events.append(('eval',arrays))
        mx=NS(eval=evaluate,synchronize=lambda:events.append('sync'))
        returned,ms=q.timed_forward(mx,forward,'tokens',ctx,clock)
        self.assertIs(returned,logits);self.assertEqual(ms,2.)
        self.assertEqual(events,['clock','forward',('eval',(logits,K,V,conv,S)),'sync','clock'])

    def test_timer_failure_never_emits_successful_interval(self):
        for where in ('forward','eval','sync'):
            with self.subTest(where=where):
                calls=[]
                def work(name,value=None):
                    calls.append(name)
                    if where==name:raise RuntimeError(name+' failed')
                    return value
                with self.assertRaisesRegex(RuntimeError,where):
                    q.timed_forward(NS(eval=lambda *a:work('eval'),synchronize=lambda:work('sync')),
                        lambda *a:work('forward',object()),None,NS(caches=[]),lambda:1)
                self.assertEqual(calls,['forward','eval','sync'][:['forward','eval','sync'].index(where)+1])
        with self.assertRaisesRegex(ValueError,'nonpositive'):
            q.timed_forward(NS(eval=lambda *a:None,synchronize=lambda:None),lambda *a:object(),None,NS(caches=[]),lambda:1)

    def test_untimed_witness_records_real_call_and_restores_raw_callables(self):
        key=(1023,((False,False),));raw=lambda *a:'actual'
        class Graph:
            def __init__(self):self._functions={}
            def _eligible(self,*args):return True
            def __call__(self,*args):
                if key not in self._functions:self._functions[key]=raw
                return self._functions[key](*args)
        graph=Graph();value,row=q.witnessed_call(graph,None,None)
        self.assertEqual(value,'actual');self.assertEqual(row,dict(invocations=1,key=repr(key),bound=1023))
        self.assertIs(type(graph._functions),dict);self.assertIs(graph._functions[key],raw)
        value,row=q.witnessed_call(graph,None,None)
        self.assertIs(graph._functions[key],raw);self.assertEqual(row['invocations'],1)

    def test_witness_exception_restores_existing_and_new_raw_functions(self):
        key=(1536,())
        def fail(*args):raise RuntimeError('actual compiled failure')
        class Graph:
            def __init__(self):self._functions={}
            def _eligible(self,*args):return True
            def __call__(self,*args):
                self._functions[key]=fail;return self._functions[key](*args)
        graph=Graph()
        with self.assertRaisesRegex(RuntimeError,'actual compiled failure'):q.witnessed_call(graph,None,None)
        self.assertIs(type(graph._functions),dict);self.assertIs(graph._functions[key],fail)
        graph._eligible=lambda *a:False
        with self.assertRaisesRegex(ValueError,'fallback'):q.witnessed_call(graph,None,None)
        self.assertIs(graph._functions[key],fail)

    def test_real_sample_body_cleans_own_products_without_next_arm(self):
        # Execute the actual small sample body with explicit CPU fakes. This
        # verifies cleanup and failure ordering, not model values or MLX memory.
        tree=ast.parse((HERE/'qualifier_timing.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sample')
        code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual-sample>','exec')
        for fail_check,fail_cleanup in ((False,False),(True,False),(False,True),(True,True)):
            with self.subTest(check=fail_check,cleanup=fail_cleanup):
                events=[];rows=[];result={'samples':[]};counter=[0]
                def clock():counter[0]+=1;return counter[0]*1_000_000
                def sync():
                    events.append('sync')
                    if fail_cleanup and events.count('sync')==2:raise RuntimeError('cleanup failure')
                def compare(label,*args):
                    events.append(label)
                    if fail_check and label.endswith('/sample-output'):raise ValueError('bit failure')
                def fresh(c):events.append('prepare');return {},'ids',NS(caches=[],positions=[],lens=[])
                ns=dict(deadline=lambda:None,fresh=fresh,cmp=NS(compare=compare,rows=[]),
                    support=NS(snapshot=lambda x:x),graph=NS(_functions={},_eligible=lambda *a:True),
                    need=lambda ok,message:None if ok else (_ for _ in ()).throw(ValueError(message)),
                    mx=NS(synchronize=sync),engines=[NS(model=object())],
                    timed_forward=lambda *a:('logits',3.),emit=rows.append,result=result,
                    failure_raw=lambda *a:events.append('failure-raw'),write=lambda *a:events.append('write'),
                    out=HERE,gc=NS(collect=lambda:events.append('gc')),time=NS(perf_counter_ns=clock))
                exec(code,ns)
                args=('fixture','A',0,False,{'arrays':{'root/tokens':[7]}},{})
                if fail_check or fail_cleanup:
                    with self.assertRaisesRegex((ValueError,RuntimeError),'bit failure' if fail_check else 'cleanup failed'):
                        ns['sample'](*args)
                else:ns['sample'](*args)
                self.assertEqual(len(result['samples']),1)
                self.assertEqual(rows[0]['event'],'sample-measured')
                self.assertEqual(rows[-1]['status'],'FAIL' if fail_check or fail_cleanup else 'PASS')
                self.assertEqual(rows[-1]['forward_ms'],3.)
                self.assertIn('gc',events);self.assertEqual(events[-1],'write')
                if fail_cleanup:self.assertIn('cleanup_error',rows[-1])
                if fail_check:self.assertEqual(rows[-1]['error']['message'],'bit failure')

    def test_actual_pinned_helpers_and_validation_are_cpu_safe(self):
        m=q.matrix();helper=m.helper();h=helper.helpers()
        plan=dict(schema='E024-route-timing-plan-v1',qualifier_sha256=h.sha(q.__file__),
            candidate=dict(path=str(HERE/'candidate.py'),sha256=m.CANDIDATE_SHA),
            e024_plan=dict(path=str(HERE.parent/'long_decode_graph/prereg-v1/qualification-plan.json'),sha256=helper.OLD_PLAN_SHA),deadline_s=1800)
        self.assertEqual(len(q.validate(plan)),8)
        with self.assertRaisesRegex(ValueError,'bounded'):q.validate(dict(plan,deadline_s=1801))
        with self.assertRaisesRegex(ValueError,'exact forward'):q.validate(dict(plan,extra=True))

    def test_help_blocks_model_numpy_and_mlx_imports(self):
        code='''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('mlx','numpy','tutorial_code'):raise RuntimeError('forbidden '+name)
sys.meta_path.insert(0,Block());sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
'''
        p=subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE/'qualifier_timing.py')],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('--plan',p.stdout)


if __name__=='__main__':unittest.main()
