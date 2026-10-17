"""Explicit CPU mocks only; none of these tests qualify real model numerics."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; spec.loader.exec_module(module); return module


c=load(HERE/'candidate.py','route_cpu_candidate')
q=load(HERE/'qualify.py','route_cpu_qualify')
base=load(HERE.parent/'long_decode_graph/test_cpu.py','route_original_cpu_fixtures')


def setup(**kwargs):
    model,r,unused,ids,ctx=base.setup(**kwargs)
    with patch.object(c,'_load_runtime',return_value=r): graph=c.RoutePreservingDecode(model,enabled=True)
    return model,r,graph,ids,ctx


class Tests(unittest.TestCase):
    def test_default_off_no_source_runtime_or_backend_touch(self):
        model=base.Model()
        with patch.object(c,'_load_runtime',side_effect=AssertionError('forbidden')):
            graph=c.RoutePreservingDecode(model)
            self.assertEqual(graph(base.Array((1,),'i32',7),base.SimpleNamespace(max_len=257)),(100,7,257))
        self.assertFalse(graph.enabled); self.assertEqual(graph._functions,{})

    def test_bucket_boundaries_and_invalid_types(self):
        for used,expected in ((256,None),(257,1023),(1023,1023),(1024,1536),(1280,1536),(1536,1536),(1537,None),(True,None),(257.0,None)):
            with self.subTest(used=used): self.assertEqual(c.bound_for_used(used),expected)
        for used in (256,257,1023,1024,1280,1536,1537):
            model,r,graph,ids,ctx=setup(used=used)
            self.assertEqual(graph(ids,ctx)[2],c.bound_for_used(used) or used)
            self.assertEqual(bool(r.compiles),257<=used<=1536)
            self.assertEqual(ctx.max_len,used)

    def test_two_callable_families_reuse_across_bound_and_batch_switch(self):
        model,r,graph,ids,ctx=setup(B=1,used=257)
        for B,used in ((1,257),(8,1023),(8,1024),(1,1280),(1,1023),(1,1536)):
            _,_,_,ids,ctx=setup(B=B,used=used)
            self.assertEqual(graph(ids,ctx)[2],c.bound_for_used(used))
        self.assertEqual(len(r.compiles),2)
        self.assertEqual({key[0] for key in graph._functions},{1023,1536})
        self.assertTrue(all(captured is model for _,captured in r.compiles))

    def test_capacity_batch_metadata_and_backend_fallback(self):
        for used,capacity,allowed in ((257,1022,False),(257,1023,False),(1023,1024,False),(257,1536,True),(1024,1535,False),(1024,1536,True)):
            with self.subTest(used=used,capacity=capacity):
                model,r,graph,ids,ctx=setup(used=used,allocated=capacity)
                graph(ids,ctx); self.assertEqual(bool(r.compiles),allowed)
        for B in (0,1,8,9):
            model,r,graph,ids,ctx=setup(B=B); graph(ids,ctx)
            self.assertEqual(bool(r.compiles),1<=B<=8)
        for mutation in (lambda r,ctx:setattr(r,'supported',False),lambda r,ctx:setattr(ctx,'capture',True),
                         lambda r,ctx:setattr(ctx.caches[0].S,'dtype','bf16')):
            model,r,graph,ids,ctx=setup(); mutation(r,ctx); graph(ids,ctx)
            self.assertFalse(r.compiles); self.assertIs(model.calls[-1][1],ctx)

    def test_original_forward_ast_preserved_and_parameter_replacement_visible(self):
        old=ast.parse((HERE.parent/'long_decode_graph/candidate.py').read_text())
        new=ast.parse((HERE/'candidate.py').read_text())
        fn=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward')
        self.assertEqual(ast.dump(fn(old),include_attributes=False),ast.dump(fn(new),include_attributes=False))
        model,r,graph,ids,ctx=setup(used=257)
        self.assertEqual(graph(ids,ctx)[0],100)
        model.weight=base.Array((1,),value=101)
        self.assertEqual(graph(ids,ctx)[0],101); self.assertEqual(len(r.compiles),1)

    def test_compiler_failure_is_not_silent_fallback(self):
        model,r,graph,ids,ctx=setup(used=257)
        r.mx.compile=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('CPU injected compiler failure'))
        with self.assertRaisesRegex(RuntimeError,'compiler failure'): graph(ids,ctx)
        self.assertEqual(model.calls,[])

    def test_canary_contract_exact_plan_and_cpu_only_validation(self):
        h=q.helpers()
        plan=dict(schema='E024-route-canary-plan-v1',qualifier_sha256=h.sha(q.__file__),
            candidate=dict(path=str(HERE/'candidate.py'),sha256=h.sha(HERE/'candidate.py')),
            e024_plan=dict(path=str(HERE.parent/'long_decode_graph/prereg-v1/qualification-plan.json'),sha256=q.OLD_PLAN_SHA),deadline_s=1800)
        result=q.validate(plan); self.assertEqual(len(result),6)
        self.assertEqual(q.fixed_plan()['arms'],[dict(name='A',mode='eager',bound=257),dict(name='C',mode='compiled',bound=1023)])
        with self.assertRaisesRegex(ValueError,'exact route'): q.validate({**plan,'extra':True})
        with self.assertRaisesRegex(ValueError,'bounded'): q.validate({**plan,'deadline_s':True})

    def test_mock_complete_numerical_rejection_stays_fail_with_saved_result(self):
        h=q.helpers()
        plan=dict(schema='E024-route-canary-plan-v1',qualifier_sha256=h.sha(q.__file__),
            candidate=dict(path=str(HERE/'candidate.py'),sha256=h.sha(HERE/'candidate.py')),
            e024_plan=dict(path=str(HERE.parent/'long_decode_graph/prereg-v1/qualification-plan.json'),sha256=q.OLD_PLAN_SHA),deadline_s=1800)
        def mock_device(*args): args[-1].update(diagnostic_complete=True,numerical_equal=False,source_model_pins_rechecked=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json'; h.write(path,plan); out=Path(tmp)/'out'
            with patch.object(q,'device',mock_device):
                self.assertEqual(q.main(['--plan',str(path),'--plan-sha256',h.sha(path),'--out',str(out)]),1)
            result=json.loads((out/'complete.json').read_text())
            self.assertEqual(result['status'],'FAIL'); self.assertFalse(result['canary_qualified'])
            self.assertTrue(result['diagnostic_complete']); self.assertIn('bit gate rejected',result['error']['message'])

    def test_help_has_no_model_or_mlx_import(self):
        code='''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('mlx','numpy','tutorial_code'): raise RuntimeError('forbidden '+name)
sys.meta_path.insert(0,Block());sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
'''
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE/'qualify.py')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr); self.assertIn('--plan',result.stdout)


if __name__=='__main__': unittest.main()
