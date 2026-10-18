"""Source/dispatch mechanisms only; no MLX import or numerical/device claim."""
import ast
import builtins
import gc
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from bench.m5_ultra_256gb.gdn_simd_scalars import candidate as c

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'tutorial_code/miniengine/kernels/gdn_metal.py'


def original_source():
    tree = ast.parse(BASE.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_SOURCE' for t in n.targets))


class Array:
    def __init__(self, shape, dtype): self.shape, self.dtype = shape, dtype
    def astype(self, dtype):
        return self if dtype == self.dtype else Array(self.shape, dtype)


class MX:
    bfloat16, float32 = 'bf16', 'fp32'
    def __init__(self):
        self.fast = SimpleNamespace(metal_kernel=self.factory)
        self.factories, self.dispatches = [], []
        self.fail = False
    def factory(self, **kwargs):
        self.factories.append(kwargs)
        def kernel(**call):
            if self.fail: raise RuntimeError('fake compiler/dispatch rejection')
            self.dispatches.append(call)
            return tuple(Array(s, d) for s, d in zip(call['output_shapes'], call['output_dtypes']))
        return kernel


def arrays(B=1):
    return [Array((B,48,128),'bf16') for _ in range(3)] + [Array((B,48),'fp32') for _ in range(2)] + [Array((B,48,128,128),'fp32')]


class Tests(unittest.TestCase):
    def setUp(self):
        self.mx = MX(); self.fallback_calls = []; self.fallback_result = object()
        def fallback(*args):
            self.fallback_calls.append(args); return self.fallback_result
        self.base = SimpleNamespace(_SOURCE=original_source(), SPLIT=4, VALUE_TILE=32, gated_delta_step_metal=fallback)

    def test_import_default_off_no_backend_kernel_or_input_touch(self):
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'mlx' or name.startswith('mlx.'): raise AssertionError('MLX import')
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded): runpy.run_path(c.__file__)
        step = c.make_step(self.mx, self.base, lambda: self.fail('OFF probed backend'))
        args = [object() for _ in range(6)]
        self.assertIs(step(*args), self.fallback_result)
        self.assertEqual(self.fallback_calls[0], tuple(args)); self.assertEqual(self.mx.factories, [])

    def test_only_scalar_prologue_changes_not_recurrence_or_reduction(self):
        original = original_source()
        self.assertEqual(original.count(c.ORIGINAL_SCALARS), 1)
        self.assertEqual(c.SOURCE.count(c.SHARED_SCALARS), 1)
        self.assertEqual(c.SOURCE.replace(c.SHARED_SCALARS, c.ORIGINAL_SCALARS), original)
        self.assertEqual(c.SOURCE.count('threadgroup_barrier'), original.count('threadgroup_barrier'))
        # Every broadcast executes after reconvergence; no dot-product tree is introduced.
        self.assertEqual(c.SHARED_SCALARS.count('simd_broadcast('), 4)
        self.assertNotIn('simd_sum', c.SOURCE)

    def test_shape_dtype_backend_fallback_keeps_input_handles(self):
        variants = [arrays(8), arrays(), arrays(), arrays()]
        variants[1][0].dtype='fp32'; variants[2][5].dtype='bf16'; variants[3][4].shape=(1,47)
        for args in variants:
            step=c.make_step(self.mx,self.base,lambda: True,enabled=True)
            self.assertIs(step(*args),self.fallback_result)
            self.assertTrue(all(a is b for a,b in zip(args,self.fallback_calls[-1])))
        step=c.make_step(self.mx,self.base,lambda: False,enabled=True)
        self.assertIs(step(*arrays()),self.fallback_result); self.assertEqual(self.mx.factories,[])

    def test_explicit_fresh_state_outputs_and_original_geometry_no_eval(self):
        step=c.make_step(self.mx,self.base,lambda: True,enabled=True)
        first,second=arrays(),arrays(); out1=step(*first); out2=step(*second)
        self.assertEqual(step.calls,dict(candidate=2,fallback=0)); self.assertEqual(len(self.mx.factories),1)
        self.assertEqual(set(self.mx.factories[0]),{'name','input_names','output_names','source'})
        for call,args,out in zip(self.mx.dispatches,(first,second),(out1,out2)):
            self.assertIs(call['inputs'][-1],args[-1]); self.assertEqual(call['grid'],(128,192,1))
            self.assertEqual(call['threadgroup'],(32,4,1))
            self.assertTrue(all(x.dtype=='fp32' for x in call['inputs']))
            self.assertTrue(all(x is not args[-1] for x in out))
        self.assertEqual(self.fallback_calls,[])

    def test_reject_changed_comparator_and_nonbool_enable(self):
        for field,value in (('_SOURCE',self.base._SOURCE+' '),('SPLIT',8),('VALUE_TILE',16)):
            base=SimpleNamespace(**vars(self.base)); setattr(base,field,value)
            with self.assertRaises(ValueError): c.make_step(self.mx,base,lambda:True,enabled=True)
        for flag in (1,None,'on'):
            with self.assertRaises(TypeError): c.make_step(self.mx,self.base,lambda:True,enabled=flag)

    def test_no_silent_kernel_failure_or_cached_input_state(self):
        step=c.make_step(self.mx,self.base,lambda:True,enabled=True); self.mx.fail=True
        with self.assertRaisesRegex(RuntimeError,'rejection'):step(*arrays())
        self.assertEqual(step.calls,dict(candidate=0,fallback=0)); self.assertEqual(self.fallback_calls,[])
        self.mx.fail=False; args=arrays(); witness=weakref.ref(args[-1]); step(*args)
        self.mx.dispatches.clear(); del args; gc.collect()
        self.assertIsNone(witness())


if __name__=='__main__': unittest.main()
