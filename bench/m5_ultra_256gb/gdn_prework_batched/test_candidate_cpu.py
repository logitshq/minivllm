"""Source-prepared fake contracts, not numerical/device qualification.

No test imports MLX or executes native kernel code. Fake gates record delegation;
their B1-only branch represents the pinned native method, not a compiler proof.
"""
import builtins
import gc
from pathlib import Path
import runpy
import shutil
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from bench.m5_ultra_256gb.gdn_prework_batched import candidate as c


class Array:
    def __init__(self, shape, dtype, value=None):
        self.shape, self.dtype, self.value = shape, dtype, value

    def reshape(self, *shape):
        count = 1
        for n in self.shape: count *= n
        if -1 in shape:
            product = 1
            for n in shape:
                if n != -1: product *= n
            shape = tuple(count // product if n == -1 else n for n in shape)
        product = 1
        for n in shape: product *= n
        if product != count: raise ValueError('fake reshape size differs')
        return Array(shape, self.dtype, self.value)

    def astype(self, dtype):
        return self if dtype == self.dtype else Array(self.shape, dtype, self.value)


class Linear:
    def __init__(self, name, K, N, events):
        self.name, self.in_features, self.out_features, self.events = name, K, N, events
        self.q = dict(c.QUANT)
        self.weight = Array((N, K//8), 'u32', name+'-w')
        self.scales = Array((N, K//16), 'u8', name+'-s')

    def __call__(self, x):
        self.events.append((self.name, self.weight, self.scales, x))
        if x.shape[1] != self.in_features: raise ValueError('wrong native projection input')
        return Array((x.shape[0], self.out_features), 'bf16', (self.weight.value, self.scales.value))


class BatchState:
    def __init__(self, B=1):
        self.conv = Array((B, 3, 10240), 'bf16', 'old-conv')
        self.S = Array((B, 48, 128, 128), 'fp32', 'old-S')


class GDN:
    Hk, Hv, Dk, Dv, conv_dim = 16, 48, 128, 128, 10240

    def __init__(self):
        self.events = []
        self._norm_reuse_enabled = False
        self._gates_compile_enabled = True
        self.conv1d_weight = Array((10240, 4), 'bf16', 'w')
        self.A_log = Array((48,), 'fp32', 'A')
        self.dt_bias = Array((48,), 'fp32', 'dt')
        for name, K, N in (('in_proj_qkv',5120,10240), ('in_proj_z',5120,6144),
                          ('in_proj_b',5120,48), ('in_proj_a',5120,48), ('out_proj',6144,5120)):
            setattr(self, name, Linear(name, K, N, self.events))

    def _decode(self, x, st, **flags):
        self.events.append(('original', x, st, flags))
        return 'original'

    def _gates(self, b, a):
        self.events.append(('gates', self.A_log, self.dt_bias, b, a))
        return Array(b.shape, 'fp32', 'beta'), Array(a.shape, 'fp32', 'g')

    def _decode_gates(self, b, a, x, st):
        self.events.append(('decode-gates', x, st))
        if x.shape == (1, 5120): self.events.append(('fake-compiled-gates',))
        return self._gates(b, a)

    def norm(self, o, z):
        if o.shape != z.shape: raise ValueError('wrong norm batch reshape')
        self.events.append(('norm', o, z))
        return o


class Tests(unittest.TestCase):
    def setUp(self):
        self.layer = GDN()
        self.state = BatchState()
        self.x = Array((1, 5120), 'bf16', 'x')
        self.fused_calls = []
        self.mx = SimpleNamespace(bfloat16='bf16', float32='fp32', uint32='u32', uint8='u8',
                                  gpu='gpu', default_device=lambda: 'gpu')
        self.module = SimpleNamespace(mx=self.mx, GatedDeltaNet=GDN, BatchState=BatchState,
                                      Linear=Linear, _m5_ultra_decode_identity=lambda: True)

    def fused(self, *args):
        raw, conv, w, g, beta, S = args
        B = raw.shape[0]
        self.assertEqual((raw.shape, conv.shape, w.shape, g.shape, beta.shape, S.shape),
                         ((B,10240), (B,3,10240), (10240,4), (B,48), (B,48), (B,48,128,128)))
        self.fused_calls.append(args)
        n = len(self.fused_calls)
        return (Array((B,48,128), 'fp32', ('o',n)),
                Array((B,48,128,128), 'fp32', ('S',n)),
                Array((B,3,10240), 'bf16', ('conv',n)))

    def batch(self, B):
        self.x = Array((B,5120), 'bf16', ('x',B))
        self.state = BatchState(B)

    def call(self, fn, **flags):
        return fn(self.layer, self.x, self.state,
                  **dict({'_compile_gates': True, '_norm_reuse_normal': True}, **flags))

    def test_import_and_off_do_not_load_runtime_or_inspect_install_arguments(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'mlx' or name.startswith('mlx.'): raise AssertionError('runtime import')
            return original(name, *args, **kwargs)
        class Poison:
            def __getattribute__(self, name): raise AssertionError('OFF inspected '+name)
        with patch('builtins.__import__', side_effect=guarded): runpy.run_path(c.__file__)
        with patch.object(c, '_load_fused', side_effect=AssertionError('OFF loaded kernel')):
            self.assertIsNone(c.install(Poison(), Poison()))
        fn = c.make_decode(self.module, self.fused)
        self.x = Poison()
        self.assertEqual(self.call(fn, _compile_gates=False), 'original')
        event = self.layer.events[-1]
        self.assertIs(event[1], self.x); self.assertIs(event[2], self.state)
        self.assertEqual(event[3], {'_compile_gates':False, '_norm_reuse_normal':True})
        self.assertEqual(self.fused_calls, [])

    def test_all_eight_batches_use_five_native_projections_and_one_fused_call(self):
        fn = c.make_decode(self.module, self.fused, enabled=True)
        for B in range(1, 9):
            with self.subTest(B=B):
                self.batch(B); self.layer.events.clear()
                before = (self.state.conv, self.state.S)
                observed = []
                original_norm = self.layer.norm
                def norm(o, z):
                    observed.append((self.state.conv, self.state.S))
                    return original_norm(o,z)
                with patch.object(self.layer, 'norm', side_effect=norm): output = self.call(fn)
                names = [e[0] for e in self.layer.events]
                expected = ['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','decode-gates']
                if B == 1: expected.append('fake-compiled-gates')
                self.assertEqual(names, expected+['gates','norm','out_proj'])
                self.assertEqual(output.shape, (B,5120))
                self.assertEqual(len(self.fused_calls), B)
                self.assertIs(self.fused_calls[-1][1], before[0])
                self.assertIs(self.fused_calls[-1][5], before[1])
                self.assertIs(observed[0][0], self.state.conv)
                self.assertIs(observed[0][1], self.state.S)
                self.assertIsNot(self.state.S, before[1])
        self.assertEqual(fn.calls, {'candidate':8, 'fallback':0})

    def test_gate_configuration_off_keeps_original_gate_branch(self):
        fn = c.make_decode(self.module, self.fused, enabled=True)
        self.layer._gates_compile_enabled = False
        for B in (1, 8):
            self.batch(B); self.layer.events.clear(); self.call(fn)
            self.assertNotIn('decode-gates', [e[0] for e in self.layer.events])
            self.assertEqual(sum(e[0] == 'gates' for e in self.layer.events), 1)

    def test_context_state_shape_dtype_quant_and_backend_rejections_preserve_inputs(self):
        changes = [lambda: setattr(self.x,'shape',(0,5120)),
                   lambda: setattr(self.x,'shape',(9,5120)),
                   lambda: setattr(self.x,'shape',(1,1,5120)),
                   lambda: setattr(self.x,'dtype','fp32'),
                   lambda: setattr(self.state.conv,'shape',(2,3,10240)),
                   lambda: setattr(self.state.S,'shape',(2,48,128,128)),
                   lambda: setattr(self.state,'record',True),
                   lambda: setattr(self.layer,'_norm_reuse_enabled',True),
                   lambda: setattr(self.layer.in_proj_a,'q',{'bits':8}),
                   lambda: setattr(self.layer.out_proj,'biases',object()),
                   lambda: setattr(self.layer.conv1d_weight,'dtype','fp32'),
                   lambda: setattr(self.module,'_m5_ultra_decode_identity',lambda:False),
                   lambda: setattr(self.mx,'default_device',lambda:'cpu')]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.setUp(); change(); before = (self.state.conv,self.state.S)
                fn = c.make_decode(self.module,self.fused,enabled=True)
                self.assertEqual(self.call(fn),'original')
                self.assertEqual(self.fused_calls,[])
                self.assertEqual([e[0] for e in self.layer.events],['original'])
                self.assertIs(self.state.conv,before[0]); self.assertIs(self.state.S,before[1])
        for flags in ({'_compile_gates':False}, {'_norm_reuse_normal':False}):
            self.setUp(); fn = c.make_decode(self.module,self.fused,enabled=True)
            self.assertEqual(self.call(fn,**flags),'original')
            self.assertEqual(self.fused_calls,[])

    def test_parameter_replacements_follow_current_objects_across_batch_reuse(self):
        fn = c.make_decode(self.module,self.fused,enabled=True)
        for serial, B in enumerate((1,8,1)):
            self.batch(B); self.layer.events.clear()
            for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj'):
                op = getattr(self.layer,name)
                op.weight = Array(op.weight.shape,op.weight.dtype,('weight',serial,name))
                op.scales = Array(op.scales.shape,op.scales.dtype,('scale',serial,name))
            self.layer.conv1d_weight = Array((10240,4),'bf16',('w',serial))
            self.layer.A_log = Array((48,),'fp32',('A',serial))
            self.layer.dt_bias = Array((48,),'fp32',('dt',serial))
            self.call(fn)
            for event in self.layer.events:
                if event[0].startswith('in_proj') or event[0] == 'out_proj':
                    op = getattr(self.layer,event[0])
                    self.assertIs(event[1],op.weight); self.assertIs(event[2],op.scales)
            gate = next(e for e in self.layer.events if e[0] == 'gates')
            self.assertIs(gate[1],self.layer.A_log); self.assertIs(gate[2],self.layer.dt_bias)
            self.assertIs(self.fused_calls[-1][2],self.layer.conv1d_weight)
        self.assertEqual(fn.calls,{'candidate':3,'fallback':0})

    def test_second_call_consumes_its_own_committed_state(self):
        self.batch(8); fn = c.make_decode(self.module,self.fused,enabled=True)
        self.call(fn); conv,S = self.state.conv,self.state.S
        self.call(fn)
        self.assertIs(self.fused_calls[1][1],conv)
        self.assertIs(self.fused_calls[1][5],S)
        self.assertEqual(self.state.S.value,('S',2))

    def test_fused_failure_does_not_commit_or_retry(self):
        self.batch(8); before = (self.state.conv,self.state.S)
        def fail(*args): raise RuntimeError('fused failed')
        fn = c.make_decode(self.module,fail,enabled=True)
        with self.assertRaisesRegex(RuntimeError,'fused failed'): self.call(fn)
        self.assertIs(self.state.conv,before[0]); self.assertIs(self.state.S,before[1])
        self.assertNotIn('original',[e[0] for e in self.layer.events])
        self.assertEqual(fn.calls,{'candidate':0,'fallback':0})

    def test_post_commit_norm_failure_propagates_without_fallback(self):
        self.batch(8); fn = c.make_decode(self.module,self.fused,enabled=True)
        with patch.object(self.layer,'norm',side_effect=RuntimeError('norm failed')):
            with self.assertRaisesRegex(RuntimeError,'norm failed'): self.call(fn)
        self.assertEqual(self.state.conv.value,('conv',1)); self.assertEqual(self.state.S.value,('S',1))
        self.assertNotIn('original',[e[0] for e in self.layer.events])
        self.assertEqual(fn.calls,{'candidate':1,'fallback':0})

    def test_scope_restores_on_exception_and_drops_layer_references(self):
        with patch.object(c,'_load_fused',return_value=self.fused):
            handle = c.install(self.layer,self.module,enabled=True)
            with self.assertRaisesRegex(RuntimeError,'caller failed'):
                with handle: raise RuntimeError('caller failed')
            self.assertTrue(handle.closed); self.assertNotIn('_decode',vars(self.layer))
            handle.restore()
        witness = weakref.ref(self.layer); del self.layer; gc.collect()
        self.assertIsNone(witness()); self.assertIsNone(handle.layer); self.assertIsNone(handle.wrapper)

    def test_foreign_hook_is_not_removed_and_overlap_is_rejected(self):
        with patch.object(c,'_load_fused',return_value=self.fused):
            handle = c.install(self.layer,self.module,enabled=True)
            with self.assertRaisesRegex(ValueError,'untouched'):
                c.install(self.layer,self.module,enabled=True)
            own = self.layer._decode
            foreign = MethodType(lambda self,*a,**kw:None,self.layer)
            object.__setattr__(self.layer,'_decode',foreign)
            with self.assertRaisesRegex(ValueError,'observer'): handle.restore()
            self.assertIs(self.layer._decode,foreign)
            object.__setattr__(self.layer,'_decode',own); handle.restore()

    def test_changed_original_state_kernel_source_rejects_before_kernel_import(self):
        root = Path(c.__file__).resolve().parents[3]/'tutorial_code/miniengine'
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td)
            for relative in c.PINS:
                target = copy/relative; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/relative,target)
            (copy/'kernels/gdn_metal.py').write_text('# altered source\n')
            module = SimpleNamespace(__file__=str(copy/'model/gdn.py'))
            with patch.object(c.importlib,'import_module',side_effect=AssertionError('premature import')):
                with self.assertRaisesRegex(ValueError,'kernels/gdn_metal.py'): c._load_fused(module)

    def test_enable_requires_real_bool_for_both_entry_points(self):
        for value in (1,0,None,'on'):
            with self.subTest(value=value):
                with self.assertRaises(TypeError): c.install(self.layer,self.module,enabled=value)
                with self.assertRaises(TypeError): c.make_decode(self.module,self.fused,enabled=value)


if __name__ == '__main__': unittest.main()
