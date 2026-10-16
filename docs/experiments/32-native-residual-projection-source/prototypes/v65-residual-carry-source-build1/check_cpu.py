"""CPU stdlib integration/source checks; never import the candidate package/MLX."""
import ast
import copy
import hashlib
import math
import sys
import types
import unittest
from pathlib import Path

P = Path('/private/tmp/minivllm-engine-optimization-20261014')
ROOT = P / 'candidate-v65-residual-carry-build1-root'
B = ROOT / 'tutorial_code/miniengine'
OLD = P / 'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine'


def method(tree, cls, name):
    return next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class T:
    def __init__(self, key, shape=(1, 5120), dtype='bf16'):
        self.key, self.shape, self.dtype = key, shape, dtype


def op(name, *xs):
    return T(hashlib.sha256((name + ':' + ':'.join(x.key for x in xs)).encode()).hexdigest())


class Norm:
    def __init__(self, name):
        self.weight = T(name, (5120,)); self.eps = 1e-6; self.zero_centered = True
    def __call__(self, x):
        return op('norm-' + self.weight.key, x)


class Embed:
    def __init__(self): self.output_dtype = 'bf16'
    def __call__(self, ids): return T('embed', dtype=self.output_dtype)
    def as_linear(self, x): return op('head', x)


class MLP:
    def __init__(self, i): self.i = i
    def __call__(self, x): return op('mlp-' + str(self.i), x)


class Attn:
    def __init__(self, i): self.i = i
    def __call__(self, x, ctx, i):
        ctx.mix_calls.append(i)
        return op('mix-' + str(self.i), x)


class GDN(Attn): pass
class Block: pass
class Model: pass
class Context: pass
class KV: pass
class State: pass


def objects():
    model = Model(); model.cfg = types.SimpleNamespace(hidden_size=5120, tie_word_embeddings=False)
    model.embed_tokens = Embed(); model.norm = Norm('final'); model.lm_head = lambda x: op('head', x)
    model.layers = []; ctx = Context(); ctx.mode = 'decode'; ctx.prefill_shape = None
    ctx.positions = T('position', (1,), 'i32'); ctx.lens = T('lens', (1,), 'i32')
    ctx.max_len = 256; ctx.caches = []; ctx.mix_calls = []
    for i in range(64):
        layer = Block(); layer.is_attn = i % 4 == 3; layer.mlp = MLP(i)
        layer.input_layernorm = Norm('in-' + str(i)); layer.post_attention_layernorm = Norm('post-' + str(i))
        if layer.is_attn:
            layer.self_attn = Attn(i); cache = KV()
            cache.K = T('K', (1, 4, 256, 256)); cache.V = T('V', (1, 4, 256, 256))
        else:
            layer.linear_attn = GDN(i); cache = State()
            cache.conv = T('conv', (1, 3, 10240)); cache.S = T('S', (1, 48, 128, 128), 'f32')
        model.layers.append(layer); ctx.caches.append(cache)
    return model, T('ids', (1,), 'i32'), ctx


def helper_namespace():
    # Execute the inert helper with its lazy import function replaced; remove only
    # the one function-local residual kernel import and inject a symbolic operator.
    tree = ast.parse((B / 'model/residual_carry.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_carry_forward':
            assert isinstance(node.body[0], ast.ImportFrom)
            node.body.pop(0)
    ns = {}; exec(compile(ast.fix_missing_locations(tree), '<CPU-symbolic-carry>', 'exec'), ns)
    mx = types.SimpleNamespace(bfloat16='bf16', float32='f32', int32='i32')
    ns['_runtime_types'] = lambda: (mx, Context, KV, State, Norm, Embed, Model, Block, MLP, Attn, GDN)
    return ns


class Checks(unittest.TestCase):
    def test_parent_and_old_AST(self):
        old_files = sorted(OLD.rglob('*.py')); new_files = sorted(B.rglob('*.py'))
        self.assertEqual((len(old_files), len(new_files)), (27, 29))
        changed = [str(p.relative_to(OLD)) for p in old_files if p.read_bytes() != (B / p.relative_to(OLD)).read_bytes()]
        self.assertEqual(changed, ['model/qwen3_5.py'])
        old = ast.parse((OLD / 'model/qwen3_5.py').read_text()); new = ast.parse((B / 'model/qwen3_5.py').read_text())
        for cls, name in [('DecoderLayer', '__call__'), ('Qwen3_5ForCausalLM', 'lm_head_fn')]:
            self.assertEqual(ast.dump(method(old, cls, name)), ast.dump(method(new, cls, name)))
        oldcall = method(old, 'Qwen3_5ForCausalLM', '__call__'); newcall = copy.deepcopy(method(new, 'Qwen3_5ForCausalLM', '__call__'))
        self.assertIsInstance(newcall.body[1], ast.If); newcall.body.pop(1)
        self.assertEqual(ast.dump(oldcall), ast.dump(newcall))
        oldinit = method(old, 'Qwen3_5ForCausalLM', '__init__'); newinit = copy.deepcopy(method(new, 'Qwen3_5ForCausalLM', '__init__'))
        self.assertEqual(ast.unparse(newinit.body[2]), 'self.set_residual_carry(False)'); newinit.body.pop(2)
        self.assertEqual(ast.dump(oldinit), ast.dump(newinit))

    def test_exact_qualified_kernel_and_inert_sources(self):
        self.assertEqual(digest(B / 'kernels/residual_norm_metal.py'), 'b53e4b88cb118b2ee05b293e6b4ba4446ae9716e8c7dd9e678a359089d21413c')
        for p in B.rglob('*.py'): compile(p.read_text(), str(p), 'exec')
        ns = {}; exec(compile((B / 'model/residual_carry.py').read_text(), '<inert>', 'exec'), ns)
        self.assertNotIn('mlx.core', sys.modules)
        self.assertEqual((ROOT / 'LICENSE-MLX-MIT').read_bytes(), (OLD.parents[1] / 'LICENSE-MLX-MIT').read_bytes())

    def test_default_off_strict_bool_metadata(self):
        tree = ast.parse((B / 'model/qwen3_5.py').read_text())
        setter = copy.deepcopy(method(tree, 'Qwen3_5ForCausalLM', 'set_residual_carry'))
        ns = {}; exec(compile(ast.fix_missing_locations(ast.Module(body=[setter], type_ignores=[])), '<setter>', 'exec'), ns)
        obj = Model(); ns['set_residual_carry'](obj, False); self.assertIs(obj._residual_carry_enabled, False)
        ns['set_residual_carry'](obj, True); self.assertIs(obj._residual_carry_enabled, True)
        self.assertEqual(list(vars(obj)), ['_residual_carry_enabled'])
        for bad in (0, 1, None, 'true'):
            with self.assertRaises(TypeError): ns['set_residual_carry'](obj, bad)

    def test_upfront_metadata_fallbacks(self):
        ns = helper_namespace(); model, ids, ctx = objects()
        self.assertTrue(ns['_eligible'](model, ids, ctx, False, ()))
        scenarios = [lambda m,i,c: setattr(i, 'shape', (2,)), lambda m,i,c: setattr(c, 'mode', 'prefill'),
                     lambda m,i,c: setattr(c, 'record', True), lambda m,i,c: setattr(c.caches[0], 'verify', True),
                     lambda m,i,c: setattr(m.norm.weight, 'dtype', 'f16'), lambda m,i,c: setattr(m.norm, 'zero_centered', False),
                     lambda m,i,c: setattr(m.norm, 'eps', math.nan), lambda m,i,c: setattr(c, 'positions', None),
                     lambda m,i,c: delattr(m.norm, 'weight')]
        for mutate in scenarios:
            m,i,c = objects(); mutate(m,i,c)
            self.assertIsNone(ns['try_residual_carry'](m,i,c,False,())); self.assertEqual(c.mix_calls, [])
        self.assertIsNone(ns['try_residual_carry'](model,ids,ctx,True,()))
        self.assertIsNone(ns['try_residual_carry'](model,ids,ctx,False,(1,)))
        class CustomContext(Context): pass
        ctx.__class__ = CustomContext
        self.assertFalse(ns['_eligible'](model,ids,ctx,False,()))

    def test_embedding_fallback_before_mutation(self):
        ns = helper_namespace(); model, ids, ctx = objects(); model.embed_tokens.output_dtype = 'f32'
        self.assertIsNone(ns['try_residual_carry'](model,ids,ctx,False,())); self.assertEqual(ctx.mix_calls, [])

    def test_128_fusions_exact_symbolic_order_both_heads(self):
        for tied in (False, True):
            ns = helper_namespace(); model, ids, ctx = objects(); model.cfg.tie_word_embeddings = tied; calls = []
            def fused(left, delta, weight, eps):
                calls.append(weight.key); residual = op('add', left, delta)
                return residual, op('norm-' + weight.key, residual)
            ns['residual_norm'] = fused
            actual = ns['try_residual_carry'](model,ids,ctx,False,())
            x = model.embed_tokens(ids)
            for layer in model.layers:
                h = layer.input_layernorm(x)
                x = op('add', x, op('mix-' + str(layer.mlp.i), h))
                x = op('add', x, layer.mlp(layer.post_attention_layernorm(x)))
            expected = model.lm_head(model.norm(x))
            self.assertEqual(actual.key, expected.key); self.assertEqual(ctx.mix_calls, list(range(64)))
            self.assertEqual(len(calls), 128)
            self.assertEqual(calls, [name for i in range(64) for name in ('post-' + str(i), 'in-' + str(i+1) if i<63 else 'final')])


if __name__ == '__main__': unittest.main()
