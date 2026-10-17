"""CPU-only mechanics tests against the real DecodeBatch with a lazy fake array.

The fake models immutable expression nodes and handle replacement, not MLX
strides, Metal allocation, floating-point arithmetic, or device qualification.
"""
import builtins
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
CORE = HERE.parents[2] / "tutorial_code/miniengine/cache.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load(HERE / "candidate.py", "e028_candidate")


def flat_index(coords, shape):
    out = 0
    for c, d in zip(coords, shape):
        out = out * d + c
    return out


class Array:
    def __init__(self, mx, shape, dtype, read, parents=()):
        self.mx, self.shape, self.dtype = mx, tuple(shape), dtype
        self.read = read
        self.parents = parents

    def values(self):
        return tuple(self.read())

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        keys += (slice(None),) * (len(self.shape) - sum(k is not None for k in keys))
        selectors, result_shape, dim = [], [], 0
        for k in keys:
            if k is None:
                result_shape.append(1)
                continue
            if isinstance(k, Array):
                self.mx.gathers += 1
                indexes = tuple(k.values())
                selectors.append(indexes)
                result_shape.append(len(indexes))
            elif isinstance(k, slice):
                indexes = tuple(range(self.shape[dim]))[k]
                selectors.append(indexes)
                result_shape.append(len(indexes))
            else:
                selectors.append((k,))
            dim += 1
        read, shape = self.read, self.shape
        positions = tuple(flat_index(c, shape) for c in itertools.product(*selectors))

        def sliced():
            data = read()
            return tuple(data[i] for i in positions)

        return Array(self.mx, result_shape, self.dtype, sliced, (self,))

    def __setitem__(self, index, value):
        # MLX-like handle replacement: already built expressions retain old nodes.
        read = self.read
        position = flat_index(index, self.shape)

        def replaced():
            data = list(read())
            data[position] = value
            return tuple(data)

        self.read = replaced


class FakeMX(ModuleType):
    def __init__(self):
        super().__init__("mlx.core")
        self.gathers = self.evaluations = self.index_arrays = 0
        self.concatenations = []

    def tensor(self, shape, seed=0, dtype="bfloat16"):
        values = tuple(seed + i for i in range(math.prod(shape)))
        return Array(self, shape, dtype, lambda: values)

    def array(self, values):
        self.index_arrays += 1
        data = tuple(values)
        return Array(self, (len(data),), "int32", lambda: data)

    def pad(self, array, pads):
        shape = tuple(d + a + b for d, (a, b) in zip(array.shape, pads))
        read, original = array.read, array.shape

        def padded():
            source = read()
            out = []
            for point in itertools.product(*(range(d) for d in shape)):
                shifted = tuple(v - p[0] for v, p in zip(point, pads))
                inside = all(0 <= v < d for v, d in zip(shifted, original))
                out.append(source[flat_index(shifted, original)] if inside else 0)
            return tuple(out)

        return Array(self, shape, array.dtype, padded, (array,))

    def concatenate(self, arrays):
        self.concatenations.append(tuple(arrays))
        shape = (sum(a.shape[0] for a in arrays),) + arrays[0].shape[1:]
        if any(a.shape[1:] != shape[1:] for a in arrays):
            raise ValueError("incompatible concatenation")
        reads = tuple(a.read for a in arrays)
        return Array(self, shape, arrays[0].dtype,
                     lambda: tuple(v for read in reads for v in read()), tuple(arrays))

    def eval(self, *arrays):
        self.evaluations += 1
        return [a.values() for a in arrays]


def fixture(enabled=False, B=2, capacity=256):
    mx = FakeMX()
    package = ModuleType("mlx")
    package.core = mx
    with patch.dict(sys.modules, {"mlx": package, "mlx.core": mx}):
        cache = load(CORE, "e028_real_cache")
    model = SimpleNamespace(layers=[SimpleNamespace(is_attn=True), SimpleNamespace(is_attn=False)])
    Batch = candidate.make_batch_class(cache, mx)
    batch = Batch(model, enabled=enabled)
    batch.req_ids, batch.lens = [f"old-{i}" for i in range(B)], list(range(2, B + 2))
    batch.layers = [
        cache.BatchKV(mx.tensor((B, 1, capacity, 2), 10), mx.tensor((B, 1, capacity, 2), 20)),
        cache.BatchState(mx.tensor((B, 3, 2), 30), mx.tensor((B, 2, 2, 2), 40, "float32")),
    ]
    new = []
    for j, off in enumerate((3, 5)):
        a, g = cache.AttnCache(), cache.GDNCache()
        a.keys, a.values = mx.tensor((1, 1, 16, 2), 100 + j), mx.tensor((1, 1, 16, 2), 200 + j)
        g.conv_state, g.S = mx.tensor((3, 2), 300 + j), mx.tensor((2, 2, 2), 400 + j, "float32")
        a.offset = g.offset = off
        new.append((f"new-{j}", [a, g], 16))
    return mx, cache, batch, new


def snapshots(arrays):
    return [(a.shape, a.dtype, a.values()) for a in arrays]


def inputs(batch, new):
    return batch.arrays() + [a for _, cs, _ in new for a in
                            (cs[0].keys, cs[0].values, cs[1].conv_state, cs[1].S)]


class CandidateTests(unittest.TestCase):
    def test_reference_source_is_exact_reviewed_core(self):
        self.assertEqual(hashlib.sha256(CORE.read_bytes()).hexdigest(), candidate.REFERENCE_CACHE_SHA256)

    def test_import_never_requests_mlx(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "mlx" or name.startswith("mlx."):
                raise AssertionError("GPU library import")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", guarded):
            module = load(HERE / "candidate.py", "e028_candidate_import_check")
        self.assertTrue(callable(module.make_batch_class))

    def test_default_off_delegates_and_instances_are_independent(self):
        mx, cache, batch, new = fixture()
        other = type(batch)(batch.model, enabled=True)
        batch.rebuild([0, 1], new)
        self.assertEqual(batch.last_path, "fallback")
        self.assertEqual(batch.last_reason, "disabled")
        self.assertEqual(mx.gathers, 4)
        self.assertEqual(batch.counters, {"identity_refill": 0, "fallback": 1})
        self.assertEqual(other.counters, {"identity_refill": 0, "fallback": 0})
        self.assertIs(cache.DecodeBatch.rebuild, type(batch).__mro__[1].rebuild)
        with self.assertRaises(TypeError):
            type(batch)(batch.model, enabled=1)
        with self.assertRaises(ValueError):
            candidate.make_batch_class(cache, FakeMX())

    def test_identity_outputs_metadata_and_unused_padding_match_original(self):
        for B in (1, 2, 4, 6):
            with self.subTest(B=B):
                mx, _, batch, new = fixture(True, B)
                ref_mx, ref_cache, ref_adapter, ref_new = fixture(False, B)
                # The oracle calls the real, unchanged superclass method directly.
                ref_cache.DecodeBatch.rebuild(ref_adapter, list(range(B)), ref_new)
                old = inputs(batch, new)
                before = snapshots(old)
                batch.rebuild(list(range(B)), new)
                self.assertEqual(batch.last_path, "identity_refill")
                self.assertEqual((mx.gathers, mx.index_arrays, mx.evaluations), (0, 0, 0))
                self.assertEqual((ref_mx.gathers, ref_mx.index_arrays), (4, 1))
                self.assertEqual(batch.req_ids, ref_adapter.req_ids)
                self.assertEqual(batch.lens, ref_adapter.lens)
                self.assertEqual(snapshots(batch.arrays()), snapshots(ref_adapter.arrays()))
                self.assertEqual(before, snapshots(old))

    def test_reference_calls_fallback_for_transition_controls(self):
        controls = ("reorder", "remove", "no_new", "grow", "empty_old")
        for control in controls:
            with self.subTest(control=control):
                mx, cache, batch, new = fixture(True)
                keep = [0, 1]
                if control == "reorder":
                    keep = [1, 0]
                elif control == "remove":
                    keep = [1]
                elif control == "no_new":
                    new = []
                elif control == "grow":
                    new[0] = (*new[0][:2], 257)
                elif control == "empty_old":
                    batch.clear()
                    keep = []
                with patch.object(cache.DecodeBatch, "rebuild", autospec=True) as original:
                    batch.rebuild(keep, new)
                    original.assert_called_once_with(batch, keep, new)
                self.assertEqual(batch.last_path, "fallback")
                self.assertEqual(mx.evaluations, 0)

    def test_malformed_layouts_delegate_before_building_any_expression(self):
        def wrong_capacity(batch, new, mx, cache):
            batch.layers[0].V = mx.tensor((2, 1, 128, 2))

        changes = [
            wrong_capacity,
            lambda b, n, m, c: setattr(b.layers[1], "S", m.tensor((1, 2, 2, 2))),
            lambda b, n, m, c: setattr(b.layers[0], "K", None),
            lambda b, n, m, c: setattr(b, "lens", [2]),
            lambda b, n, m, c: n[0][1].pop(),
            lambda b, n, m, c: b.layers.pop(),
            lambda b, n, m, c: setattr(b.layers[0], "K", m.tensor((2, 256, 2))),
            lambda b, n, m, c: b.layers.__setitem__(1, b.layers[0]),
        ]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                mx, cache, batch, new = fixture(True)
                change(batch, new, mx, cache)
                with patch.object(cache.DecodeBatch, "rebuild", autospec=True) as original:
                    batch.rebuild([0, 1], new)
                    original.assert_called_once_with(batch, [0, 1], new)
                self.assertEqual(batch.last_path, "fallback")
                self.assertEqual((mx.gathers, mx.index_arrays, mx.evaluations), (0, 0, 0))
                self.assertEqual(mx.concatenations, [])

    def test_every_attention_layer_must_have_unchanged_capacity(self):
        mx, cache, batch, new = fixture(True)
        batch.model.layers.append(SimpleNamespace(is_attn=True))
        batch.layers.append(cache.BatchKV(mx.tensor((2, 1, 512, 2)), mx.tensor((2, 1, 512, 2))))
        for _, cs, _ in new:
            cs.append(cs[0])
        with patch.object(cache.DecodeBatch, "rebuild", autospec=True) as original:
            batch.rebuild([0, 1], new)
            original.assert_called_once()
        self.assertEqual(batch.last_reason, "capacity_changed")

    def test_bool_keep_and_tuple_keep_do_not_enter_identity_branch(self):
        for keep in ([False, True], (0, 1)):
            mx, cache, batch, new = fixture(True)
            with patch.object(cache.DecodeBatch, "rebuild", autospec=True) as original:
                batch.rebuild(keep, new)
                original.assert_called_once_with(batch, keep, new)
            self.assertEqual(batch.last_reason, "nonidentity_keep")

    def test_views_and_delayed_source_writes_preserve_captured_values(self):
        mx, _, batch, new = fixture(True)
        # Offset and stepped fake views exercise provenance, not real device strides.
        parent = mx.tensor((2, 3, 4), 500)
        batch.layers[1].conv = parent[:, :, 1::2]
        old = inputs(batch, new)
        saved = snapshots(old)
        batch.rebuild([0, 1], new)
        expected = snapshots(batch.arrays())
        self.assertIs(mx.concatenations[2][0], old[2])
        self.assertIs(mx.concatenations[3][0], old[3])
        self.assertIs(mx.concatenations[0][0].parents[0], old[0])
        self.assertTrue(all(len(parts) >= 2 for parts in mx.concatenations))
        for array in old:
            array[(0,) * len(array.shape)] = -999
        mx.eval(*batch.arrays())
        self.assertEqual(expected, snapshots(batch.arrays()))
        self.assertNotEqual(saved, snapshots(old))
        self.assertEqual(batch.last_path, "identity_refill")

    def test_output_writes_do_not_change_retained_old_or_staging_handles(self):
        mx, _, batch, new = fixture(True)
        old = inputs(batch, new)
        before = snapshots(old)
        batch.rebuild([0, 1], new)
        mx.eval(*batch.arrays())
        for array in batch.arrays():
            array[(0,) * len(array.shape)] = -333
        mx.eval(*batch.arrays())
        self.assertEqual(before, snapshots(old))

    def test_enabled_is_runtime_default_off_and_does_not_change_existing_state(self):
        mx, _, batch, new = fixture(False)
        old = tuple(batch.arrays())
        batch.enabled = True
        self.assertEqual(old, tuple(batch.arrays()))
        batch.rebuild([0, 1], new)
        batch.enabled = False
        batch.rebuild(list(range(batch.size)), [])
        self.assertEqual(batch.counters, {"identity_refill": 1, "fallback": 1})
        self.assertEqual(batch.last_reason, "disabled")


if __name__ == "__main__":
    unittest.main()
