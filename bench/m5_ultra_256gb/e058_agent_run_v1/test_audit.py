"""Stdlib-only CPU tests for the E058 raw safetensors byte auditor."""
import hashlib
import io
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit


def build_safetensors(tensors, raw_header=None):
    """Build raw safetensors bytes from {name: (dtype, shape, payload)}."""
    if raw_header is not None:
        header = raw_header
        payload = b''.join(t[2] for t in tensors.values())
    else:
        entries = {}
        cursor = 0
        chunks = []
        for name, (dtype, shape, payload) in tensors.items():
            entries[name] = {
                'dtype': dtype,
                'shape': list(shape),
                'data_offsets': [cursor, cursor + len(payload)],
            }
            cursor += len(payload)
            chunks.append(payload)
        header = json.dumps(entries).encode('utf-8')
        payload = b''.join(chunks)
    return struct.pack('<Q', len(header)) + header + payload


def bf16(bits):
    return struct.pack('<H', bits)


def f32(bits):
    return struct.pack('<I', bits)


SMALL_TENSORS = {
    'a': ('BF16', (2,), bf16(0x3F80) + bf16(0x0001)),
    'b': ('F32', (1, 2), f32(0x3F800000) + f32(0xBF800000)),
}


class ParserTests(unittest.TestCase):
    def test_roundtrip(self):
        parsed = audit.parse_safetensors(build_safetensors(SMALL_TENSORS))
        self.assertEqual(set(parsed), {'a', 'b'})
        self.assertEqual(parsed['a'], SMALL_TENSORS['a'])
        self.assertEqual(parsed['b'], SMALL_TENSORS['b'])

    def test_empty_file_and_short_length_prefix(self):
        for blob in (b'', b'\x01\x02\x03', b'\x00' * 8):
            with self.assertRaises(ValueError):
                audit.parse_safetensors(blob)
        with self.assertRaises(ValueError):
            audit.parse_safetensors(struct.pack('<Q', 1000) + b'{}')

    def test_malformed_header_json(self):
        payload = struct.pack('<Q', 5) + b'{oops'
        with self.assertRaises(ValueError):
            audit.parse_safetensors(payload)
        not_object = struct.pack('<Q', 22) + b'[["a",{"x":1}],[1,2]]  '
        with self.assertRaises(ValueError):
            audit.parse_safetensors(not_object)

    def test_duplicate_header_keys(self):
        raw = b'{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},' \
              b'"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
        blob = build_safetensors({'_': ('BF16', (1,), bf16(0))}, raw_header=raw)
        with self.assertRaisesRegex(ValueError, 'duplicate header key'):
            audit.parse_safetensors(blob)

    def test_overlapping_and_gapped_offsets(self):
        overlap = {
            'a': ('BF16', (2,), bf16(1) + bf16(2)),
            'b': ('BF16', (1,), bf16(3)),
        }
        blob = build_safetensors(overlap)
        header = {
            'a': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
            'b': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [2, 4]},
        }
        raw = json.dumps(header).encode('utf-8')
        data = struct.pack('<Q', len(raw)) + raw + blob[8 + len(json.dumps({
            n: {'dtype': d, 'shape': list(s),
                'data_offsets': [0, len(p)]} for n, (d, s, p) in
            overlap.items()}).encode('utf-8')):]
        with self.assertRaisesRegex(ValueError, 'overlapping or gapped'):
            audit.parse_safetensors(data)
        gapped = {
            'a': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
            'b': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [5, 7]},
        }
        raw = json.dumps(gapped).encode('utf-8')
        payload = bf16(1) * 2 + b'\x00' + bf16(3)
        with self.assertRaisesRegex(ValueError, 'overlapping or gapped'):
            audit.parse_safetensors(struct.pack('<Q', len(raw)) + raw + payload)

    def test_offsets_not_covering_payload_end(self):
        header = {'a': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, 2]}}
        raw = json.dumps(header).encode('utf-8')
        blob = struct.pack('<Q', len(raw)) + raw + bf16(1) + bf16(2)
        with self.assertRaisesRegex(ValueError, 'cover the data buffer'):
            audit.parse_safetensors(blob)

    def test_invalid_dtype_shape_and_byte_length(self):
        payload = bf16(0) * 4
        cases = [
            {'a': {'dtype': 'F64', 'shape': [1], 'data_offsets': [0, 8]}},
            {'a': {'dtype': 'BF16', 'shape': [0], 'data_offsets': [0, 0]}},
            {'a': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 6]}},
            {'a': {'dtype': 'BF16', 'shape': [-1], 'data_offsets': [0, 0]}},
            {'a': {'dtype': 'BF16', 'shape': [1.5], 'data_offsets': [0, 3]}},
            {'a': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 3]}},
        ]
        for header in cases:
            raw = json.dumps(header).encode('utf-8')
            blob = struct.pack('<Q', len(raw)) + raw + payload
            with self.assertRaises(ValueError, msg=header):
                audit.parse_safetensors(blob)

    def test_metadata_extra_keys_rejected(self):
        header = {'a': {'dtype': 'BF16', 'shape': [1],
                        'data_offsets': [0, 2], 'extra': 1}}
        raw = json.dumps(header).encode('utf-8')
        blob = struct.pack('<Q', len(raw)) + raw + bf16(0)
        with self.assertRaises(ValueError):
            audit.parse_safetensors(blob)

    def test_nonfinite_bit_patterns_rejected(self):
        for dtype, size, bad in (
                ('BF16', 2, (0x7F80, 0xFF80, 0x7FC0, 0x7FFF)),
                ('F32', 4, (0x7F800000, 0xFF800000, 0x7FC00000, 0x7FFFFFFF))):
            pack = bf16 if dtype == 'BF16' else f32
            for bits in bad:
                tensors = {'x': (dtype, (2,), pack(bits) + pack(0))}
                with self.assertRaisesRegex(ValueError, 'nonfinite'):
                    audit.parse_safetensors(build_safetensors(tensors))
        # Largest finite values pass.
        tensors = {
            'x': ('BF16', (1,), bf16(0x7F7F)),
            'y': ('F32', (1,), f32(0x7F7FFFFF)),
        }
        audit.parse_safetensors(build_safetensors(tensors))


class ComparatorTests(unittest.TestCase):
    SPEC = {'x': ('BF16', (2,))}

    def test_same_sized_payload_mutation_fails(self):
        ref = ('BF16', (2,), bf16(0x3F80) + bf16(0x0001))
        cand = ('BF16', (2,), bf16(0x3F80) + bf16(0x0002))
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            audit.compare_arrays(self.SPEC, 'x', ref, cand)

    def test_signed_zero_difference_fails_bytes(self):
        ref = ('BF16', (2,), bf16(0x0000) + bf16(0x8000))
        cand = ('BF16', (2,), bf16(0x8000) + bf16(0x0000))
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            audit.compare_arrays(self.SPEC, 'x', ref, cand)
        ref32 = ('F32', (1,), f32(0x00000000))
        cand32 = ('F32', (1,), f32(0x80000000))
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            audit.compare_arrays({'x': ('F32', (1,))}, 'x', ref32, cand32)

    def test_dtype_and_shape_mismatch_fail(self):
        ref = ('BF16', (2,), bf16(1) * 2)
        with self.assertRaisesRegex(ValueError, 'dtype'):
            audit.compare_arrays(self.SPEC, 'x', ref, ('F32', (2,), ref[2]))
        with self.assertRaisesRegex(ValueError, 'shape'):
            audit.compare_arrays(self.SPEC, 'x', ref, ('BF16', (1, 2), ref[2]))

    def test_equal_pair_returns_byte_count(self):
        ref = ('BF16', (2,), bf16(0x3F80) + bf16(0x8000))
        self.assertEqual(
            audit.compare_arrays(self.SPEC, 'x', ref, tuple(ref)), 4)


class ScanEquivalenceTests(unittest.TestCase):
    """Vectorized (NumPy) finite scans must match the stdlib scalar scans."""

    def test_numpy_backend_active_when_available(self):
        # Repo venv ships NumPy; on stdlib-only hosts the scalar path is the
        # reference implementation itself, so either backend is acceptable.
        if audit._np is not None:
            self.assertIs(audit._bf16_nonfinite, audit._bf16_nonfinite_numpy)
            self.assertIs(audit._f32_nonfinite, audit._f32_nonfinite_numpy)
        else:
            self.assertIs(audit._bf16_nonfinite, audit._bf16_nonfinite_scalar)
            self.assertIs(audit._f32_nonfinite, audit._f32_nonfinite_scalar)

    def test_all_65536_bf16_bit_patterns_match_scalar(self):
        payload = b''.join(bf16(bits) for bits in range(0x10000))
        expected = audit._bf16_nonfinite_scalar(payload)
        actual = audit._bf16_nonfinite_numpy(payload)
        self.assertTrue(expected)
        self.assertEqual(actual, expected)
        # Exhaustive per-pattern agreement on single-word payloads.
        for bits in range(0x10000):
            word = bf16(bits)
            self.assertEqual(audit._bf16_nonfinite_numpy(word),
                             audit._bf16_nonfinite_scalar(word),
                             'BF16 bits {:#06x}'.format(bits))

    def test_bf16_finite_only_payloads_pass_both(self):
        for bits in (0x0000, 0x8000, 0x3F80, 0x7F7F, 0xFF7F, 0x0001, 0x8080):
            word = bf16(bits)
            self.assertFalse(audit._bf16_nonfinite_numpy(word))
            self.assertFalse(audit._bf16_nonfinite_scalar(word))

    def test_f32_representative_and_random_patterns_match_scalar(self):
        representative = [
            0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F7FFFFF,
            0xFF7FFFFF, 0x00000001, 0x007FFFFF, 0x80800000, 0x7F800000,
            0xFF800000, 0x7F800001, 0x7FFFFFFF, 0xFFC00000, 0xFF800001,
        ]
        rng = random.Random(20261021)
        patterns = representative + [rng.getrandbits(32) for _ in range(4096)]
        for bits in patterns:
            word = f32(bits)
            self.assertEqual(audit._f32_nonfinite_numpy(word),
                             audit._f32_nonfinite_scalar(word),
                             'F32 bits {:#010x}'.format(bits))
        # Bulk random blob agreement.
        blob = b''.join(f32(bits) for bits in patterns)
        self.assertEqual(audit._f32_nonfinite_numpy(blob),
                         audit._f32_nonfinite_scalar(blob))
        finite_blob = b''.join(f32(bits) for bits in patterns
                               if bits & 0x7F800000 != 0x7F800000)
        self.assertFalse(audit._f32_nonfinite_numpy(finite_blob))
        self.assertFalse(audit._f32_nonfinite_scalar(finite_blob))


class ContractTests(unittest.TestCase):
    def test_required_labels_exactly(self):
        labels = audit.expected_labels()
        self.assertEqual(len(labels), 24)
        self.assertEqual(
            set(labels),
            {'T{}/k{}/{}'.format(t, k, p)
             for t in (4, 8) for k in range(1, t + 1)
             for p in ('selected', 'continuation')})

    def test_label_parser(self):
        self.assertEqual(audit.parse_label('T8/k7/continuation'), (8, 7, 'continuation'))
        for bad in ('T2/k1/selected', 'T4/k5/selected', 'T4/k1/final',
                    'T4/k0/selected', 't4/k1/selected', 'T4/1/selected',
                    'T4/k1/selected/x', 'T4k1selected', 1, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                audit.parse_label(bad)

    def test_expected_array_contract(self):
        for width in (4, 8):
            for k in (1, width // 2, width):
                selected = audit.expected_arrays(width, k, 'selected')
                continuation = audit.expected_arrays(width, k, 'continuation')
                for spec in (selected, continuation):
                    self.assertEqual(len(spec), 130)
                    attn = {n for n in spec if n.endswith('/K')}
                    self.assertEqual(
                        attn, {'state/{}/K'.format(i)
                               for i in range(64) if i % 4 == 3})
                    for i in range(64):
                        if i % 4 == 3:
                            self.assertEqual(spec['state/{}/K'.format(i)],
                                             ('BF16', (1, 4, 1536, 256)))
                            self.assertEqual(spec['state/{}/V'.format(i)],
                                             ('BF16', (1, 4, 1536, 256)))
                        else:
                            self.assertEqual(spec['state/{}/conv'.format(i)],
                                             ('BF16', (1, 3, 10240)))
                            self.assertEqual(spec['state/{}/S'.format(i)],
                                             ('F32', (1, 48, 128, 128)))
                self.assertEqual(selected['logits'], ('BF16', (k, 248320)))
                self.assertEqual(selected['features'], ('BF16', (k, 25600)))
                self.assertEqual(continuation['logits'], ('BF16', (1, 248320)))
                self.assertEqual(continuation['features'], ('BF16', (1, 25600)))


class AuditTree:
    """Small on-disk manifest+pair builder under patched module constants."""

    PATCH = {
        'NUM_LAYERS': 8,
        'LOGITS_COLS': 5,
        'FEATURES_COLS': 3,
        'KV_SHAPE': (1, 2, 3, 2),
        'CONV_SHAPE': (1, 2, 2),
        'S_SHAPE': (1, 2, 2, 1),
    }

    def __init__(self, case):
        self.case = case
        self.saved = {k: getattr(audit, k) for k in self.PATCH}
        for key, value in self.PATCH.items():
            setattr(audit, key, value)
        self.root = tempfile.mkdtemp(prefix='e058-audit-')
        self.labels = audit.expected_labels()
        self.small_payload = self._make_small_files()

    def _make_small_files(self):
        """Return {label: (ref_rel, cand_rel)} building equal byte pairs."""
        mapping = {}
        for index, label in enumerate(self.labels):
            width, k, phase = audit.parse_label(label)
            spec = audit.expected_arrays(width, k, phase)
            tensors = {}
            for name in sorted(spec):
                dtype, shape = spec[name]
                count = 1
                for dim in shape:
                    count *= dim
                if dtype == 'BF16':
                    payload = b''.join(
                        bf16((index + j + 1) % 0x7F7F + 1) for j in range(count))
                else:
                    payload = b''.join(
                        f32(0x3F800000 + index + j) for j in range(count))
                tensors[name] = (dtype, shape, payload)
            blob = build_safetensors(tensors)
            rels = []
            for side in ('ref', 'cand'):
                rel = '{}-{}.safetensors'.format(label.replace('/', '_'), side)
                with open(os.path.join(self.root, rel), 'wb') as handle:
                    handle.write(blob)
                rels.append(rel)
            mapping[label] = tuple(rels)
        return mapping

    def manifest(self, mutate=None):
        pairs = []
        for label in self.labels:
            ref_rel, cand_rel = self.small_payload[label]
            record = {'label': label, 'reference': {}, 'candidate': {}}
            for side, rel in (('reference', ref_rel), ('candidate', cand_rel)):
                with open(os.path.join(self.root, rel), 'rb') as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
                record[side] = {'path': rel, 'sha256': digest}
            pairs.append(record)
        manifest = {'schema': audit.SCHEMA, 'pairs': pairs}
        if mutate:
            mutate(manifest)
        path = os.path.join(self.root, 'manifest.json')
        with open(path, 'w') as handle:
            json.dump(manifest, handle)
        return path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            setattr(audit, key, value)
        shutil.rmtree(self.root, ignore_errors=True)
        return False


class AuditTests(unittest.TestCase):
    def test_pass_report_counts_bytes_and_pairs(self):
        with AuditTree(self) as tree:
            per_pair = len(audit.expected_arrays(4, 1, 'selected'))
            report = audit.audit(tree.root, tree.manifest())
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['schema'], 'E058-feature-pairs-v1')
        self.assertEqual(report['pairs_checked'], 24)
        self.assertEqual(report['arrays_checked'], 24 * per_pair)
        self.assertGreater(report['bytes_checked'], 0)
        self.assertEqual(len(report['pairs']), 24)
        self.assertEqual(report['bytes_checked'],
                         sum(p['bytes'] for p in report['pairs']))

    def test_manifest_schema_and_pairs_type(self):
        with AuditTree(self) as tree:
            path = tree.manifest(lambda m: m.update(schema='other'))
            with self.assertRaisesRegex(ValueError, 'schema'):
                audit.audit(tree.root, path)
            path = tree.manifest(lambda m: m.update(pairs={}))
            with self.assertRaisesRegex(ValueError, 'pairs'):
                audit.audit(tree.root, path)
            with self.assertRaisesRegex(ValueError, 'JSON object'):
                audit.audit(tree.root, ['not', 'a', 'dict'])
            path = tree.manifest(lambda m: m.update(extra='nope'))
            with self.assertRaisesRegex(ValueError, 'only schema and pairs'):
                audit.audit(tree.root, path)
            path = tree.manifest(lambda m: m.pop('pairs'))
            with self.assertRaisesRegex(ValueError, 'only schema and pairs'):
                audit.audit(tree.root, path)

    def test_incomplete_and_reordered_labels_rejected(self):
        with AuditTree(self) as tree:
            def drop_last(m):
                m['pairs'] = m['pairs'][:-1]
            with self.assertRaisesRegex(ValueError, '24 required labels'):
                audit.audit(tree.root, tree.manifest(drop_last))

            def reorder(m):
                m['pairs'] = [m['pairs'][1], m['pairs'][0]] + m['pairs'][2:]
            with self.assertRaisesRegex(ValueError, '24 required labels'):
                audit.audit(tree.root, tree.manifest(reorder))

            def dup(m):
                m['pairs'] = m['pairs'] + [m['pairs'][0]]
            with self.assertRaisesRegex(ValueError, '24 required labels'):
                audit.audit(tree.root, tree.manifest(dup))

    def test_sha256_mismatch_rejected(self):
        with AuditTree(self) as tree:
            def corrupt(m):
                m['pairs'][3]['reference']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
                audit.audit(tree.root, tree.manifest(corrupt))
            def bad_hex(m):
                m['pairs'][0]['candidate']['sha256'] = 'z' * 64
            with self.assertRaisesRegex(ValueError, 'malformed sha256'):
                audit.audit(tree.root, tree.manifest(bad_hex))

    def test_path_safety_rejected(self):
        with AuditTree(self) as tree:
            outside = os.path.join(tree.root, 'outside.safetensors')
            with open(outside, 'wb') as handle:
                handle.write(b'x')
            cases = []
            def absolute(m):
                m['pairs'][0]['reference']['path'] = outside
            cases.append(('relative', absolute))
            def escape(m):
                m['pairs'][0]['reference']['path'] = '../outside.safetensors'
            cases.append(('bad components', escape))
            def missing(m):
                m['pairs'][0]['reference']['path'] = 'nope.safetensors'
            cases.append(('does not exist', missing))
            def directory(m):
                m['pairs'][0]['reference']['path'] = '.'
                m['pairs'][0]['reference']['sha256'] = '0' * 64
            cases.append(('bad components', directory))
            for pattern, mutate in cases:
                with self.assertRaisesRegex(ValueError, pattern):
                    audit.audit(tree.root, tree.manifest(mutate))

    def test_symlink_member_rejected(self):
        with AuditTree(self) as tree:
            label = tree.labels[0]
            ref_rel, _ = tree.small_payload[label]
            link = os.path.join(tree.root, 'link.safetensors')
            os.symlink(ref_rel, link)
            def use_link(m):
                m['pairs'][0]['reference']['path'] = 'link.safetensors'
                m['pairs'][0]['reference']['sha256'] = \
                    m['pairs'][0]['candidate']['sha256']
            with self.assertRaisesRegex(ValueError, 'symlink'):
                audit.audit(tree.root, tree.manifest(use_link))

    def test_safetensors_violations_rejected_per_pair(self):
        with AuditTree(self) as tree:
            label = tree.labels[0]
            ref_rel, cand_rel = tree.small_payload[label]
            with open(os.path.join(tree.root, ref_rel), 'rb') as handle:
                good = handle.read()
            cases = {
                'bytes differ': bytearray(good),
                'nonfinite': bytearray(good),
            }
            cases['bytes differ'][-1] ^= 0x01
            (hlen,) = struct.unpack('<Q', good[:8])
            cases['nonfinite'][hlen + 8] = 0x80
            cases['nonfinite'][hlen + 9] = 0x7F
            for pattern, blob in cases.items():
                bad_rel = 'bad.safetensors'
                with open(os.path.join(tree.root, bad_rel), 'wb') as handle:
                    handle.write(bytes(blob))
                digest = hashlib.sha256(bytes(blob)).hexdigest()
                def corrupt(m, rel=bad_rel, sha=digest):
                    m['pairs'][0]['candidate'] = {'path': rel, 'sha256': sha}
                with self.assertRaisesRegex(ValueError, pattern):
                    audit.audit(tree.root, tree.manifest(corrupt))

    def test_signed_zero_difference_fails_audit(self):
        with AuditTree(self) as tree:
            label = tree.labels[1]
            ref_rel, cand_rel = tree.small_payload[label]
            width, k, phase = audit.parse_label(label)
            spec = audit.expected_arrays(width, k, phase)
            tensors_ref = {}
            tensors_cand = {}
            for n, (dtype, shape) in sorted(spec.items()):
                count = 1
                for dim in shape:
                    count *= dim
                unit = bf16(0x0000) if dtype == 'BF16' else f32(0x00000000)
                tensors_ref[n] = (dtype, shape, unit * count)
                sign = bf16(0x8000) if dtype == 'BF16' else f32(0x80000000)
                tensors_cand[n] = (dtype, shape, sign + unit * (count - 1))
            blob = build_safetensors(tensors_cand)
            # Keep candidate from ref side: identical except one signed zero.
            with open(os.path.join(tree.root, 'sz.safetensors'), 'wb') as h:
                h.write(blob)
            digest = hashlib.sha256(blob).hexdigest()
            index = tree.labels.index(label)
            def corrupt(m):
                m['pairs'][index]['candidate'] = {'path': 'sz.safetensors',
                                                  'sha256': digest}
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                audit.audit(tree.root, tree.manifest(corrupt))

    def test_missing_file_arrays_rejected(self):
        with AuditTree(self) as tree:
            label = tree.labels[0]
            width, k, phase = audit.parse_label(label)
            spec = audit.expected_arrays(width, k, phase)
            name = sorted(spec)[0]
            tensors = {}
            for n, (dtype, shape) in sorted(spec.items()):
                if n == name:
                    continue
                count = 1
                for dim in shape:
                    count *= dim
                unit = bf16(1) if dtype == 'BF16' else f32(0x3F800000)
                tensors[n] = (dtype, shape, unit * count)
            blob = build_safetensors(tensors)
            with open(os.path.join(tree.root, 'short.safetensors'), 'wb') as h:
                h.write(blob)
            digest = hashlib.sha256(blob).hexdigest()
            def corrupt(m):
                m['pairs'][0]['reference'] = {'path': 'short.safetensors',
                                              'sha256': digest}
            with self.assertRaisesRegex(ValueError, 'expected 130'):
                audit.audit(tree.root, tree.manifest(corrupt))

    def test_cli_convenience(self):
        with AuditTree(self) as tree:
            manifest = tree.manifest()
            out = os.path.join(tree.root, 'report.json')
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = audit.main(['--root', tree.root, '--manifest', manifest,
                                 '--out', out])
            self.assertEqual(rc, 0)
            with open(out) as handle:
                report = json.load(handle)
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['pairs_checked'], 24)
            self.assertIn('"PASS"', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
