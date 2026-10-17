"""E026 screen mechanisms only. No MLX/NumPy/device/listener/model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e026_screen', HERE / 'screen.py')
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class MechanismTests(unittest.TestCase):
    def test_all_65536_words_covered_without_float_conversion(self):
        covered = set()
        for kind in screen.KINDS[:-1]:
            payload = screen.fixture_bytes(kind)
            self.assertEqual(len(payload), 61440)
            words = struct.unpack('<' + 'H' * screen.WORDS, payload)
            self.assertEqual(words[0], int(kind.split('-')[1]))
            covered.update(words)
        self.assertEqual(covered, set(range(65536)))
        for signed_zero_or_nan in (0, 0x8000, 0x7f81, 0x7fc1, 0xff81, 0xffff):
            self.assertIn(signed_zero_or_nan, covered)

    def test_finite_fixture_checks_exponents_and_signed_zero(self):
        words = struct.unpack('<' + 'H' * screen.WORDS, screen.fixture_bytes('finite'))
        self.assertEqual(words[:2], (0, 0x8000))
        self.assertTrue(all((w & 0x7f80) != 0x7f80 for w in words))
        with self.assertRaises(ValueError):
            screen.fixture_bytes('anything')

    def test_contract_fixed_small_scope_not_hard_cap_claim(self):
        c = screen.contract('on-lazy')
        self.assertEqual((c['lengths'], c['case_count']), ([1, 19, 935, 978], 16))
        self.assertLess(c['largest_parent_logical_bytes'], 20 << 20)
        self.assertLess(c['conservative_named_buffer_envelope_bytes'], 64 << 20)
        self.assertIn('not a hard process-memory cap', c['envelope_scope'])
        self.assertIn('cannot prevent transient excess', c['memory_guard_scope'])
        self.assertEqual(c['child_best_effort_alarm_seconds'], 30)

    def test_snapshot_deduplicates_backing_via_runtime_api(self):
        # Different Python wrappers share buffer 1; only the API owns dedup.
        parent = SimpleNamespace(buffer=1, size=81920)
        tail = SimpleNamespace(buffer=1, size=81920)
        output = SimpleNamespace(buffer=2, size=61440)
        calls = []
        def size(*trees):
            calls.append(trees)
            return sum({x.buffer: x.size for x in trees}.values())
        mx = SimpleNamespace(get_array_buffer_size=size, get_active_memory=lambda: 143360,
                             get_peak_memory=lambda: 143360, get_cache_memory=lambda: 0)
        value = screen.snapshot(mx, {'parent': parent, 'tail': tail, 'output': output}, 'eval', 'case', 0)
        self.assertEqual(value['unique_buffer_bytes'], 143360)
        self.assertEqual(value['group_buffer_bytes'], {'parent': 81920, 'tail': 81920, 'output': 61440})
        screen.guard_snapshot(value)
        json.dumps(value)  # No array wrapper leaked into recorded metadata.
        self.assertEqual(len(calls), 4)

    def test_missing_api_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'API unavailable'):
            screen.snapshot(object(), {}, 'phase', 'case', 0)

    def test_memory_guard_is_strict_post_boundary(self):
        good = {'active_bytes': 61440, 'peak_bytes': 81920, 'cache_bytes': 0, 'unique_buffer_bytes': 61440}
        screen.guard_snapshot(good)
        for patch in ({'peak_bytes': 64 << 20}, {'active_bytes': -1}, {'unique_buffer_bytes': 61441}):
            with self.assertRaises(ValueError):
                screen.guard_snapshot({**good, **patch})

    @staticmethod
    def measurements(arm):
        parent, output = 81920, 81920 if arm == 'off' else 61440
        union = parent if arm == 'off' else parent + output
        return [
            {'group_buffer_bytes': {'parent': parent, 'tail': parent}, 'unique_buffer_bytes': parent},
            {'group_buffer_bytes': {'output': output}, 'unique_buffer_bytes': union},
            {'active_bytes': union},
            {'unique_buffer_bytes': output, 'active_bytes': output},
            {'unique_buffer_bytes': 0},
        ]

    def test_owned_buffer_and_actual_active_release_required(self):
        for arm in screen.ARMS:
            result = screen.ownership_checks(arm, *self.measurements(arm))
            self.assertEqual(result['retained_backing_reduction_bytes'], 0 if arm == 'off' else 20480)
        for alteration in ('not-distinct', 'still-large', 'no-active-release', 'wrong-original-alias', 'retained-root'):
            rows = copy.deepcopy(self.measurements('on-lazy'))
            if alteration == 'not-distinct': rows[1]['unique_buffer_bytes'] = 81920
            if alteration == 'still-large': rows[1]['group_buffer_bytes']['output'] = 81920
            if alteration == 'no-active-release': rows[3]['active_bytes'] = rows[2]['active_bytes']
            if alteration == 'wrong-original-alias': rows[0]['group_buffer_bytes']['tail'] = 61440
            if alteration == 'retained-root': rows[4]['unique_buffer_bytes'] = 1
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                screen.ownership_checks('on-lazy', *rows)

    def test_fixed_candidate_and_copied_source_hashes(self):
        pins = screen.verify_sources()
        self.assertEqual(pins[str(HERE / 'candidate.py')], screen.CANDIDATE_SHA)
        self.assertEqual(pins[str(HERE / 'source-pins.json')], screen.SOURCE_MANIFEST_SHA)
        self.assertEqual(Path(screen.load_candidate().own_conv_tail.__code__.co_filename).resolve(), HERE / 'candidate.py')

    def test_default_help_and_bad_args_never_import_device_or_socket(self):
        code = '''
import importlib.abc, importlib.util, socket, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('socket'))
s=importlib.util.spec_from_file_location('screen',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
raise SystemExit(m.main(sys.argv[2:]))
'''
        for args, expected in (([], 0), (['--help'], 0), (['--unknown'], 2),
                               (['--run'], 2), (['--out', '/unused-output-path'], 2),
                               (['--arm', 'on-lazy'], 0)):
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(HERE / 'screen.py'), *args],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertNotIn('AssertionError', result.stderr)
            if not args:
                self.assertEqual(json.loads(result.stdout)['case_count'], 16)


if __name__ == '__main__':
    unittest.main()
