"""Composition-record mutation checks; no device or numerical producer import."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_e050_audit_tests',HERE/'audit.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.a=m.support()
        child=HERE.parents[2]/'.artifacts/m5-ultra-256gb/c8-batched-prefix-v1-numerical'
        cls.events=[cls.a.parse(raw) for raw in (child/'composition.jsonl').read_bytes().splitlines()]
        cls.claimed=cls.a.read(child/'complete.json')['composition']

    def reject(self,change):
        events=copy.deepcopy(self.events);claimed=copy.deepcopy(self.claimed)
        change(events,claimed)
        with self.assertRaises(ValueError):m.composition(self.a,events,claimed)

    def test_actual_python_counts_and_empty_original_cache(self):
        out=m.composition(self.a,self.events,self.claimed)
        self.assertEqual(out['python_branch_totals'],dict(candidate=240,fallback=0))
        self.assertFalse(out['per_batch_counts_available']);self.assertFalse(out['gpu_invocations_counted'])
        self.assertEqual(self.events[-1]['graph_functions_before_restore'],0)

    def test_missing_or_duplicate_layer_rejected(self):
        self.reject(lambda e,c:e[1]['layers'].pop())
        self.reject(lambda e,c:e[0]['layers'].__setitem__(1,copy.deepcopy(e[0]['layers'][0])))

    def test_fallback_or_unreached_candidate_rejected(self):
        self.reject(lambda e,c:e[1]['layers'][0]['final_counts'].__setitem__('fallback',1))
        self.reject(lambda e,c:e[1]['layers'][0]['final_counts'].__setitem__('candidate',0))

    def test_changed_wrapper_identity_rejected(self):
        self.reject(lambda e,c:e[1]['layers'][0].__setitem__('wrapper_id',1234))

    def test_install_after_trace_rejected(self):
        self.reject(lambda e,c:e[0].__setitem__('graph_functions_before',1))

    def test_unknown_extra_or_missing_lifetime_event_rejected(self):
        self.reject(lambda e,c:e.append(copy.deepcopy(e[-1])))
        self.reject(lambda e,c:e.pop())

    def test_unrestored_or_promoted_counter_scope_rejected(self):
        self.reject(lambda e,c:e[-1].__setitem__('native_decode_restored',False))
        self.reject(lambda e,c:e[-1].__setitem__('counter_scope','GPU calls'))
        self.reject(lambda e,c:c.__setitem__('handle_count',47))


if __name__=='__main__':unittest.main(verbosity=2)
