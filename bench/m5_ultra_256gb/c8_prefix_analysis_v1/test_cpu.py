"""Offline mutation checks using already-closed E049 receipts, no model run."""
import copy
import importlib.util
import json
from pathlib import Path
import types
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_prefix_audit_tests', HERE/'audit.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
A=HERE.parents[2]/'.artifacts/m5-ultra-256gb'


class SavedReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=m.support();a=cls.a
        cls.http=a.load(a.HTTP_AUDIT,a.HTTP_AUDIT_SHA);cls.base=a.load(a.BASE,a.BASE_SHA)
        cls.child=A/'c8-prefix-v1-numerical'
        plan=a.read(cls.child/'plan.json');numeric=a.read(cls.child/'base/plan.json')
        cls.recording=a.read(plan['recording']['path'])
        cls.fixtures=a.read(numeric['fixtures']['path'])['fixtures']
        cls.config=a.read(Path(numeric['model'])/'config.json');cls.config=cls.config.get('text_config',cls.config)
        with (cls.child/'base/steps.jsonl').open('rb') as f:
            cls.first=next(a.parse(line) for line in f if b'"arrays"' in line)
        cls.claimed=a.read(cls.child/'complete.json')['prefix']

    def check_row(self,row):
        case=self.http.case_type(self.base)(self.recording,self.fixtures,self.config)
        case.accept(row)
        return case

    def test_original_first_step_full_coverage(self):
        case=self.check_row(copy.deepcopy(self.first))
        self.assertGreater(case.array_counts['all_arrays'],0)
        self.assertEqual(case.step,1)

    def test_reject_each_nonfinite_arm_or_unequal_bits(self):
        for key in ('reference_finite','candidate_finite','bit_equal'):
            with self.subTest(key=key):
                row=copy.deepcopy(self.first);row['arrays'][0][key]=False
                with self.assertRaises(ValueError):self.check_row(row)

    def test_reject_missing_state_array(self):
        row=copy.deepcopy(self.first)
        i=next(i for i,x in enumerate(row['arrays']) if x['name'].endswith('/S'))
        del row['arrays'][i]
        with self.assertRaises(ValueError):self.check_row(row)

    def test_reject_candidate_output_changed(self):
        row=copy.deepcopy(self.first);row['candidate_delivered'][0][1]+=1
        with self.assertRaises(ValueError):self.check_row(row)

    def test_reject_recorded_and_both_arms_changed_together(self):
        row=copy.deepcopy(self.first)
        for key in ('candidate_delivered','reference_delivered'):row[key][0][1]+=1
        row['recorded']['results'][0][1]+=1
        with self.assertRaises(ValueError):self.check_row(row)

    def test_prefix_boundary_no_fabricated_drain_or_tail(self):
        c=self.claimed
        case=types.SimpleNamespace(step=49,requests={r:dict(ids=ids) for r,ids in c['delivered_prefix_ids'].items()},
            seen=set(c['active_request_ids']),completed=[],hist={},array_counts={},ahead_counts={})
        self.assertEqual(m.prefix_finish(self.a,case,c,self.recording)['delivered_ids'],380)
        for changed in (dict(steps=50),dict(completed=[dict(rid=m.D06)]),dict(full_history_qualified=True)):
            bad={**c,**changed}
            with self.assertRaises(ValueError):m.prefix_finish(self.a,case,bad,self.recording)

    def test_strict_duplicate_and_nonfinite_json(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):self.a.parse(raw)


if __name__=='__main__':unittest.main(verbosity=2)
