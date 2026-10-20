"""Independent audit mechanisms; no actual run outputs or backend imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('_e048_audit_cpu',Path(__file__).with_name('audit.py'))
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def rows():
    result=[]
    for n in range(5):
        for arm in ('native','candidate'): result.append(dict(phase='warmup',warmup=n,arm=arm,elapsed_ns=123))
    for group in range(5):
        for pair in range(10):
            for slot,arm in enumerate(('native','candidate') if pair%2==0 else ('candidate','native')):
                result.append(dict(phase='measured',group=group,pair=pair,slot=slot,arm=arm,elapsed_ns=100 if arm=='native' else 200))
    return result


class Audit(unittest.TestCase):
    def run_ledger(self,data,*,truncated=False):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'timings.jsonl'; path.write_text('\n'.join(json.dumps(r) for r in data)+('' if truncated else '\n'))
            return a.ledger(path)

    def test_recomputes_counts_and_keeps_all_slow_candidate_groups(self):
        result=self.run_ledger(rows())
        self.assertEqual((result['timing_count'],result['warmup_count'],result['expected_candidate_calls']),(100,10,56))
        self.assertEqual(result['median_group_ratio'],.5)
        self.assertEqual([g['native_over_candidate'] for g in result['groups']],[.5]*5)

    def test_rejects_duplicate_missing_or_misordered_rows(self):
        base=rows()
        for altered in (base[:-1],base+[base[-1]],base[:10]+[base[11],base[10]]+base[12:]):
            with self.assertRaises(ValueError): self.run_ledger(altered)
        with self.assertRaises(ValueError): self.run_ledger(base,truncated=True)

    def test_rejects_boolean_noninteger_negative_or_extra_timing_fields(self):
        for value in (True,0,-1,3.5):
            data=rows(); data[10]['elapsed_ns']=value
            with self.assertRaises(ValueError): self.run_ledger(data)
        data=rows(); data[10]['discard']=True
        with self.assertRaises(ValueError): self.run_ledger(data)

    def test_checks_match_named_arrays_independent_of_loader_dict_order(self):
        expected=[dict(check='native/output',name='root/'+name,shape=shape,dtype=dtype,
            reference_finite=True,candidate_finite=True,bit_equal=True)
            for name,shape,dtype in (('output',[2,5120],'mlx.core.bfloat16'),('conv',[2,3,10240],'mlx.core.bfloat16'),('S',[2,48,128,128],'mlx.core.float32'))]
        a.checks(expected[::-1],[('native/output','output')],2)
        with self.assertRaises(ValueError): a.checks(expected+[expected[0]],[('native/output','output')],2)
        expected[0]['bit_equal']=False
        with self.assertRaises(ValueError): a.checks(expected,[('native/output','output')],2)

    def test_original_byte_comparator_does_not_merge_signed_zeros(self):
        positive=dict(shape=[1],encoding='BF16',raw=b'\0\0')
        negative=dict(shape=[1],encoding='BF16',raw=b'\0\x80')
        with self.assertRaisesRegex(ValueError,'first differing byte'): a.raw.pair(positive,negative,'signed-zero')


if __name__=='__main__': unittest.main()
