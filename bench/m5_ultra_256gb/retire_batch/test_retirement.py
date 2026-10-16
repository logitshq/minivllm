"""CPU lifecycle tests: exercise the real candidate methods with inert caches."""
import importlib.util, os, sys, unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tutorial_code.miniengine.engine.model_runner import ModelRunner

source = Path(os.environ.get('MINIVLLM_RETIREMENT_SOURCE', str(ROOT)))
path = source / 'tutorial_code/miniengine/engine/model_runner.py'
spec = importlib.util.spec_from_file_location('tutorial_code.miniengine.engine.retire_candidate', path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

class Batch:
    def __init__(self):
        self.req_ids = ['a', 'b', 'c', 'd']
        self.values = [11, 22, 33, 44]
        self.rebuilds = self.clears = 0
    def rebuild(self, keep, new):
        assert not new
        self.req_ids = [self.req_ids[i] for i in keep]
        self.values = [self.values[i] for i in keep]
        self.rebuilds += 1
    def clear(self):
        self.req_ids, self.values = [], []
        self.clears += 1

def runner(cls):
    r = cls.__new__(cls)
    r.batch = Batch()
    r.staging = {k: object() for k in ('a', 'b', 'c', 'd', 'waiting')}
    r.feats = dict(r.staging)
    r._ahead = object()
    r.spec = None
    return r

class Retirement(unittest.TestCase):
    def test_subset_preserves_order_and_values_with_one_copy(self):
        old, new = runner(ModelRunner), runner(candidate.ModelRunner)
        for rid in ('c', 'a', 'missing'):
            old.drop(rid)
        new.drop_many(['c', 'a', 'missing'])
        self.assertEqual((old.batch.req_ids, old.batch.values), (new.batch.req_ids, new.batch.values))
        self.assertEqual(new.batch.req_ids, ['b', 'd'])
        self.assertEqual(set(old.staging), set(new.staging))
        self.assertEqual(set(old.feats), set(new.feats))
        self.assertEqual(new.batch.rebuilds, 1)
        self.assertIsNone(new._ahead)
    def test_whole_batch_releases_without_copy(self):
        r = runner(candidate.ModelRunner)
        r.drop_many(['a', 'b', 'c', 'd'])
        self.assertEqual((r.batch.rebuilds, r.batch.clears), (0, 1))
        self.assertEqual(r.batch.req_ids, [])
        self.assertEqual(set(r.staging), {'waiting'})
    def test_empty_retirement_preserves_lookahead(self):
        r = runner(candidate.ModelRunner)
        ahead = r._ahead
        r.drop_many([])
        self.assertIs(r._ahead, ahead)
        self.assertEqual((r.batch.rebuilds, r.batch.clears), (0, 0))
    def test_single_drop_api_and_spec_cleanup(self):
        r = runner(candidate.ModelRunner)
        drops, resets = [], []
        r.spec = SimpleNamespace(drop=drops.append, drafter=SimpleNamespace(reset=lambda: resets.append(True)))
        r.drop('a')
        r.drop_many(['c', 'd'])
        self.assertEqual(drops, ['a', 'c', 'd'])
        self.assertEqual(len(resets), 2)
        self.assertEqual(r.batch.req_ids, ['b'])

if __name__ == '__main__':
    unittest.main()
