import ast,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Protocol(unittest.TestCase):
 def test_client_and_statistics_preserved(self):
  ref=D.parents[2]/'.artifacts/m5-ultra-256gb/source/reference/v19-e2e-incremental-build2'
  self.assertEqual((D/'replay.py').read_bytes(),(ref/'replay.py').read_bytes())
  old=ast.parse((ref/'driver.py').read_bytes());new=ast.parse((D/'protocol.py').read_bytes())
  for name in ('run_one','cleanup_owned','derived_snapshot','wait_port_absence'):
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
   b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(a),ast.dump(b),name)
 def test_sourcefinder_before_runtime(self):
  s=(D/'child.py').read_text();self.assertLess(s.index('sys.meta_path.insert(0,source_finder)'),s.index('import mlx.core as mx'))
if __name__=='__main__':unittest.main()
