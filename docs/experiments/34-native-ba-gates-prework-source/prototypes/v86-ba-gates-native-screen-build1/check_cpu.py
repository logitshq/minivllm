"""CPU source/clock checks; no MLX/NumPy/native import."""
import ast
import json
from pathlib import Path
import unittest
D=Path(__file__).resolve().parent
P=D.parent
S=(D/"screen.py").read_text()
OLD=(P/"v44-native-decode-screen-build1/screen.py").read_text()
Q=(P/"v82-native-fullqual-build1/fullqual.py").read_text()
def tree(s):return ast.parse(s)
def dump(n):return ast.dump(n,include_attributes=False)
def named(s,name):return next(n for n in ast.walk(tree(s)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
def clock(s):
    request=next(n for n in ast.walk(tree(s)) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==["index","phase"])
    start=next(i for i,n in enumerate(request.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="t0")
    end=next(i for i,n in enumerate(request.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="t2")
    return ast.Module(body=request.body[start:end+1],type_ignores=[])
class Checks(unittest.TestCase):
    def test_clock_exact44(self):
        self.assertEqual(dump(clock(S)),dump(clock(OLD)))
        inner=next(n for n in ast.walk(clock(S)) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=="range")
        calls={ast.unparse(n.func) for n in ast.walk(inner) if isinstance(n,ast.Call)}
        self.assertEqual(calls,{"range","step_fn","copy_fn","RuntimeError","ids.append"})
    def test_setup_exact44(self):
        self.assertEqual(dump(named(S,"setup")),dump(named(OLD,"setup")))
    def test_actual_layout_and_hooks_exact82(self):
        for name in ("observed_layout","capture_norm_engagement","retain_layout_x","retain_activated","target_fingerprints"):
            self.assertEqual(dump(named(S,name)),dump(named(Q,name)),name)
        self.assertEqual((D/"layout_probe.metal").read_bytes(),(P/"v82-native-fullqual-build1/layout_probe.metal").read_bytes())
        self.assertIn('len(layout_rows)==384',S)
        self.assertIn('len(binding_proof)==288',S)
        self.assertIn('target_fingerprints()==target_before',S)
    def test_readonly_and_import_boundary(self):
        main=named(S,"main")
        inert=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=="args.inert")
        self.assertTrue(any(isinstance(n,ast.Return) for n in inert.body))
        self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name.startswith(("mlx","numpy")) for a in n.names) for n in tree(S).body))
        self.assertIn('seed_baselines=[h.raw() for h in seed_holders]',S)
        self.assertIn('output_baselines=[h.raw() for h in output_holders]',S)
        c=clock(S);self.assertNotIn('originals_readonly',ast.unparse(c));self.assertNotIn('target_fingerprints',ast.unparse(c))
        self.assertEqual(S.count('base.originals_readonly('),2)
        self.assertEqual(S.count('timing_raw=timing_fn()'),1)
        self.assertIn('"schedule":["warm"]*3+["measure"]*3',S)
if __name__=="__main__":unittest.main()
