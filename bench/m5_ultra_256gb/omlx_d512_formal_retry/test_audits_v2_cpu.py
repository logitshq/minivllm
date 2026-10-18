"""Post-run receipt correction regressions; stdlib, retained data only."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from bench.m5_ultra_256gb.omlx_d512_formal_retry import audit_resources as old
from bench.m5_ultra_256gb.omlx_d512_formal_retry import audit_resources_v2 as r
from bench.m5_ultra_256gb.omlx_d512_formal_retry import audit_formal_v2 as a

ROOT=Path(__file__).resolve().parents[3]
ACTUAL=ROOT/'.artifacts/m5-ultra-256gb/omlx-d512-formal-v2-owner'


def save(path,value):
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


class PostRunReceiptTests(unittest.TestCase):
    def test_actual_closed_six_children_old_schema_error_new_binding(self):
        self.assertEqual(r.sha(ACTUAL/'owner-plan.json'),'3d2d1c5880fd3bc32900a84611097d50277a226b9fdb88d04dfa1d37330aa45b')
        with self.assertRaisesRegex(KeyError,'stderr'):old.closed_bindings(ACTUAL)
        complete,plan,stopped,attempt,children,lives=r.closed_bindings(ACTUAL)
        self.assertEqual([x['phase'] for x in lives],r.PHASES)
        self.assertEqual([x['pid'] for x in lives],[123,4546,5913,7208,14550,18078])
        self.assertEqual(len(children),6)
        for life in lives:
            folder=ACTUAL/'phases'/life['phase']
            for name in ('child-absence-00.json','port-absence-00.json'):
                self.assertEqual(set(r.read(folder/name)),{'command','returncode','stdout'})
            proofs=r.inner_absence(folder,life['pid'],plan['http']['port'])
            self.assertEqual(set(proofs),{'child-absence.json','port-after.json'})

    def test_outer_missing_stderr_is_not_synthesized_and_bad_evidence_rejects(self):
        original=r.read(ACTUAL/'phases/r1-c1/child-absence-00.json')
        command=['/bin/ps','-p','123','-o','pid=']
        before=copy.deepcopy(original);r.outer_absence(original,command)
        self.assertEqual(original,before);self.assertNotIn('stderr',original)
        changes=[{'returncode':True},{'returncode':0},{'stdout':'123\n'},
                 {'command':['/bin/ps','-p','999','-o','pid=']},{'stderr':''}]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                r.outer_absence(dict(original,**change),command)

    def inner_fixture(self,root):
        folder=root/'phase';trial=folder/'trial';trial.mkdir(parents=True)
        actual=ACTUAL/'phases/r1-c1/trial'
        for name in ('complete.json','child-absence.json','port-after.json'):
            (trial/name).write_bytes((actual/name).read_bytes())
        return folder

    def test_inner_actual_stderr_pid_port_required_even_with_updated_hash(self):
        changes=[('child-absence.json','stderr','real diagnostic'),
                 ('child-absence.json','returncode',True),
                 ('child-absence.json','stdout','123\n'),
                 ('child-absence.json','command',['/bin/ps','-p','999','-o','pid=']),
                 ('port-after.json','command',['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn']),
                 ('port-after.json','stderr',None)]
        for name,key,value in changes:
            with self.subTest(name=name,key=key),tempfile.TemporaryDirectory() as temp:
                folder=self.inner_fixture(Path(temp));path=folder/'trial'/name
                row=r.read(path);row[key]=value;save(path,row)
                terminal=r.read(folder/'trial/complete.json')
                terminal['artifacts'][name]={'sha256':r.sha(path),'bytes':path.stat().st_size}
                save(folder/'trial/complete.json',terminal)
                with self.assertRaises(ValueError):r.inner_absence(folder,123,18882)

    def test_missing_inner_stderr_and_missing_payload_reject(self):
        for missing in ('field','file'):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as temp:
                folder=self.inner_fixture(Path(temp));path=folder/'trial/child-absence.json'
                if missing=='file':path.unlink()
                else:
                    row=r.read(path);del row['stderr'];save(path,row)
                    terminal=r.read(folder/'trial/complete.json')
                    terminal['artifacts'][path.name]={'sha256':r.sha(path),'bytes':path.stat().st_size}
                    save(folder/'trial/complete.json',terminal)
                with self.assertRaises((ValueError,FileNotFoundError)):r.inner_absence(folder,123,18882)

    def test_inner_bytes_remain_bound_to_closed_trial_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=self.inner_fixture(Path(temp));path=folder/'trial/child-absence.json'
            path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'hash/size'):r.inner_absence(folder,123,18882)

    def test_resource_algorithm_and_formal_math_are_unchanged(self):
        self.assertIs(r.resources,r.raw.resources)
        self.assertEqual(r.sha(r.HERE/'audit_resources.py'),r.FROZEN_V1_SHA)
        self.assertEqual(r.sha(a.HERE/'audit_formal.py'),
                         '31935470b6cf6c0edad1e84b880c05284df1099586a6d335a479b252f452294d')
        before=ast.parse((a.HERE/'audit_formal.py').read_text())
        after=ast.parse(Path(a.__file__).read_text())
        funcs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        left,right=funcs(before),funcs(after)
        self.assertEqual(set(left),set(right))
        for name in left:
            if name!='audit':self.assertEqual(left[name],right[name],name)
        self.assertEqual(a.load('audit_resources_v2.py').__file__,str(r.HERE/'audit_resources_v2.py'))


if __name__=='__main__':unittest.main()
