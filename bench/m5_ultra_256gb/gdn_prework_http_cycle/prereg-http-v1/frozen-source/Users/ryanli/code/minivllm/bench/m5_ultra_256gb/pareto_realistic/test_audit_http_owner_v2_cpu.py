"""Zero-byte provenance regression plus the preserved owner mechanism tests."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.pareto_realistic import audit_http_owner as old
from bench.m5_ultra_256gb.pareto_realistic import audit_http_owner_v2 as a
from bench.m5_ultra_256gb.pareto_realistic import test_audit_http_owner_cpu as previous


class PreservedTests(previous.Tests):
    def setUp(self):
        binding=patch.object(previous,'a',a);binding.start();self.addCleanup(binding.stop)
        super().setUp()


def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return a.sha(path)


def encoded(value):
    return (json.dumps(value,sort_keys=True)+'\n').encode()


def fixture(root):
    """Real files, copied as the actual producer does; no fabricated file stats."""
    runtime=root/'runtime';model=root/'model';package=root/'package'
    pins={}
    empty=[]
    for i in range(122):
        path=runtime/('httpcore/_backends/__init__.py' if i==0 else f'module{i}.py')
        pins[str(path)]=write(path,b'' if i==0 else b'# CPU fixture\n')
        if i==0:empty.append(path)
    core=package/'tutorial_code/miniengine';files={}
    for i in range(31):
        rel=('kernels/__init__.py','model/__init__.py')[i] if i<2 else f'module{i}.py'
        path=core/rel;files[rel]=write(path,b'' if i<2 else b'# CPU frozen core\n')
        if i<2:empty.append(path)
    config=model/'config.json';write(config,b'{}\n')
    manifests={
        'fixtures':{},'source_manifest':{'files':files},
        'model_manifest':{'schema':'E023-model-v1','model':str(model),'files':{'config.json':a.sha(config)}}}
    numeric={'model':str(model),'package_root':str(package),'execution':{'pins':pins}}
    for name,data in manifests.items():
        path=root/(name+'.json');digest=write(path,encoded(data));numeric[name]={'path':str(path),'sha256':digest}
    plan=root/'numeric-plan.json';plan_sha=write(plan,encoded(numeric))
    wrapper={'qualification':{'path':str(plan),'sha256':plan_sha},'execution':{'pins':{}}}
    expected={**pins,str(plan):plan_sha,**{x['path']:x['sha256'] for x in (numeric[n] for n in manifests)},
              **{str(core/name):digest for name,digest in files.items()}}
    assert len(expected)==157
    owner=root/'owner'
    for phase in ('preflight','postflight'):
        folder=owner/phase;write(folder/'plan.json',plan.read_bytes());verified={}
        for source,digest in expected.items():
            payload=Path(source).read_bytes();write(folder/'frozen-source'/source.lstrip('/'),payload)
            verified[source]={'sha256':digest,'source_copied':True,'bytes':len(payload)}
        write(folder/'verified-pins.json',encoded(verified))
        stat=config.stat()
        proof={str(config):dict(sha256=a.sha(config),bytes=stat.st_size,inode=stat.st_ino,mtime_ns=stat.st_mtime_ns)}
        write(folder/'verified-model-pins.json',encoded(proof))
    return owner,wrapper,numeric,plan_sha,empty


class ZeroByteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.owner,self.wrapper,self.numeric,self.plan_sha,self.empty=fixture(self.root)

    def check(self,module=a):
        with patch.object(module,'BASE_PLAN_SHA',self.plan_sha):
            return module.preparation_pins(self.owner,self.wrapper,self.numeric)

    def test_three_empty_producer_files_old_rejects_new_accepts_exact_bytes(self):
        empty_sha=hashlib.sha256(b'').hexdigest()
        self.assertEqual(len(self.empty),3)
        for path in self.empty:self.assertEqual(a.sha(path),empty_sha);self.assertEqual(path.stat().st_size,0)
        with self.assertRaisesRegex(ValueError,'recorded source pin'):self.check(old)
        result=self.check();self.assertEqual(result['verified_entries_each'],157)
        self.assertFalse(result['model_shards_rehashed'])

    def test_negative_noninteger_and_bool_sizes_rejected(self):
        path=self.owner/'preflight/verified-pins.json';original=a.read(path)
        for value in (-1,False,True,0.0,'0'):
            bad=copy.deepcopy(original);bad[str(self.empty[0])]['bytes']=value;write(path,encoded(bad))
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'recorded source pin'):self.check()

    def test_zero_byte_requires_exact_hash_and_actual_size(self):
        proof=self.owner/'preflight/verified-pins.json';original=a.read(proof)
        bad=copy.deepcopy(original);bad[str(self.empty[0])]['sha256']='0'*64;write(proof,encoded(bad))
        with self.assertRaisesRegex(ValueError,'recorded source pin'):self.check()
        write(proof,encoded(original))
        copied=self.owner/'preflight/frozen-source'/str(self.empty[0]).lstrip('/')
        write(copied,b'x')
        with self.assertRaisesRegex(ValueError,'pinned file size'):self.check()

    def test_ast_preparation_only_changes_zero_bound_all_other_checks_preserved(self):
        self.assertEqual(a.sha(old.__file__),a.FROZEN_V1_SHA)
        left=ast.parse(Path(old.__file__).read_text());right=ast.parse(Path(a.__file__).read_text())
        funcs=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        l,r=funcs(left),funcs(right);self.assertEqual(set(l),set(r));changes=0
        target=l['preparation_pins']
        for node in ast.walk(target):
            if (isinstance(node,ast.Compare) and isinstance(node.left,ast.Subscript) and
                isinstance(node.left.value,ast.Name) and node.left.value.id=='row' and
                isinstance(node.left.slice,ast.Constant) and node.left.slice.value=='bytes' and
                len(node.ops)==1 and isinstance(node.ops[0],ast.Gt) and
                len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==0):
                node.ops=[ast.GtE()];changes+=1
        self.assertEqual(changes,1)
        for name in l:
            if name!='audit':self.assertEqual(ast.dump(l[name],include_attributes=False),ast.dump(r[name],include_attributes=False),name)


if __name__=='__main__':unittest.main()
