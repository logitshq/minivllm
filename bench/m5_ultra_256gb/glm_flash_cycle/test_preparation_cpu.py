"""Prepared fake registration/input tests; no execution performed during timing."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def module(name):
    sp=importlib.util.spec_from_file_location('glm_preparation_test_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
p=module('preregister');builder=module('prepare_inputs')
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()

    def test_description_and_partial_flags_never_prepare(self):
        for mod in (p,builder):
            with patch.object(mod,'prepare',side_effect=AssertionError('no preparation')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mod.main([]),0)
                with self.assertRaises(ValueError):mod.main(['--out',str(self.root/'missing')])

    def test_output_equal_or_ancestor_collisions_rejected(self):
        points=[self.root/name for name in ('registration','owner','journal','analysis.json')]
        p.validate_outputs(points)
        for bad in (points[:3]+[points[0]],points[:3]+[points[1]/'analysis.json'],[self.root,*points[1:]]):
            with self.assertRaisesRegex(ValueError,'non-overlapping'):p.validate_outputs(bad)

    def test_saved_python_preserves_venv_entry_not_resolved_binary(self):
        binary=self.root/'python-real';binary.write_text('fake executable')
        entry=self.root/'venv/bin/python';entry.parent.mkdir(parents=True);entry.symlink_to(binary)
        self.assertEqual(p.python_entry(entry),str(entry));self.assertNotEqual(p.python_entry(entry),str(entry.resolve()))

    def test_saved_actual_commands_use_plan_digest_not_last_source_digest(self):
        source=self.root/'extra-source.py';source.write_text('# unique final source\n')
        probe=self.root/'probe.json';write(probe,{'deadlines':{'work_seconds':1800}})
        inventory=self.root/'inventory.json';write(inventory,{})
        prior=self.root/'prior.json';audit=self.root/'audit.json';write(prior,{});write(audit,{})
        python=self.root/'venv/bin/python';python.parent.mkdir(parents=True);python.write_text('fake interpreter')
        args=SimpleNamespace(out=self.root/'registration',owner=self.root/'owner',journal=self.root/'journal',analysis=self.root/'result.json',
            python=python,probe_plan=probe,probe_plan_sha256=p.sha(probe),verification=inventory,verification_sha256=p.sha(inventory),
            predecessor_journal=prior,predecessor_journal_sha256=p.sha(prior),predecessor_audit=audit,predecessor_audit_sha256=p.sha(audit),
            child_limit_bytes=200<<30,interval_seconds=1.,probe_seconds=5.)
        seen=[]
        def real_ref(ref):
            path=Path(ref['path']);self.assertEqual(p.sha(path),ref['sha256']);return path
        owner=SimpleNamespace(SCHEMA='GLM-cold-load-owner-plan-v1',PHASE='glm-official-load',CONTRACT='contract',PROPOSAL='b'*64,
            reference=real_ref,source_pins=lambda plan:{str(source):p.sha(source)},validate_plan=lambda arg:seen.append(arg))
        guard=SimpleNamespace(policy=lambda **kw:dict(schema='CPU-policy',**kw))
        controller=SimpleNamespace(predecessor=lambda binding:({},[]),source_pins=lambda plan:dict(plan['execution']['pins']))
        modules={'owner.py':owner,'physical_guard.py':guard,'controller.py':controller}
        with patch.object(p,'module',side_effect=lambda path,digest,name:modules[Path(path).name]):result=p.prepare(args)
        path=args.out/'owner-plan.json';commands=p.read(args.out/'commands.json');actual=p.sha(path)
        self.assertEqual(result['owner_plan_sha256'],actual);self.assertNotEqual(actual,p.sha(source))
        self.assertEqual(seen[0].out,args.owner)
        for name in ('controller_prepare','owner','closed_audit'):
            flag='--owner-plan-sha256' if name=='controller_prepare' else '--plan-sha256'
            argv=commands[name];self.assertEqual(argv[argv.index(flag)+1],actual)
        self.assertIn('--run',commands['owner']);self.assertNotIn('--prepare-only',json.dumps(commands))
        self.assertFalse((args.out/'receipt.json').exists());self.assertFalse(args.journal.exists())
        with self.assertRaises(FileNotFoundError):
            with patch.object(p,'module',return_value=SimpleNamespace(owner_plan=lambda *args:None)):
                p.seal_prepared(args.out)

    def test_changed_plan_rejects_before_prepared_receipt(self):
        folder=self.root/'registered';folder.mkdir();write(folder/'owner-plan.json',{})
        write(folder/'plan-build.json',dict(status='CPU_PLAN_ONLY',owner_plan_sha256='0'*64,source_pins={}))
        with self.assertRaisesRegex(ValueError,'actual built plan SHA'):p.seal_prepared(folder)
        self.assertFalse((folder/'receipt.json').exists())

    def test_inventory_never_hashes_shard_payloads_and_resolves_runtime(self):
        root=self.root;here=root/'probe';site=root/'site';snapshot=root/'snapshot';snapshot.mkdir();site.mkdir()
        runtime=site/'ordinary.py';runtime.write_text('# fake runtime\n')
        outer=root/'outer';outer.write_text('python');inner=root/'venv/bin/python';inner.parent.mkdir(parents=True);inner.symlink_to(outer)
        for name in ('core.cpython-313-darwin.so','lib/libmlx.dylib','lib/libjaccl.dylib'):
            path=site/'mlx'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('tiny fake binary')
        sizes=[builder.TOTAL_BYTES//62]*62;sizes[-1]+=builder.TOTAL_BYTES-sum(sizes)
        shards=[dict(name=f'model-{i:05d}.safetensors',bytes=n,sha256=f'{i:064x}') for i,n in enumerate(sizes)]
        stats={r['name']:dict(bytes=r['bytes'],mtime_ns=1,inode=i,device=2) for i,r in enumerate(shards)}
        write(here/'metadata-v1/manifest.json',dict(revision='rev',shards=shards))
        receipts=root/'.artifacts/m5-ultra-256gb/glm-flash-download-v1'
        write(receipts/'complete.json',dict(status='PASS_PINNED_DOWNLOAD_AND_SHA_ONLY',revision='rev',snapshot=str(snapshot),
              shards=[dict(r,sha256_verified=True) for r in shards]))
        write(receipts/'header.json',dict(status='PASS_HEADER_INVENTORY_ONLY',snapshot=str(snapshot)))
        probe=SimpleNamespace(REPO=root,HERE=here,SITE=site,SNAPSHOT=snapshot,REVISION='rev',VENV=root/'venv',REQUIRED_SOURCES=[runtime],
            FIXED_RECEIPTS={'download_receipt':('complete.json',builder.digest(receipts/'complete.json')),
                           'header_inventory':('header.json',builder.digest(receipts/'header.json'))},
            model_metadata=lambda:dict(shard_stats=stats))
        original_stamp,original_digest=builder.stamp,builder.digest;hashed=[]
        def digest(path,check=lambda:None):
            path=Path(path);self.assertNotEqual(path.suffix,'.safetensors');hashed.append(path);return original_digest(path,check)
        def stat(path):return stats[Path(path).name] if Path(path).suffix=='.safetensors' else original_stamp(path)
        with patch.object(builder,'digest',side_effect=digest),patch.object(builder,'stamp',side_effect=stat),patch.object(builder.sys,'executable',str(outer)):
            result,refs,metadata=builder.inventory(probe,outer,lambda:None)
        self.assertEqual(len(result['shards']),62);self.assertEqual(sum(x['stat']['bytes'] for x in result['shards'].values()),builder.TOTAL_BYTES)
        self.assertIn(str(outer.resolve()),result['runtime']);self.assertNotIn(str(inner),result['runtime'])
        self.assertEqual(result['payload_scope'],'retained-download-full-SHA256-and-registered-stat;no-new-payload-hash')


if __name__=='__main__':unittest.main()
