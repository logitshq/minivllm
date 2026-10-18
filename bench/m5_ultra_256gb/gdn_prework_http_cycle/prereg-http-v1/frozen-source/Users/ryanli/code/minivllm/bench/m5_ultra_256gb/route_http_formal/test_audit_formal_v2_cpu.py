"""CPU-only tokenizer-reference correction regressions; no model/backend import.

One read-only case uses the actual closed registration's tokenizer (~20 MB),
not any model shard. All corruptions use fresh temporary fake snapshot trees.
The original failed auditor and experiment artifacts are never modified.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from bench.m5_ultra_256gb.route_http_formal import audit_formal as old
from bench.m5_ultra_256gb.route_http_formal import audit_formal_v2 as new

ROOT=Path('/Users/ryanli/code/minivllm')
ACTUAL=ROOT/'.artifacts/m5-ultra-256gb/route-http-formal-v2-owner'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fixture(root,symlink=True):
    model=root/'hub/models--CPU--fixture/snapshots'/('a'*40);model.mkdir(parents=True)
    target=root/'hub/blobs/ab/payload';target.parent.mkdir(parents=True);target.write_bytes(b'{"CPU":"tokenizer fixture"}\n')
    path=model/'tokenizer.json'
    if symlink:path.symlink_to(target)
    else:path.write_bytes(target.read_bytes());target=path
    st=target.stat();pin=dict(sha256=new.sha(target),bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns)
    manifest=root/'model-manifest.json'
    write(manifest,dict(schema='E023-model-v1',model=str(model),revision=model.name,files={'tokenizer.json':pin['sha256']}))
    numeric=dict(model=str(model),model_manifest=dict(path=str(manifest),sha256=new.sha(manifest)))
    owner=root/'owner'
    for stage in ('preflight','postflight'):write(owner/stage/'verified-model-pins.json',{str(path):pin})
    return dict(path=str(path),sha256=pin['sha256']),numeric,owner,target


class TokenizerReferenceTests(unittest.TestCase):
    def test_actual_closed_snapshot_old_rejects_new_checks_original_bytes(self):
        specs=new.read(ACTUAL/'formal-specs.json');numeric=new.read(ACTUAL/'preflight/plan.json')
        item=specs[0]['tokenizer']
        self.assertTrue(Path(item['path']).is_symlink())
        with self.assertRaisesRegex(ValueError,'canonical unchanged pinned ref'):old.pinned(item)
        path,proof=new.tokenizer_reference(item,numeric,ACTUAL)
        self.assertEqual(str(path),item['path']);self.assertTrue(proof['is_symlink'])
        self.assertEqual(proof['sha256'],item['sha256']);self.assertEqual(proof['bytes'],19989325)
        self.assertNotEqual(proof['resolved_path'],proof['path'])
        self.assertEqual(proof['resolved_path'],str(path.resolve()))

    def test_registered_regular_and_symlink_references_accept(self):
        for symlink in (False,True):
            with self.subTest(symlink=symlink),tempfile.TemporaryDirectory() as tmp:
                item,numeric,owner,target=fixture(Path(tmp).resolve(),symlink)
                path,proof=new.tokenizer_reference(item,numeric,owner)
                self.assertEqual(proof['is_symlink'],symlink);self.assertEqual(path,Path(item['path']))
                self.assertEqual(proof['resolved_path'],str(target));self.assertEqual(proof['sha256'],item['sha256'])

    def test_unregistered_path_digest_or_manifest_reject(self):
        for mutation in ('path','digest','manifest-model','manifest-revision','manifest-pin'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                item,numeric,owner,target=fixture(Path(tmp).resolve())
                if mutation=='path':
                    alias=Path(item['path']).with_name('other.json');alias.symlink_to(target);item['path']=str(alias)
                elif mutation=='digest':item['sha256']='f'*64
                else:
                    p=Path(numeric['model_manifest']['path']);manifest=new.read(p)
                    if mutation=='manifest-model':manifest['model']+='-different'
                    if mutation=='manifest-revision':manifest['revision']='b'*40
                    if mutation=='manifest-pin':manifest['unregistered']='changed'
                    write(p,manifest)
                    if mutation!='manifest-pin':numeric['model_manifest']['sha256']=new.sha(p)
                with self.assertRaises(ValueError):new.tokenizer_reference(item,numeric,owner)

    def test_retarget_same_bytes_different_inode_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            item,numeric,owner,target=fixture(Path(tmp).resolve());other=target.with_name('other')
            other.write_bytes(target.read_bytes());st=target.stat();os.utime(other,ns=(st.st_atime_ns,st.st_mtime_ns))
            path=Path(item['path']);path.unlink();path.symlink_to(other)
            self.assertEqual(new.sha(other),item['sha256']);self.assertNotEqual(other.stat().st_ino,st.st_ino)
            with self.assertRaisesRegex(ValueError,'target bytes/stat'):new.tokenizer_reference(item,numeric,owner)

    def test_payload_tamper_even_with_preserved_size_and_mtime_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            item,numeric,owner,target=fixture(Path(tmp).resolve());st=target.stat()
            raw=target.read_bytes();target.write_bytes(raw.replace(b'CPU',b'BAD'))
            os.utime(target,ns=(st.st_atime_ns,st.st_mtime_ns))
            self.assertEqual(target.stat().st_size,st.st_size)
            with self.assertRaisesRegex(ValueError,'target bytes/stat'):new.tokenizer_reference(item,numeric,owner)

    def test_missing_nonregular_and_mismatched_prepost_targets_reject(self):
        for mutation in ('missing','directory','post-metadata'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                item,numeric,owner,target=fixture(Path(tmp).resolve())
                if mutation=='post-metadata':
                    p=owner/'postflight/verified-model-pins.json';rows=new.read(p);rows[item['path']]['inode']+=1;write(p,rows)
                else:
                    target.unlink()
                    if mutation=='directory':target.mkdir()
                with self.assertRaises((ValueError,FileNotFoundError)):new.tokenizer_reference(item,numeric,owner)

    def test_generic_receipt_gate_still_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            item,numeric,owner,target=fixture(Path(tmp).resolve())
            with self.assertRaisesRegex(ValueError,'canonical unchanged pinned ref'):new.pinned(item)
            manifest=numeric['model_manifest'];self.assertEqual(new.pinned(manifest),Path(manifest['path']))

    def test_original_functions_unchanged_except_explicit_tokenizer_entry_delta(self):
        old_source=Path(old.__file__).read_text();new_source=Path(new.__file__).read_text()
        self.assertEqual(hashlib.sha256(old_source.encode()).hexdigest(),new.FROZEN_V1_SHA)
        old_fns={n.name:n for n in ast.parse(old_source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        new_fns={n.name:n for n in ast.parse(new_source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertEqual(set(new_fns),set(old_fns)|{'tokenizer_reference'})
        for name,fn in old_fns.items():
            if name!='audit':self.assertEqual(ast.dump(fn),ast.dump(new_fns[name]),name)
        expected=ast.get_source_segment(old_source,old_fns['audit'])
        expected=expected.replace("schema='E024-route-http-formal-independent-audit-v1'","schema='E024-route-http-formal-independent-audit-v2'")
        expected=expected.replace("owner_directory=str(owner),audit_source_sha256=sha(__file__),", "owner_directory=str(owner),audit_source_sha256=sha(__file__),\n        frozen_v1_audit_source_sha256=FROZEN_V1_SHA,\n        parser_correction='Only registered HF snapshot tokenizer ref permits a symlink; original generic receipt canonical gate is unchanged.',")
        expected=expected.replace("    try:\n        data=read(journal/'journal.json')", "    try:\n        need(sha(HERE/'audit_formal.py')==FROZEN_V1_SHA,'original frozen v1 auditor changed')\n        data=read(journal/'journal.json')")
        expected=expected.replace("        tokenizer=tokenizers.Tokenizer.from_file(str(pinned(specs[0]['tokenizer'])))", "        tokenizer_path,tokenizer_proof=tokenizer_reference(specs[0]['tokenizer'],numeric,owner)\n        result['tokenizer_reference']=tokenizer_proof\n        tokenizer=tokenizers.Tokenizer.from_file(str(tokenizer_path))")
        expected=expected.replace("        comparisons=paired_summary(points)\n", "        comparisons=paired_summary(points)\n        same(tokenizer_reference(specs[0]['tokenizer'],numeric,owner)[1],tokenizer_proof,'tokenizer target changed during audit')\n")
        self.assertEqual(ast.dump(ast.parse(expected).body[0]),ast.dump(new_fns['audit']))


if __name__=='__main__':unittest.main()
