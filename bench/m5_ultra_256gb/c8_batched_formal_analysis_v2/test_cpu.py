"""CPU-only regression tests for the narrowly pinned tokenizer resolver."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e055_v2',HERE/'audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.target=self.root/'blob';self.target.write_bytes(b'pinned-tokenizer')
        self.link=self.root/'tokenizer.json';self.link.symlink_to(self.target)
        self.digest=hashlib.sha256(self.target.read_bytes()).hexdigest()
        self.ref={'path':str(self.link),'sha256':self.digest}
        p=patch.multiple(a,TOKENIZER=self.link,TARGET=self.target,TOKENIZER_SHA=self.digest,TOKENIZER_BYTES=self.target.stat().st_size)
        p.start();self.addCleanup(p.stop)
    def fallback(self,ref):
        p=Path(ref['path'])
        if not p.is_absolute() or p.resolve()!=p or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:
            raise ValueError('original strict fallback')
        return p
    def test_exact_bound_symlink(self):
        self.assertEqual(a.pinned_tokenizer(self.ref,self.fallback),self.target)
    def test_other_regular_path_uses_fallback(self):
        self.assertEqual(a.pinned_tokenizer({'path':str(self.target),'sha256':self.digest},self.fallback),self.target)
    def test_other_symlink_rejected(self):
        other=self.root/'other.json';other.symlink_to(self.target)
        with self.assertRaises(ValueError):a.pinned_tokenizer({'path':str(other),'sha256':self.digest},self.fallback)
    def test_reference_hash_cannot_change(self):
        with self.assertRaises(ValueError):a.pinned_tokenizer(dict(self.ref,sha256='0'*64),self.fallback)
    def test_extra_reference_fields_rejected(self):
        with self.assertRaises(ValueError):a.pinned_tokenizer(dict(self.ref,bytes=16),self.fallback)
    def test_same_size_mutated_bytes_rejected(self):
        self.target.write_bytes(b'PINNED-tokenizer')
        with self.assertRaises(ValueError):a.pinned_tokenizer(self.ref,self.fallback)
    def test_size_mutation_rejected(self):
        self.target.write_bytes(b'x')
        with self.assertRaises(ValueError):a.pinned_tokenizer(self.ref,self.fallback)
    def test_retarget_to_identical_bytes_rejected(self):
        other=self.root/'other-blob';other.write_bytes(self.target.read_bytes());self.link.unlink();self.link.symlink_to(other)
        with self.assertRaises(ValueError):a.pinned_tokenizer(self.ref,self.fallback)
    def test_target_replaced_by_symlink_rejected(self):
        other=self.root/'other-blob';other.write_bytes(self.target.read_bytes());self.target.unlink();self.target.symlink_to(other)
        with self.assertRaises(ValueError):a.pinned_tokenizer(self.ref,self.fallback)
    def test_regular_snapshot_is_not_exception(self):
        self.link.unlink();self.link.write_bytes(self.target.read_bytes())
        with self.assertRaises(ValueError):a.pinned_tokenizer(self.ref,self.fallback)
    def test_frozen_math_code_retained(self):
        self.assertIs(a._original_audit.__code__,a.base.audit.__code__)
        self.assertIs(a._scope['reduce_pairs'],a.base.reduce_pairs)
        self.assertIs(a._scope['DECISION'],a.base.DECISION)
    def test_wrapper_does_not_patch_original_module(self):
        original=SimpleNamespace(pinned=self.fallback,read=object())
        unrelated=object();trial=SimpleNamespace(FORMAL='formal',module=lambda path:original if path=='formal' else unrelated)
        with patch.object(a.base,'load',return_value=trial):
            self.assertIs(a.adapted_load(Path('/other'),'digest'),trial)
            wrapped=a.adapted_load(a.base.TRIAL,'digest')
            self.assertIs(wrapped.module('other'),unrelated)
            self.assertIs(wrapped.module('formal').read,original.read)
            self.assertEqual(wrapped.module('formal').pinned(self.ref),self.target)
            with self.assertRaises(ValueError):original.pinned(self.ref)

if __name__=='__main__':unittest.main()
