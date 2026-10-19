"""Only default-inert/explicit-child admission; no device numerical claim."""
import builtins
import contextlib
import importlib.util
import io
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

class EntryGuard(unittest.TestCase):
    def test_default_and_imported_run_are_inert(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.startswith(('mlx','numpy','socket','subprocess')):raise AssertionError('forbidden import '+name)
            return original(name,*args,**kwargs)
        def forbidden(*args,**kwargs):raise AssertionError('forbidden IO/timer')
        spec=importlib.util.spec_from_file_location('_real_probe_cpu',Path(__file__).with_name('real_projection_probe_v1.py'))
        with patch.object(builtins,'__import__',guarded),patch.object(Path,'open',forbidden),patch.object(Path,'mkdir',forbidden),patch.object(signal,'setitimer',forbidden):
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with contextlib.redirect_stdout(io.StringIO()) as output:self.assertEqual(module.main([]),0)
            self.assertIn('DESCRIPTION ONLY',output.getvalue())
            with self.assertRaises(ValueError):module.main(['--out','/private/tmp/not-created'])
            with self.assertRaises(ValueError):module.main(['--run','--out','/private/tmp/not-created'])
if __name__=='__main__':unittest.main()
