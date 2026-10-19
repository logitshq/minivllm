"""CPU entry guards only; importing smoke does not import MLX or arm timers."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('smoke.py')
spec = importlib.util.spec_from_file_location('nvfp4_smoke_cpu', PATH)
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


class Tests(unittest.TestCase):
    def test_default_and_rejected_arguments_do_not_touch_device_or_outputs(self):
        with patch.object(s, 'device', side_effect=AssertionError('device')), \
             patch.object(s.Path, 'mkdir', side_effect=AssertionError('write')), \
             patch.object(s, 'sha', side_effect=AssertionError('source read')), \
             patch.object(s.signal, 'setitimer', side_effect=AssertionError('timer')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(s.main([]), 0)
            for argv in (['--out', '/private/tmp/not-created'], ['--run'],
                         ['--run', '--out', 'relative']):
                with self.assertRaises(ValueError): s.main(argv)


if __name__ == '__main__': unittest.main()
