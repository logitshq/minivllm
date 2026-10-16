"""CPU-only dispatch guard and renamed-server regression checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from tutorial_code.miniengine.model import layers
from resources_v2 import native_processes


class QualificationGuards(unittest.TestCase):
    def supported(self, version, arch, device, shape=None, gpu=True):
        layers._prefill_nax_identity.cache_clear()
        with patch.object(layers, '_prefill_backend_identity', return_value=(version, arch)), \
             patch.object(mx, 'device_info', return_value={'device_name': device}), \
             patch.object(mx, 'default_device', return_value=mx.gpu if gpu else mx.cpu):
            result = layers.prefill_batch_backend_supported(shape)
        layers._prefill_nax_identity.cache_clear()
        return result

    def test_m5_bounded_domain(self):
        for shape in (None, (2, 1), (26, 19), (32, 128), (8, 512)):
            self.assertTrue(self.supported('0.32.3', 'applegpu_g17d', 'Apple M5 Ultra', shape))
        for shape in ((1, 19), (33, 19), (2, 513), (9, 512), (2, 0)):
            self.assertFalse(self.supported('0.32.3', 'applegpu_g17d', 'Apple M5 Ultra', shape))

    def test_unknown_and_cpu_fallback(self):
        for version, arch, device in [('0.32.4', 'applegpu_g17d', 'Apple M5 Ultra'),
                                      ('0.32.3', 'applegpu_g17d', 'Apple M5 Max'),
                                      ('0.32.3', 'applegpu_g18d', 'Apple M5 Ultra'),
                                      (None, None, None)]:
            self.assertFalse(self.supported(version, arch, device, (8, 19)))
        self.assertFalse(self.supported('0.32.3', 'applegpu_g17d', 'Apple M5 Ultra', (8, 19), gpu=False))

    def test_legacy_domain_unchanged(self):
        self.assertTrue(self.supported('0.32.3', 'applegpu_g16s', 'Apple M4 Max', (64, 600)))

    def test_native_process_title(self):
        text = '74384 1 36204272 62.8 S omlx-server\n123 5 1000 1.0 R /a path/python\n456 1 50 0.0 T /opt/bin/omlx-server\n'
        rows = native_processes(text, owned_pid=123, suspended_pids=(456,))
        self.assertEqual([r['pid'] for r in rows], [74384, 123, 456])
        self.assertFalse(rows[0]['owned'] or rows[0]['authorized_suspended'])
        self.assertTrue(rows[1]['owned'])
        self.assertTrue(rows[2]['authorized_suspended'])
        self.assertFalse(native_processes(text, suspended_pids=(74384,))[0]['authorized_suspended'])


if __name__ == '__main__':
    unittest.main()
