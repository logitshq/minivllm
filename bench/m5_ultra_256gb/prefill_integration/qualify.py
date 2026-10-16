"""Run preserved numerical cases against the integrated core, no candidate patch."""
from contextlib import contextmanager
import importlib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prefill_nax'))
from tutorial_code.miniengine.engine import model_runner


@contextmanager
def integrated(active=True):
    previous = model_runner.prefill_batch_backend_supported
    if not active:
        model_runner.prefill_batch_backend_supported = lambda *args: False
    try:
        yield
    finally:
        model_runner.prefill_batch_backend_supported = previous


mode = sys.argv.pop(1)
assert mode in ('boundaries', 'lifecycle')
module = importlib.import_module(mode)
module.enabled = integrated
out = Path(sys.argv[sys.argv.index('--out') + 1])
adapter = out.with_suffix('.adapter.json')
assert not adapter.exists()
adapter.write_text(json.dumps({'mode': mode, 'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              'reference': 'Same integrated core with runner grouping disabled',
                              'candidate': 'Actual integrated core, without experiment monkeypatch'}, indent=2) + '\n')
module.main()
