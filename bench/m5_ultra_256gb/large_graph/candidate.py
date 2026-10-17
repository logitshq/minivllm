"""E011: graph reuse above B8, preserving every original projection/kernel.

Load an isolated two-line variant of the qualified class. Production files are
never rewritten by this experiment; exact source bytes are recorded by screen.
"""
from pathlib import Path
import types
from tutorial_code.miniengine.engine import compiled_decode


def candidate_class():
    source = Path(compiled_decode.__file__).read_text()
    changes = {
        '1 <= tokens.shape[0] <= 8': '1 <= tokens.shape[0] <= 64',
        'fused = self._fuse_decode and all(': 'fused = self._fuse_decode and tokens.shape[0] <= 8 and all(',
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType('tutorial_code.miniengine.engine.e011_candidate')
    module.__package__ = 'tutorial_code.miniengine.engine'
    exec(compile(source, '<E011 isolated graph candidate>', 'exec'), module.__dict__)
    return module.CompiledDecode, source
