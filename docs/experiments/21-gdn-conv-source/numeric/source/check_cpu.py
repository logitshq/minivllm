"""Small CPU source/evidence check; no MLX import or data numerical tests."""
from pathlib import Path
import ast
import importlib.util
import sys

D = Path(__file__).resolve().parent
before = {n for n in sys.modules if n == 'mlx' or n.startswith('mlx.')}
spec = importlib.util.spec_from_file_location('v23_numeric_cpu_check', D / 'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.source_check()
actual = runner.evidence_check(runner.DEFAULT_EVIDENCE, 4)
assert actual['explicit_sum_y'] == 4
for wrong in (0, 1, 2, 3, 5, True, 4.0):
    try:
        runner.evidence_check(runner.DEFAULT_EVIDENCE, wrong)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong branch accepted: ' + str(wrong))
for file in D.glob('*.py'):
    ast.parse(file.read_text(), filename=str(file))
proto = runner.PROTO
expected = (proto / 'conv_silu_raw_tail.metal').read_text().replace(
    'act[channel] = v18_bf16_activation(conv).activated;', 'conv_out[channel] = conv;').replace('new_tail', 'probe_tail')
assert (D / 'conv_probe.metal').read_text() == expected
assert 'ensure_row_contiguous=False' in (D / 'runner.py').read_text()
assert 'internal_flags_observed' in (D / 'runner.py').read_text()
assert not [n for n in sys.modules if (n == 'mlx' or n.startswith('mlx.')) and n not in before]
assert runner.main(['--sum-y', '4']) == 0
print('CPU_SOURCE_EVIDENCE_PASS; NO_MLX_IMPORT; NO_GPU; NO_NUMERICAL_TEST')
