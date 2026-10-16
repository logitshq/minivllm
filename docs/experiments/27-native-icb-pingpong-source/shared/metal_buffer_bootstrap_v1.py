"""Record fixed child identity before MLX import, then run the original entry script.
MAX_MB_PER_BUFFER changes tracked command-buffer byte thresholds, not a memory limit.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

P = Path(__file__).resolve().parent
NATIVE_PYTHON = '/Users/ryanli/code/minivllm/.venv/bin/python'
OMLX_ROOT = Path('/Users/ryanli/code/minivllm-tutorial/.artifacts/omlx-baseline/.venv')
OMLX_PYTHON = str(OMLX_ROOT / 'bin/python')
OMLX_CLI = str(OMLX_ROOT / 'bin/omlx')
KNOBS = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
MB_KEY = 'MLX_MAX_MB_PER_BUFFER'
KNOWN = set(KNOBS) | {MB_KEY}
EXPECTED_BINARIES = {
 'core.cpython-313-darwin.so': '963a14eefb170f27a0cf453b32687e2c55b9fa4ea4dde3f97360f60287dad5ef',
 'lib/libmlx.dylib': 'ebc8a5f5465afb79d3d9c8f5b0929de1a9b277b7179db8672f92da77ae110a1e',
 'lib/libjaccl.dylib': '949e901df832d600a71ff2bc5f3738d55b010668102f36e071e1890807a965cd',
}

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, value):
 with Path(path).open('x') as f:
  f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def environment(env, mb, child=False):
 assert str(mb) in {'50', '512'}, 'unfrozen MB arm'
 assert 'MLX_METAL_GPU_ARCH' not in env, 'MLX_METAL_GPU_ARCH must be unset'
 unknown = sorted(k for k in env if k.startswith('MLX_') and k not in KNOWN)
 assert not unknown, 'Unreviewed MLX override key names: ' + ', '.join(unknown)
 result = dict(env)
 result.update(KNOBS)
 result[MB_KEY] = str(mb)
 if child:
  assert all(env.get(k) == v for k, v in result.items()), 'actual child knobs differ'
 return result

def environment_record(env):
 return {'knobs': {k: env.get(k) for k in sorted(KNOWN)},
         'other_mlx_key_value_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(env.items()) if k.startswith('MLX_') and k not in KNOWN},
         'gpu_arch_is_unset': 'MLX_METAL_GPU_ARCH' not in env,
         'scope': 'Only the four validated variables accepted; any other MLX override stops before launch. No secret environment values recorded.'}

def verify_frozen():
 manifest_path = P / 'metal-buffer-exploration-frozen-v1.json'
 d = json.loads(manifest_path.read_text())
 assert d['schema'] == 'metal-buffer-exploration-frozen-v1'
 for name, expected in d['files'].items():
  assert sha(P / name) == expected, 'frozen input changed: ' + name
 return {'manifest_sha256': sha(manifest_path), 'file_count': len(d['files'])}

def engine_identity(engine):
 assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules), 'MLX already imported before receipt'
 expected_python = NATIVE_PYTHON if engine == 'tutorial' else OMLX_PYTHON
 assert sys.executable == expected_python, 'unexpected actual child interpreter'
 mlx = importlib.metadata.distribution('mlx')
 versions = {k: importlib.metadata.version(k) for k in ['mlx', 'mlx-metal']}
 assert versions == {'mlx': '0.32.3', 'mlx-metal': '0.32.3'}
 binaries = {}
 for relative, expected in EXPECTED_BINARIES.items():
  path = Path(mlx.locate_file('mlx/' + relative))
  observed = sha(path)
  assert observed == expected, 'actual MLX native binary differs: ' + relative
  binaries[str(path)] = observed
 if engine == 'tutorial':
  mpath = P / 'source-manifest-v7-accepted13-build1.json'
  m = json.loads(mpath.read_text()); root = Path(m['source_root'])
  actual = {str(f.relative_to(root)): sha(f) for f in root.rglob('*.py')}
  assert len(actual) == 25 and actual == {k: v['after_sha256'] for k, v in m['files'].items()}
  source = {'scope': 'pure accepted parent25, no V15/V16', 'root': str(root), 'source_count': len(actual), 'source_sha256': actual, 'manifest_sha256': sha(mpath)}
 else:
  mpath = P / 'official-omlx-environment-recovery-v1.json'; m = json.loads(mpath.read_text())
  actual = {name: sha(name) for name in m['source_and_native_file_sha256']}
  assert actual == m['source_and_native_file_sha256'], 'official source/native file differs'
  tree = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
  assert tree == m['source_and_native_tree_sha256']
  assert importlib.metadata.version('omlx') == '0.7.1.dev1'
  spec = importlib.util.find_spec('omlx')  # top-level discovery does not execute package code
  assert spec is not None and spec.origin == m['actual_environment']['module_resolution']['omlx']['origin'], 'actual official package resolves elsewhere'
  source = {'scope': 'actual installed official source/native full frozen tree', 'source_count': len(actual), 'tree_sha256': tree, 'manifest_sha256': sha(mpath), 'cli_path': OMLX_CLI, 'cli_sha256': sha(OMLX_CLI)}
 return {'python_executable': sys.executable, 'python_version': sys.version, 'python_binary_sha256': sha(Path(sys.executable).resolve()), 'versions': versions, 'native_binaries_sha256': binaries, 'engine_source': source, 'mlx_not_imported': True}

def redact(command):
 result = list(command)
 if '--api-key' in result:
  result[result.index('--api-key') + 1] = '<api-key>'
 return result

def target_command(engine, original):
 if engine == 'tutorial':
  assert original[:2] == [NATIVE_PYTHON, str(P / 'core_http_precise_v1.py')], 'unexpected native entry'
  assert original[2::2][:2] == ['--model', '--port'], 'unexpected native argv'
  return original[1:]
 assert engine == 'omlx' and original[:2] == [OMLX_CLI, 'serve'], 'unexpected official CLI entry'
 return original

def main(argv=None):
 ap = argparse.ArgumentParser()
 ap.add_argument('--receipt', type=Path, required=True)
 ap.add_argument('--engine', choices=['tutorial', 'omlx'], required=True)
 ap.add_argument('--mb', choices=['50', '512'], required=True)
 ap.add_argument('original', nargs=argparse.REMAINDER)
 a = ap.parse_args(argv)
 original = a.original[1:] if a.original[:1] == ['--'] else a.original
 target = target_command(a.engine, original)
 frozen = verify_frozen()
 environment(os.environ, a.mb, child=True)
 receipt = {'schema': 'metal-buffer-preimport-child-v1', 'pid': os.getpid(), 'ppid': os.getppid(), 'engine': a.engine, 'mb': int(a.mb), 'frozen_identity': frozen, 'environment': environment_record(os.environ), 'actual_identity': engine_identity(a.engine), 'original_entry_argv_redacted': redact(target), 'bootstrap_path': str(Path(__file__)), 'bootstrap_sha256': sha(__file__), 'timing_scope': 'completed before original entry and before MLX import; outside all requests'}
 write_new(a.receipt, receipt)
 sys.argv = target
 runpy.run_path(target[0], run_name='__main__')

if __name__ == '__main__':
 main()
