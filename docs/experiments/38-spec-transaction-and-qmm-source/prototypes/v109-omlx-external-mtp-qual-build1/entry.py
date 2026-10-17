"""Pinned official CLI with qualification observers; never a performance entry."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys

D = Path(__file__).resolve().parent
OLD = D.parent / 'm4-final-native15-official-attempt2'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('none', 'mtp'), required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--pins-sha256', required=True)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    from support import verify
    config = verify(args.pins_sha256)
    assert not any(x == 'mlx' or x.startswith('mlx.') for x in sys.modules)
    reg = json.loads((OLD / 'registry.json').read_text())['omlx']
    assert sys.executable == reg['actual_environment']['python_executable']
    for name, expected in config['versions'].items():
        assert importlib.metadata.version(name) == expected, name
    for path, digest in reg['source_native_files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    controls = {k: os.environ.get(k) for k in config['child_controls']}
    assert controls == config['child_controls']
    assert not [k for k in os.environ if k.startswith(('OMLX_', 'DYLD_', 'METAL_'))]
    assert not [k for k in os.environ if k.startswith('MLX_') and k not in controls]
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command[:2] == reg['reference_launch'][:2]
    pre = {'engine': 'omlx', 'pid': os.getpid(), 'ppid': os.getppid(),
           'python_executable': sys.executable, 'python_version': sys.version,
           'source_native_tree_sha256': reg['tree_sha256'], 'files_checked': len(reg['source_native_files']),
           'controls': controls, 'mlx_not_imported': True, 'qualification_observers_added': True,
           'vendor_sources_unchanged': True, 'versions': config['versions']}
    with (args.run_dir / 'official-preimport.json').open('x') as stream:
        json.dump(pre, stream, indent=2)
    from observer import install
    install(args.run_dir / 'observer.jsonl', args.mode, config['draft_model_dir'])
    sys.argv = command
    runpy.run_path(command[0], run_name='__main__')

if __name__ == '__main__':
    main()
