"""V107 CPU identity/HOLD entry. Deliberately contains no MLX/native/draft loader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--inert',action='store_true')
    mode.add_argument('--execute-gpu',action='store_true')
    args=ap.parse_args(argv)
    pins=json.loads((D/'source-pins.json').read_text())
    for path,expected in pins.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,('pinned source changed',path)
    protocol=json.loads((D/'protocol.json').read_text())
    assert protocol['status']=='HOLD_MISSING_SAME_VERIFY_PREFIX_CACHE_WITNESS' and not protocol['numeric_ready']
    assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
    print(json.dumps({'status':protocol['status'],'CPU_identity_verified':True,'GPU_executed':False,'MLX_imported':False,'numerical_checks_executed':0,'old96_ID_gate_passed':False,'GPU_entry_refused':bool(args.execute_gpu),'reason':'existing chunk forward returns final S only; no same-verify intermediate prefix cache byte witness; rollback default64 differs verify8'}))
    return 3 if args.execute_gpu else 0

if __name__=='__main__':raise SystemExit(main())
