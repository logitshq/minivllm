"""V120 thin candidate116 factory. Import/help/inert are stdlib-only; qualification is mandatory."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

D = Path(__file__).resolve().parent
P = D.parent
ROOT = P / 'candidate-v116-bounded-draftloader-build1-root'
QUAL_STATUS = 'PASS_V117_BOUNDED_LOAD_VERIFY_BROADCAST_M1_OLD96IDS_LITERAL_AND_ORDINARY_M1_COMMIT128_LASTLOGITS_STRICT'
CONTROLS = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
            'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualification(binding_path):
    """No model import or inference. An unbound draft cannot launch a server."""
    b = json.loads(Path(binding_path).read_text())
    if b.get('status') != 'BOUND_V117_ACTUAL_PASS':
        raise RuntimeError('HOLD: actual V117 PASS and release not bound')
    for kind in ('receipt', 'release'):
        if sha(b[kind]) != b[kind + '_sha256']:
            raise RuntimeError('qualification evidence changed: ' + kind)
    q = json.loads(Path(b['receipt']).read_text())
    r = json.loads(Path(b['release']).read_text())
    if q.get('status') != QUAL_STATUS or q.get('target1349_readonly') is not True:
        raise RuntimeError('V117 strict full-model PASS required')
    if q.get('completed_old_ID_checks') != {'A': 96, 'B': 96}:
        raise RuntimeError('V117 old96 per-arm result missing')
    if not q.get('same_frontier_commit_checks') or not all(x.get('hard_gate_pass') is True for x in q['same_frontier_commit_checks']):
        raise RuntimeError('V117 same-frontier hard gates absent')
    if r.get('status') != 'PASS_EXTERNAL_RELEASE' or r.get('GPU_released') is not True:
        raise RuntimeError('V117 external release required')
    absent = r.get('strict_absence', [])
    if not absent or not all(x.get('returncode') == 1 and x.get('stdout') == x.get('stderr') == '' for x in absent):
        raise RuntimeError('strict owned PID absence required')
    for p in ROOT.rglob('*.py'):
        if q['pins'].get(str(p)) != sha(p):
            raise RuntimeError('qualified candidate source mismatch')
    return b


def assemble(engine, mode, *, ModelRunner, load_drafter, draft_path):
    """Dependency injection permits CPU checks; production receives original classes."""
    if mode not in ('ordinary', 'speculative'):
        raise ValueError('mode must be ordinary or speculative')
    gdns = [layer.linear_attn for layer in engine.model.layers if not layer.is_attn]
    if len(gdns) != 48:
        raise ValueError('48 GDN layers required')
    for g in gdns:
        g.set_gates_compile(True)
        g.set_conv_norm_reuse_cache(True)
    if engine.runner.spec is not None:
        raise RuntimeError('factory requires a fresh ordinary engine')
    if mode == 'speculative':
        draft = load_drafter(str(draft_path), bounded_load=True)
        engine.runner = ModelRunner(engine.model, 64, draft, spec_block=5)
        engine.runner.spec.set_verify_transaction(True, eos_ids=engine.eos_ids, capacity=64, observer=None)
        engine.runner.spec.set_verify_broadcast_m1(True, logits_observer=None)
        if len(engine.runner.spec._verify_m1_ops114) != 497:
            raise RuntimeError('all497 qualified target Linears required')
    return engine


def build(mode, binding_path, *, execute_gpu=False):
    """Short19→32 only. One target load/fresh server; speculative adds one bounded draft load."""
    if execute_gpu is not True:
        raise RuntimeError('explicit root --execute-gpu required')
    binding = qualification(binding_path)  # Must precede every inference import.
    if os.environ.get('DYLD_INSERT_LIBRARIES'):
        raise RuntimeError('candidate116 has no native interposition')
    if any(os.environ.get(k) != v for k, v in CONTROLS.items()):
        raise RuntimeError('pinned50/50/20/0 controls required')
    if any(n == 'mlx' or n.startswith(('mlx.', 'tutorial_code.miniengine')) for n in sys.modules):
        raise RuntimeError('fresh process imports required')
    if {n: importlib.metadata.version(n) for n in ('mlx', 'mlx-metal')} != {'mlx': '0.32.3', 'mlx-metal': '0.32.3'}:
        raise RuntimeError('MLX0.32.3 required')
    import mlx.core as mx
    from mlx.utils import tree_flatten
    if mx.device_info().get('device_name') != 'Apple M4 Max' or mx.device_info().get('architecture') != 'applegpu_g16s':
        raise RuntimeError('qualified M4 Max required')
    mx.set_memory_limit(34 << 30)
    mx.set_cache_limit(2 << 30)
    sys.path.insert(0, str(ROOT))
    from tutorial_code.miniengine.cache import AttnCache, DecodeBatch
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from tutorial_code.miniengine.engine.model_runner import ModelRunner
    from tutorial_code.miniengine.spec.dflash2 import load_drafter
    from tutorial_code.miniengine.sampling_params import SamplingParams
    AttnCache.step = DecodeBatch.step = 64  # Before engine construction and any token, same117.
    protocol = json.loads((P / 'v117-verify-broadcast-m1-bounded-load-build1/protocol.json').read_text())
    e = LLMEngine(protocol['target_model_dir'], max_seqs=1, max_len=64, max_batched_tokens=512,
                  dtype=mx.bfloat16, cache_limit_gb=2)  # No drafter_dir: avoids unbounded loader.
    before = [(n, id(v)) for n, v in tree_flatten(e.model.parameters())]
    if len(before) != 1349:
        raise RuntimeError('1349 original target parameter identities required')
    assemble(e, mode, ModelRunner=ModelRunner, load_drafter=load_drafter, draft_path=protocol['draft_snapshot'])
    mx.synchronize()
    if before != [(n, id(v)) for n, v in tree_flatten(e.model.parameters())]:
        raise RuntimeError('target parameter identity changed')
    return e, SamplingParams, {'mode': mode, 'model_loads': 1, 'draft_loads': int(mode == 'speculative'),
        'bounded_load': mode == 'speculative', 'spec_block': 5 if mode == 'speculative' else None,
        'capacity': 64, 'cache_allocation_step': 64, 'target_parameter_count': 1349,
        'ordinary_ahead_retained': mode == 'ordinary', 'qualification_observer_callbacks': False,
        'intrinsic_candidate_transaction_metadata_retained': mode == 'speculative',
        'qualification_receipt_sha256': binding['receipt_sha256'], 'numerical_kernel_changes': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inert', action='store_true')
    args = ap.parse_args()
    print(json.dumps({'status': 'CPU_READY_SHORT_C1_ACTUAL117_BOUND_GPU_UNRUN', 'GPU_used': False,
        'thin_factory': str(D / 'adapter.py'), 'HTTP_draft': str(D / 'server.py'),
        'qualification_bound': True, 'ordinary_ahead_retained': True}))


if __name__ == '__main__':
    main()
