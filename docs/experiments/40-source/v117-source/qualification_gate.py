"""Fail closed before MLX until successful V113 + release + independent review."""
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require_v113():
    b=json.loads((D/'v113-binding.json').read_text())
    if b['status']!='BOUND_ACTUAL_V113_STRICT_PASS_AND_INDEPENDENT_REVIEW':
        raise RuntimeError('HOLD_V114_V113_ACTUAL_PASS_RELEASE_INDEPENDENT_REVIEW_NOT_BOUND')
    for key in ('receipt','source_pins','release','independent_review'):
        if not b.get(key) or sha(b[key])!=b[key+'_sha256']:
            raise RuntimeError('HOLD_V114_missing_or_changed_'+key)
    q=json.loads(Path(b['receipt']).read_text());release=json.loads(Path(b['release']).read_text());review=json.loads(Path(b['independent_review']).read_text())
    assert q['status']=='PASS_V113_ALL497_SAME_INPUT_BROADCAST_M1_STRICT_NUMERIC_ONLY'
    assert q['broadcastM1_vs_rowM1_strict_equal_count']==497 and q['broadcastM1_vs_rowM1_strict_different_count']==0
    assert len(q['comparisons'])==1491 and q['original112_all497_input_output_hashes_reproduced']
    assert q['keep0_restored_all128'] and q['target1349_identity_readonly'] and q['source_identity_final']
    rows=[r for r in q['comparisons'] if r['variant']=='broadcastM1_vs_rowM1']
    assert len(rows)==497 and all(r['strict_equal'] and r['left_sha256']==r['right_sha256'] for r in rows)
    pins=json.loads(Path(b['source_pins']).read_text());assert len(pins)==100 and q['pins']==pins
    helper=Path(b['source_pins']).parent/'broadcast_m1.py'
    assert sha(helper)==b['helper_sha256']=='77ac3b6a18801f20672d7a3395e4be96c468d6db28ed9553b33f76d0011aab95'
    assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==0
    assert all(r['returncode']==1 and r['stdout']==r['stderr']=='' for r in release['strict_absence'])
    assert review['status']=='PASS_INDEPENDENT_V113_TWO_ATTEMPT_SAVED_RESULTS_AUDIT'
    assert review['attempt2']['receipt_sha256']==b['receipt_sha256']
    assert review['attempt2']['classification']==q['status']
    assert review['attempt2']['comparisons']==1491 and review['attempt2']['reaped_and_external_absence']
    return b
