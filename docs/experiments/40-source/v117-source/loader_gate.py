"""V117 stdlib prerequisite gate, before any MLX import."""
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require_v116():
    b=json.loads((D/'v116-binding.json').read_text())
    if b['status']!='BOUND_V116_ACTUAL175_STRICT_RELEASE_AND_INDEPENDENT_REVIEW':
        raise RuntimeError('HOLD_V117_V116_INDEPENDENT_REVIEW_NOT_BOUND')
    for key in ('receipt','source_pins','release','independent_review'):
        assert b.get(key) and sha(b[key])==b[key+'_sha256'],('missing/changed116proof',key)
    q=json.loads(Path(b['receipt']).read_text());release=json.loads(Path(b['release']).read_text());review=json.loads(Path(b['independent_review']).read_text())
    assert q['status']=='PASS_V116_BOUNDED_DRAFT_LOADER_175_PARAMETERS_STRICT_A_READONLY'
    assert q['strict_parameters']==175 and q['A175_readonly'] and q['independent_output_owner_objects']==350 and q['source_identity_final']
    assert not q['target_model_loaded'] and q['draft_loads']==[{'arm':'A','bounded_load':False},{'arm':'B','bounded_load':True}]
    assert len(q['comparisons'])==175 and all(r['strict_equal'] and r['A_readonly'] and r['different_objects'] and r['A']==r['B'] for r in q['comparisons'])
    assert len(q['quantized47_configuration'])==47 and all(v=={'group_size':64,'bits':4,'mode':'affine'} for v in q['quantized47_configuration'].values())
    assert q['pins']==json.loads(Path(b['source_pins']).read_text())
    loader=Path(b['candidate_root'])/'tutorial_code/miniengine/spec/dflash2.py'
    assert q['pins'][str(loader)]==b['loader_sha256']==sha(loader)
    assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==0
    assert all(r['returncode']==1 and r['stdout']==r['stderr']=='' for r in release['strict_absence'])
    assert review['status']=='PASS_INDEPENDENT_V116_SAVED_RESULTS_AND_SOURCE_REPLAY'
    assert review['actual_receipt_sha256']==b['receipt_sha256'] and review['actual_release_sha256']==b['release_sha256']
    assert review['strict_parameters']==175 and review['A175_readonly'] and review['independent_owner_objects']==350
    assert review['source_pins_verified']==78 and review['B_processed_sources']==81 and review['B_quantized_branches']==47
    assert review['allocator_stages']==88 and review['review_GPU_used'] is False
    return b
