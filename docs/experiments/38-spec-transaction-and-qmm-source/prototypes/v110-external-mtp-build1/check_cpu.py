"""Actual source/API contracts only. Never imports MLX or simulates its values."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

D=Path(__file__).resolve().parent
P=D.parent
ROOT=P/'candidate-v110-external-mtp-build1-root'
BASE=P/'candidate-v44-conv-normreuse-build1-root'
Q=Path('tutorial_code/miniengine/model/qwen3_5.py')
NEW=ROOT/'tutorial_code/miniengine/spec/external_mtp.py'
contract_path=ROOT/'tutorial_code/miniengine/spec/mtp_contract.py'
spec=importlib.util.spec_from_file_location('v110_actual_contract',contract_path)
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)

def rejects(call):
    try:call()
    except (TypeError,ValueError):return
    raise AssertionError('invalid boundary accepted')

def method(source,name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)

def main():
    checks=0
    for f in list(ROOT.rglob('*.py'))+list(D.glob('*.py')):
        compile(f.read_text(),str(f),'exec');checks+=1
    for f in BASE.rglob('*'):
        if not f.is_file():continue
        rel=f.relative_to(BASE)
        if rel==Q:continue
        assert f.read_bytes()==(ROOT/rel).read_bytes(),rel;checks+=1
    a=(BASE/Q).read_text();b=(ROOT/Q).read_text()
    for name in ('__call__','lm_head_fn'):
        # Select methods of Qwen target, not earlier MLP.__call__.
        def target(src):
            cls=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5ForCausalLM')
            return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(target(a),include_attributes=False)==ast.dump(target(b),include_attributes=False);checks+=1
    # Removing exactly the new method restores the entire parent text.
    start=b.index('\n    def forward_with_hidden(')
    assert b[:start]==a;checks+=1
    metadata=json.loads((D/'checkpoint-schema.json').read_text())
    config,index,tensors=contract.checkpoint_metadata(metadata['draft_dir'])
    assert len(index)==len(tensors)==23 and len(contract.PROJECTIONS)==8 and len(contract.NORMS)==7;checks+=1
    for key,(dtype,shape) in contract.schema().items():
        assert tensors[key]['dtype']==dtype and tensors[key]['shape']==shape;checks+=1
    for v in (0,1,None,'true',[]):rejects(lambda:contract.require_bool(v));checks+=1
    assert contract.require_bool(False) is False and contract.require_bool(True) is True;checks+=1
    for appended,drafted,accepted in [(0,1,0),(1,2,0),(1,2,1),(1,2,2),(6,7,7),(7,7,0)]:
        plan=contract.accept_plan(appended,drafted,accepted,accepted+1,accepted+1)
        assert plan['trim']==appended-min(appended,accepted)
        assert plan['replay_draft_indices']==list(range(min(appended,accepted),accepted));checks+=1
    for args in [(1,2,1,1,2),(1,2,0,1,3),(3,2,0,1,1),(1,2,3,4,4),(True,2,0,1,1)]:
        rejects(lambda:contract.accept_plan(*args));checks+=1
    # Execute only the real stdlib adapter class with a fail-if-called loader.
    # This tests its OFF/fallback guards, not array/math/GPU behavior.
    tree=ast.parse(NEW.read_text());adapter=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ExternalMTPAdapter')
    def forbidden(*args,**kwargs):raise AssertionError('OFF path reached draft allocation/loader')
    namespace={'require_bool':contract.require_bool,'load_draft':forbidden}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[adapter],type_ignores=[])),str(NEW),'exec'),namespace)
    instance=namespace['ExternalMTPAdapter']()
    assert instance.enabled is False and instance.draft is None
    assert instance.prepare(None,None).try_draft(None,None) is None;checks+=1
    rejects(lambda:instance.prepare(None,None,enabled=1));checks+=1
    # Bind a callable sentinel only for dispatch rejection; no fake model arrays.
    class DispatchSentinel:
        def draft_block(self,*args):raise AssertionError('unsupported request dispatched')
    instance.draft=DispatchSentinel();instance.enabled=True
    for kwargs in ({'batch_size':2},{'batch_size':True},{'greedy':False},{'greedy':1},{'features':True},{'mode':'prefill'}):
        assert instance.try_draft(None,None,**kwargs) is None;checks+=1
    text=NEW.read_text()
    for name in ('_target','_borrowed','_parameter_identity','_cache','_seed_token','_seed_hidden','_last_drafts'):
        assert f"object.__setattr__(self, '{name}'" in text;checks+=1
    assert 'mx.dequantize' not in text and 'mx.quantize' not in text;checks+=1
    assert 'return self.norm(self.layers[0](h, ctx, 0))' in text and '.astype(mx.int32)[None]' in text;checks+=1
    assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
    result={'status':'CPU_SOURCE_CONTRACT_OFF_FALLBACK_ONLY','checks':checks,'GPU_used':False,'MLX_numpy_imported':False,
            'old27_source_inverse_except_added_hidden_method':True,'old_target_call_AST_same':True,'exact_header_keys':23,'projection_pairs':8,'BF16_norms':7,
            'numeric_or_parameter_identity_claim':False}
    print(json.dumps(result));return result
if __name__=='__main__':main()
