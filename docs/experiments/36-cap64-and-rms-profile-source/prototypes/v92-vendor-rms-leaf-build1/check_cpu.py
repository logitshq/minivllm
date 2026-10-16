"""CPU-only source/guard checks. No numerical MLX stand-in or GPU claim."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

D=Path(__file__).resolve().parent;P=D.parent
C=P/'candidate-v92-vendor-rms-leaf-build1-root'
BASE=P/'candidate-v44-conv-normreuse-build1-root'
REL='tutorial_code/miniengine/model/layers.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda node:ast.dump(node,include_attributes=False)

def check():
    checks=[]
    old=ast.parse((BASE/REL).read_text());new=ast.parse((C/REL).read_text())
    classes=lambda t:{n.name:n for n in t.body if isinstance(n,ast.ClassDef)}
    oc,nc=classes(old),classes(new)
    assert dump(oc['GatedRMSNorm'])==dump(nc['GatedRMSNorm']);checks.append('GatedRMSNorm_AST_exact')
    stripped=copy.deepcopy(nc['RMSNorm'])
    stripped.body=[n for n in stripped.body if not isinstance(n,ast.FunctionDef) or n.name in ('__init__','__call__')]
    for n in stripped.body:
        if isinstance(n,ast.FunctionDef) and n.name=='__init__':n.body=n.body[:-4]
        if isinstance(n,ast.FunctionDef) and n.name=='__call__':n.body=n.body[1:]
    assert dump(stripped)==dump(oc['RMSNorm']);checks.append('remove_profile_restores_RMSNorm_AST_exact')
    files=sorted(BASE.rglob('*.py'));assert len(files)==27
    assert all((C/p.relative_to(BASE)).is_file() for p in files)
    assert all(p.relative_to(BASE).as_posix()==REL or p.read_bytes()==(C/p.relative_to(BASE)).read_bytes() for p in files)
    checks.append('26_other_Python_files_byte_exact_to44')
    for p in C.rglob('*.py'):compile(p.read_text(),str(p),'exec')
    for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
    checks.append('all_candidate_and_aux_compile_without_import')
    # Execute only the actual isolated strict-bool method, never model math.
    method=next(n for n in nc['RMSNorm'].body if isinstance(n,ast.FunctionDef) and n.name=='set_vendor_rms_leaf')
    tree=ast.Module(body=[copy.deepcopy(method)],type_ignores=[]);ns={};exec(compile(ast.fix_missing_locations(tree),'bool-guard','exec'),ns)
    class Holder:pass
    h=Holder()
    for value in (True,False):ns['set_vendor_rms_leaf'](h,value);assert h._vendor_rms_leaf_enabled is value
    for value in (None,0,1,[],{},'true'):
        try:ns['set_vendor_rms_leaf'](h,value)
        except TypeError:pass
        else:raise AssertionError('non-bool accepted')
    checks.append('actual_strict_bool_method_2accepted_6rejected')
    source=(C/REL).read_text()
    for assignment in ('_vendor_rms_leaf_gamma','_vendor_rms_leaf_weight','_vendor_rms_leaf_signature','_vendor_rms_leaf_enabled'):
        for node in ast.walk(new):
            if isinstance(node,ast.Attribute) and isinstance(node.ctx,ast.Store):assert node.attr!=assignment
    assert 'weight is self.weight' in source
    checks.append('profile_arrays_only_object_setattr_identity_guard_present')
    # Import independent helper: no MLX/NumPy imported. Inspect actual context rollback.
    forbidden_before={k for k in sys.modules if k.split('.')[0] in ('mlx','numpy')}
    spec=importlib.util.spec_from_file_location('v92_ref_cpu',D/'vendor_reference.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    assert {k for k in sys.modules if k.split('.')[0] in ('mlx','numpy')}==forbidden_before
    class Plain:
        def __call__(self,x):return x
    original=Plain.__call__
    profile=m.PreparedProfile(None,[],Plain,original,(),None,{})
    profile.validate_current_bindings=lambda:True
    for raise_inside in (False,True):
        try:
            with profile.reference_context():
                assert Plain.__call__ is not original
                if raise_inside:raise LookupError('intentional rollback check')
        except LookupError:pass
        assert Plain.__call__ is original and not profile._in_reference
    with profile.reference_context():
        try:
            with profile.reference_context():pass
        except RuntimeError:pass
        else:raise AssertionError('nested context accepted')
    assert Plain.__call__ is original
    checks.append('reference_context_repeat_exception_nested_rollback_guards')
    plan=json.loads((D/'raw-gamma-specs.json').read_text());arrays=plan['arrays']
    assert len(arrays)==len({s['key'] for s in arrays})==161
    assert sum(s['nbytes'] for s in arrays)==1337344
    assert sum(s['shape']==[5120] for s in arrays)==129 and sum(s['shape']==[256] for s in arrays)==32
    assert all(s['dtype']=='BF16' and s['nbytes']==s['shape'][0]*2 for s in arrays)
    assert not any('.linear_attn.norm.' in s['key'] for s in arrays)
    checks.append('161_raw_specs_129x5120_32x256_excludes_gatednorm')
    pins_path=D/'source-pins.json'
    if pins_path.exists():
        pins=json.loads(pins_path.read_text());assert all(sha(p)==h for p,h in pins.items())
        checks.append('all_frozen_source_pins_match')
    return {'status':'CPU_SOURCE_AST_GUARDS_INERT_PASS_ONLY','checks':checks,'MLX_imported':False,
            'GPU_executed':False,'numerical_qualified':False,'files_checked':27}

if __name__=='__main__':
    if sys.argv[1:] not in ([],['--inert']):raise SystemExit('CPU only; optional --inert')
    print(json.dumps(check(),indent=2))
