"""V117 stdlib source inverse, before-MLX HOLD, inherited semantic boundaries."""
import ast
import json
from pathlib import Path
import sys
D=Path(__file__).resolve().parent;P=D.parent
OLD=P/'v114-verify-broadcast-m1-build1'
class Inverse(ast.NodeTransformer):
    def visit_ImportFrom(self,n):return None if n.module=='loader_gate' else n
    def visit_Assign(self,n):
        if any(isinstance(t,ast.Name) and t.id=='loader_prerequisite' for t in n.targets):return None
        return self.generic_visit(n)
    def visit_FunctionDef(self,n):
        if n.name=='bounded_loader_stage':return None
        return self.generic_visit(n)
    def visit_Dict(self,n):
        keep=[i for i,k in enumerate(n.keys) if not(isinstance(k,ast.Constant) and k.value in ('V116_binding','V116_loader_prerequisite','bounded_loader_stages'))]
        n.keys=[n.keys[i] for i in keep];n.values=[n.values[i] for i in keep]
        return self.generic_visit(n)
    def visit_Call(self,n):
        if isinstance(n.func,ast.Name) and n.func.id=='load_drafter':n.keywords=[k for k in n.keywords if k.arg not in ('bounded_load','stage_observer')]
        return self.generic_visit(n)
    def visit_Constant(self,n):
        if isinstance(n.value,str):
            n.value=n.value.replace('V117 root-only full27B','V114 root-only full27B').replace('PASS_V117_BOUNDED_LOAD_VERIFY_BROADCAST_M1_OLD96IDS_LITERAL_AND_ORDINARY_M1_COMMIT128_LASTLOGITS_STRICT','PASS_V114_VERIFY_BROADCAST_M1_OLD96IDS_LITERAL_AND_ORDINARY_M1_COMMIT128_LASTLOGITS_STRICT')
        return n

def main():
    checks=0
    for f in D.glob('*.py'):compile(f.read_text(),str(f),'exec');checks+=1
    old=ast.parse((OLD/'fullqual.py').read_text());new=Inverse().visit(ast.parse((D/'fullqual.py').read_text()))
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False);checks+=1
    for name in ('ordinary_reference.py','qualification_gate.py','v113-binding.json','assertion-gold.json','header-pins.json'):
        assert (D/name).read_bytes()==(OLD/name).read_bytes();checks+=1
    a=(OLD/'runtime_support.py').read_text();b=(D/'runtime_support.py').read_text().replace('V117 numerical qualification','V114 numerical qualification').replace('candidate-v116-bounded-draftloader-build1-root','candidate-v114-verify-broadcast-m1-build1-root')
    assert a==b;checks+=1
    source=(D/'fullqual.py').read_text();assert source.index('loader_prerequisite=require_v116()')<source.index('import mlx.core as mx');checks+=1
    assert 'bounded_load=True,stage_observer=bounded_loader_stage' in source;checks+=1
    from qualification_gate import require_v113
    require_v113();checks+=1
    from loader_gate import require_v116
    binding=json.loads((D/'v116-binding.json').read_text())
    if binding['status'].startswith('HOLD_'):
        try:require_v116()
        except RuntimeError as e:assert 'HOLD_V117' in str(e)
        else:raise AssertionError('unbound116review accepted')
    else:require_v116()
    checks+=1
    assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
    print(json.dumps({'status':'PASS_V117_CPU_HARNESS_INVERSE_AND_PREREQUISITE_GUARDS_ONLY','checks':checks,'GPU_used':False,'MLX_NumPy_imported':False,'V116_binding':binding['status']}))
if __name__=='__main__':main()
