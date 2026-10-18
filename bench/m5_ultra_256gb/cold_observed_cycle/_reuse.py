"""Reuse frozen cold_cycle_v3 functions in a truthful new module namespace."""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType
HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'cold_cycle_v3'
FIXED={'controller.py': '674b21fbcafa0e79164dedb78875f513e9a24d478337c2aeeb67a439e69a4dc9', 'owner.py': '4c2cc6f305de3ecf0c6495e45838baae885efbf7a8f479ed2043dc0ffb67f31c', 'audit_restore.py': '34d608b54ed5779b97e85a0463c17f8796876be731f6f0aafcbe804ffc7b15ca'}

def install(namespace,name):
    path=ORIGINAL/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=FIXED[name]:
        raise ValueError('frozen lifecycle source changed')
    spec=importlib.util.spec_from_file_location('_observed_cold_original_'+path.stem,path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    namespace.update({k:v for k,v in vars(original).items() if not k.startswith('__')})
    namespace['HERE']=Path(namespace['__file__']).resolve().parent
    if name=='owner.py':namespace['CONTROLLER']=namespace['HERE']/'controller.py'
    for key,value in vars(original).items():
        if isinstance(value,FunctionType) and value.__globals__ is vars(original):
            f=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            f.__kwdefaults__=value.__kwdefaults__
            namespace[key]=f
    return original

def pins():
    result={str(ORIGINAL/n):h for n,h in FIXED.items()}
    result[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for p,h in result.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('frozen support changed')
    return result
