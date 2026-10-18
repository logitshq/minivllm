"""Fixed GLM v1 code reuse. No backend, service action or donor mutation.

Only functions defined by the selected frozen module are rebound to the actual
new module dictionary. Imported helpers/classes retain their original identity.
Controller's three dependency-bound methods are rebuilt explicitly there.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'glm_flash_cycle'
FIXED={'owner.py': '37f47300c14d8da721d6c30fcc71f52325132bad3dc3ba902c1cc08af4326b40', 'controller.py': '98bc920d97b529c07f9af0a64213e47e73728b63caee2e867e496f1ac30bde88', 'audit_restore.py': '732c60870b9400d1d7b6bfb9c303bce19b9c92032bc06bc9db193b66ace3bf6d', 'audit_closed.py': '0158be373f614c55ed03393320337b7cca4c1525360fd850f5081a5462694d48', 'preregister.py': '7f937f709b0c6806a0dfca7606122e1178b2c495b4574665e7df1eea2862e842', 'execute.py': 'cbc28848a124c74fabd9aede971b57b8fb9162051031e9a720dde5f87029635a', 'prepare_inputs.py': 'da10a83b826150b88f44e1442423660b9e7edd990114e368fe2fea98032207fb'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name):
    path=ORIGINAL/name
    if name not in FIXED or sha(path)!=FIXED[name]:raise ValueError('fixed original GLM source changed: '+str(path))
    spec=importlib.util.spec_from_file_location('_glm_v2_original_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def bind(function,namespace):
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    result.__annotations__=dict(function.__annotations__)
    return result

def install(namespace,name,overrides=None):
    original=load(name)
    for key,value in vars(original).items():
        if not key.startswith('__'):namespace[key]=value
    namespace['HERE']=Path(namespace['__file__']).resolve().parent
    if overrides:namespace.update(overrides)
    for key,value in vars(original).items():
        if isinstance(value,FunctionType) and value.__globals__ is vars(original):
            namespace[key]=bind(value,namespace)
    return original

def support_pins():
    pins={str(ORIGINAL/name):digest for name,digest in FIXED.items()}
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('frozen GLM support changed: '+path)
    pins[str(Path(__file__).resolve())]=sha(__file__)
    return pins
