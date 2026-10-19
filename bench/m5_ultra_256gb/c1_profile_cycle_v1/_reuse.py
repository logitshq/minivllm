"""Fixed source reuse; new globals only, no donor mutation or device import."""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import FunctionType

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FIXED={
 'gdn_graph_combined_formal/audit_formal.py':'b1956036eb7c6df3939e68e2ba5b69b965bfb80951c694107325adb6feed3e5f',
 'lithos_d512_formal/audit_resources.py':'398ff7a52dba307418f4c2725f33988b1571d65b11e3f46c9e97759f3a167c9e',
 'gdn_graph_combined_formal/owner.py':'511b48fbe722ea031a7b4586439eaa67f25c9c9514af12b614de74fb8c53c103',
 'gdn_graph_combined_formal/controller.py':'3b8b3033269fcd326e0a52e20470ee27faf65d313be3e3bf0da8881f26f3ac4c',
 'gdn_graph_combined_formal/audit_restore.py':'157d6470a9cb87223c419a3f4edd82e8556b3714a7157ea1ad505270c8ed7493',
 'glm_flash_cycle/preregister.py':'7f937f709b0c6806a0dfca7606122e1178b2c495b4574665e7df1eea2862e842',
 'glm_flash_cycle/execute.py':'cbc28848a124c74fabd9aede971b57b8fb9162051031e9a720dde5f87029635a',
 'glm_flash_cycle_v3/controller.py':'ca35f97e5ee39d3ca81a59ab851249423d3317fd8176f569d8580260328d5731',
 'glm_flash_cycle_v3/audit_restore.py':'fca6b668df7240ac12ce47380f99af5e670eecd885af4d21d44462268f3c7bab',
 'c1_pipeline_profile/capture.py':'9f05a8c813dfa787b61df715b3dac97ecdecb2bce333f9486cd86c67edb7e68e',
 'c1_pipeline_profile/owner.py':'da9936416135b10583f6a3c40b94662a473fa0b2f0f61340ddfe9e813d372ad3'}


def need(ok,message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,digest):
    path=Path(path);need(sha(path)==digest,'fixed source changed: '+str(path))
    spec=importlib.util.spec_from_file_location('_c1_cycle_'+path.parent.name+'_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def fixed(name): return load(ROOT/name,FIXED[name])
def local(name): return load(HERE/name,sha(HERE/name))
def bind(fn,namespace):
    result=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    result.__kwdefaults__=fn.__kwdefaults__;return result
def install(namespace,name):
    original=fixed(name)
    namespace.update({k:v for k,v in vars(original).items() if not k.startswith('__')})
    namespace['HERE']=Path(namespace['__file__']).resolve().parent
    for name,value in vars(original).items():
        if isinstance(value,FunctionType) and value.__globals__ is vars(original):namespace[name]=bind(value,namespace)
    return original
def variant(namespace,function,replacements):
    source=inspect.getsource(function)
    for before,after in replacements:
        need(source.count(before)==1,'fixed function anchor changed: '+function.__name__);source=source.replace(before,after,1)
    tree=ast.parse(source);need(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef),'one adapted function')
    exec(compile(tree,namespace['__file__'],'exec'),namespace);return source
def pins():
    result={str(ROOT/name):digest for name,digest in FIXED.items()}
    need(all(sha(path)==digest for path,digest in result.items()),'fixed reuse source changed')
    return {**result,**{str(p):sha(p) for p in HERE.glob('*.py')},str(HERE/'predecessor-manifest.json'):sha(HERE/'predecessor-manifest.json')}
