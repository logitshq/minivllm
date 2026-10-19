"""Fixed local function rebinding; no donor writes or service/backend imports."""
import ast, hashlib, importlib.util, inspect
from pathlib import Path
from types import FunctionType
ROOT=Path(__file__).resolve().parent.parent
FIXED={'glm_flash_cycle/audit_closed.py': '0158be373f614c55ed03393320337b7cca4c1525360fd850f5081a5462694d48', 'glm_flash_cycle/audit_restore.py': '732c60870b9400d1d7b6bfb9c303bce19b9c92032bc06bc9db193b66ace3bf6d', 'glm_flash_cycle/controller.py': '98bc920d97b529c07f9af0a64213e47e73728b63caee2e867e496f1ac30bde88', 'glm_flash_cycle/execute.py': 'cbc28848a124c74fabd9aede971b57b8fb9162051031e9a720dde5f87029635a', 'glm_flash_cycle/owner.py': '37f47300c14d8da721d6c30fcc71f52325132bad3dc3ba902c1cc08af4326b40', 'glm_flash_cycle/prepare_inputs.py': 'da10a83b826150b88f44e1442423660b9e7edd990114e368fe2fea98032207fb', 'glm_flash_cycle/preregister.py': '7f937f709b0c6806a0dfca7606122e1178b2c495b4574665e7df1eea2862e842', 'glm_flash_cycle_v3/audit_closed.py': '59775b95f8b575cc2314f281238d77e36485d4de4cb49787fe55907d041c2396', 'glm_flash_cycle_v3/audit_restore.py': 'fca6b668df7240ac12ce47380f99af5e670eecd885af4d21d44462268f3c7bab', 'glm_flash_cycle_v3/controller.py': 'ca35f97e5ee39d3ca81a59ab851249423d3317fd8176f569d8580260328d5731', 'glm_flash_cycle_v3/execute.py': '1dacd788206b71765f81cf94270e0a958daeaaf2032b7b0bf414b410528b56a8', 'glm_flash_cycle_v3/owner.py': '9d69c5a64dc50b7d65330ad130b8af9f757b91e884aa197d1e828775b27c88a8', 'glm_flash_cycle_v3/preregister.py': '319df1f18efee2ec5587d353c213a79aa8e7373ece904c13e1fd0e06529f68af', 'c1_profile_cycle_v1/_reuse.py': '8bed13ebb236e68bc3c5b8407b72de116a5d6413687a9b3b538da36f5e5dcc48', 'c1_profile_cycle_v1/audit_closed.py': '6c5e1d28bc61065d803d7ddcfe39c4ac698690060a17051778b32b164936af2d', 'c1_profile_cycle_v1/audit_restore.py': '76f93301de2ea4016afd04841014925d0b66e5ea43fa4e601e70b9db65d2d593', 'c1_profile_cycle_v1/controller.py': '318ef9a8f3615bfebaea46bca203ec456876b319367c4d9f477144dbb5cfc467', 'c1_profile_cycle_v1/execute.py': '64fc9dc8bfd551108bcfddf7ac2a631981c95622aeac65e55a7e0f5e2b30e6a8', 'c1_profile_cycle_v1/owner.py': 'ef7daf7bd56a045d7ea81f0808dafe283c2c91430463f26f57aa12cb34a0ce15', 'c1_profile_cycle_v1/preregister.py': '97e154fb280daf0437412ece71635b622b9938eb0a8858dad89b3d8a87e7fb9a', 'glm_flash_cycle_v2/_reuse.py': 'fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97', 'c1_profile_cycle_v1/predecessor-manifest.json': 'ad1254e01dbd610d29f8abdebdd80db9c416f2c34ec5d8df61990bc54df84075'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixed(name):
 p=ROOT/name
 if name not in FIXED or sha(p)!=FIXED[name]:raise ValueError('fixed source changed: '+str(p))
 sp=importlib.util.spec_from_file_location('_stacked_'+name.replace('/','_').replace('.','_'),p)
 m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def bind(fn,ns):
 out=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
 out.__kwdefaults__=fn.__kwdefaults__;return out
def install(ns,name):
 m=fixed(name);ns.update({k:v for k,v in vars(m).items() if not k.startswith('__')})
 ns['HERE']=Path(ns['__file__']).resolve().parent
 for k,v in vars(m).items():
  if isinstance(v,FunctionType) and v.__globals__ is vars(m):ns[k]=bind(v,ns)
 return m
def variant(ns,source,changes):
 for before,after in changes:
  if source.count(before)!=1:raise ValueError('exact fixed source delta changed: '+before)
  source=source.replace(before,after,1)
 tree=ast.parse(source)
 if len(tree.body)!=1 or not isinstance(tree.body[0],(ast.FunctionDef,ast.AsyncFunctionDef)):raise ValueError('one function')
 exec(compile(tree,ns['__file__'],'exec'),ns);return source
def pins():
 out={str(ROOT/name):h for name,h in FIXED.items()}
 if not all(sha(p)==h for p,h in out.items()):raise ValueError('fixed reuse source changed')
 out[str(Path(__file__).resolve())]=sha(__file__);return out
