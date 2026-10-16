"""Local runtime/source identity; historical M4 binary identities are not reused."""
import hashlib, importlib.machinery, importlib.metadata, json, sys
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while b:=stream.read(1<<20):h.update(b)
 return h.hexdigest()

class VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
 def __init__(self,entry):self.entry=entry;super().__init__(entry['path'],entry['path'])
 def get_code(self,fullname):
  data=Path(self.entry['path']).read_bytes()
  if hashlib.sha256(data).hexdigest()!=self.entry['sha256']:raise ImportError('source changed '+fullname)
  return compile(data,self.entry['path'],'exec',dont_inherit=True)

def reject_preloaded(modules):
 if any(k=='mlx' or k.startswith(('mlx.','tutorial_code.')) or k=='tutorial_code' for k in modules):raise ValueError('runtime preloaded')

def preflight(f):
 if sys.version!=f['python_version']:raise ValueError('interpreter version changed')
 versions={k:importlib.metadata.version(k) for k in f['distributions']}
 if versions!=f['distributions']:raise ValueError('distribution versions changed')
 return {'python':sys.version,'executable':sys.executable,'distributions':versions}

def module_identity(name,module,entry):
 if Path(module.__file__).resolve()!=Path(entry['path']).resolve() or digest(module.__file__)!=entry['sha256']:raise ValueError('module identity changed '+name)
 return {'path':module.__file__,'sha256':entry['sha256']}

def runtime_identity(f,mx,nn,activations):
 result={}
 for name in ('mlx.core','mlx.nn','mlx.nn.layers.activations','numpy'):
  module=sys.modules[name];p=Path(module.__file__).resolve()
  if str(p) not in f['files'] or digest(p)!=f['files'][str(p)]:raise ValueError('unpinned module '+name)
  result[name]={'path':str(p),'sha256':f['files'][str(p)]}
 return result
