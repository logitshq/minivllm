"""Small-array utilities; no GPU/MLX/NumPy imports at import time."""
import hashlib,importlib.metadata,importlib.util,json,math,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent
BASE=P/'candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine/kernels/gdn_metal.py'
BANKROOT=P/'v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1'
CONTRACTS={'u':((1,10240),'bfloat16'),'g':((1,48),'float32'),'beta':((1,48),'float32'),'S':((1,48,128,128),'float32')}
ONE_STEP=2**-16;CHAIN=2**-12;CHAIN_STEPS=16
CONTROLS={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def pins():
 expected=json.loads((D/'source-pins.json').read_text())
 for path,value in expected.items():
  if sha(path)!=value:raise ValueError('source changed: '+path)
 return expected
def runtime():
 versions={n:importlib.metadata.version(n) for n in ('mlx','mlx-metal')}
 if set(versions.values())!={'0.32.3'}:raise ValueError('requires MLX/mlx-metal0.32.3')
 import mlx.core as mx
 import mlx.nn as nn
 import numpy as np
 info=mx.device_info()
 if mx.default_device()!=mx.gpu or info.get('device_name')!='Apple M4 Max' or info.get('architecture')!='applegpu_g16s':raise ValueError('requires actual M4 Max GPU')
 mx.set_memory_limit(512<<20);mx.set_cache_limit(64<<20)
 return mx,nn,np,{'versions':versions,'device':info,'memory_limit':512<<20,'cache_limit':64<<20}
def load_bank(directory,mx,np):
 directory=Path(directory);meta=json.loads((directory/'bank.json').read_text());result={'name':directory.name,'provenance':{'bank':str(directory),'bank_sha256':sha(directory/'bank.json')}}
 for name,(shape,dtype) in CONTRACTS.items():
  rec=meta['arrays'][name];f=directory/rec['file'];width=2 if dtype=='bfloat16' else 4
  if Path(rec['file']).name!=rec['file'] or tuple(rec['shape'])!=shape or rec['dtype']!=dtype or f.stat().st_size!=math.prod(shape)*width or sha(f)!=rec['sha256']:raise ValueError('invalid saved '+name)
  a=mx.array(np.fromfile(f,dtype='<u2' if width==2 else '<f4').reshape(shape));result[name]=a.view(mx.bfloat16) if width==2 else a
 return result
def compact(u,mx,nn):
 act=nn.silu(u);q,k,v=mx.split(act,[2048,4096],axis=-1);B=u.shape[0]
 return q.reshape(B,16,128),k.reshape(B,16,128),v.reshape(B,48,128)
def reference(q,k,v,g,beta,S,mx,baseline):
 return baseline.gated_delta_step_metal(mx.repeat(q,3,axis=1),mx.repeat(k,3,axis=1),v,g,beta,S)
def transposed_storage(S,mx):return mx.swapaxes(mx.contiguous(mx.swapaxes(S,-1,-2)),-1,-2)
def host(a,mx,np):
 mx.eval(a)
 if a.dtype==mx.bfloat16:
  h=np.ascontiguousarray(np.asarray(a.view(mx.uint16)),dtype='<u2')
  if np.any((h&0x7f80)==0x7f80):raise ValueError('nonfinite BF16')
 else:
  h=np.ascontiguousarray(np.asarray(a),dtype='<f4')
  if not np.isfinite(h).all():raise ValueError('nonfinite FP32')
 return h
def hashes(values,mx,np):return [hashlib.sha256(host(a,mx,np).tobytes()).hexdigest() for a in values]
def compare(a,b,limit,mx,np):
 if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('output shape/dtype mismatch')
 ha,hb=host(a,mx,np),host(b,mx,np);diff=np.abs(ha.astype(np.float64)-hb.astype(np.float64));scale=max(1.,float(np.max(np.abs(ha))))
 error=float(np.max(diff));ok=error<=limit*scale
 row={'shape':list(a.shape),'dtype':str(a.dtype),'max_abs':error,'reference_max_abs':scale,'scaled_max_abs':error/scale,'limit':limit,'normalized_l2':float(np.linalg.norm(diff.ravel())/max(1.,np.linalg.norm(ha.astype(np.float64).ravel()))),'pass':ok,'reference_sha256':hashlib.sha256(ha.tobytes()).hexdigest(),'candidate_sha256':hashlib.sha256(hb.tobytes()).hexdigest()}
 if not ok:raise ValueError('fixed numerical tolerance failed: '+str(row))
 return row
def layout_probe(mx):
 src='''if (thread_position_in_grid.x == 0) { for (uint d=0;d<4;++d) { layout[d]=ulong(S_shape[d]); layout[4+d]=ulong(S_strides[d]); } }'''
 kernel=mx.fast.metal_kernel(name='v27_state_descriptor',input_names=['S'],output_names=['layout'],source=src,ensure_row_contiguous=False,compile_options={'math_mode':'safe'})
 def read(S):
  a,=kernel(inputs=[S],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(8,)],output_dtypes=[mx.uint64]);mx.eval(a);v=a.tolist();return {'shape':v[:4],'strides_elements':v[4:],'ensure_row_contiguous':False}
 return read
