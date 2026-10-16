"""Small local device qualification, not a model or performance run."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mlx.core as mx, mlx.nn as nn
import numpy as np
from tutorial_code.miniengine.kernels import gdn_norm_reuse_metal as n
from tutorial_code.miniengine.kernels.gdn_metal import gated_delta_step_metal
mx.random.seed(256)
u=(mx.random.normal((1,10240))*.1).astype(mx.bfloat16);g=mx.full((1,48),-.1);beta=mx.full((1,48),.5);S=mx.random.normal((1,48,128,128))*.01
q,k,v=mx.split(nn.silu(u),[2048,4096],axis=-1);q=q.reshape(1,16,128).astype(mx.float32);k=k.reshape(1,16,128).astype(mx.float32);v=v.reshape(1,48,128).astype(mx.float32)
qa,ka,va,scales=n._prepare(u,mx)
def same(a,b):return np.asarray(a.astype(mx.float32)).tobytes()==np.asarray(b.astype(mx.float32)).tobytes()
assert all(same(a,b) for a,b in zip((q,k,v),(qa,ka,va))),'BF16 intermediate rounding differs'
a=gated_delta_step_metal(mx.repeat(q,3,axis=1),mx.repeat(k,3,axis=1),v,g,beta,S);b=n.gated_delta_step_norm_reuse_metal(u,g,beta,S)
assert all(same(x,y) for x,y in zip(a,b)),'small recurrent kernel exact mismatch'
w=mx.random.normal((32,128)).astype(mx.bfloat16);packed=mx.quantize(w,group_size=16,bits=4,mode='nvfp4');x=mx.ones((1,128),mx.bfloat16);y=mx.quantized_matmul(x,*packed,group_size=16,bits=4,mode='nvfp4',transpose=True);mx.eval(y);assert y.shape==(1,32) and bool(mx.all(mx.isfinite(y)))
print(json.dumps({'device':mx.device_info(),'metal':mx.metal.is_available(),'BF16_intermediate_exact':True,'norm_reuse_small_recurrent_exact':True,'nvfp4_group16':True,'compile_options':n.COMPILE_OPTIONS,'prework_grid':[8192,1,1],'prework_threadgroup':[128,1,1],'state_grid':[128,192,1],'state_threadgroup':[32,4,1]},indent=2))
