"""Root-only true feedback tiny: dynamic token/position/lens/state, then reset."""
import ctypes
import json
from pathlib import Path
import sys

out = Path(sys.argv[1]); out.mkdir()
lib = ctypes.CDLL(None)
u, ptr, st = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_char_p
for name, args in {
    'v33_pin': [u,u,st], 'v33_begin': [], 'v33_end_build': [],
    'v33_register_output': [u,u,st], 'v34_register_input': [u,u,ctypes.c_uint32,st],
    'v34_link': [u,u,u], 'v36_register_pingpong': [u,u,u], 'v34_seal_feedback': [ctypes.c_uint32,ctypes.c_uint32],
    'v34_reset': [], 'v34_write_input': [u,u,ptr], 'v34_replay_step': [],
    'v34_compare_output': [u,u,ptr], 'v34_copy_output': [u,u,ptr],
}.items():
    getattr(lib,name).argtypes=args;getattr(lib,name).restype=ctypes.c_int
lib.v34_report_json.argtypes=[];lib.v34_report_json.restype=st

def report(label):
    r=json.loads(lib.v34_report_json().decode())
    (out/(label+'.json')).write_text(json.dumps(r,indent=2)+'\n');return r

def call(name,*args):
    rc=getattr(lib,name)(*args)
    if rc:
        report('native-failure');raise RuntimeError(f'{name} rc={rc}')

import mlx.core as mx
import numpy as np

class Shared:
    def __init__(self,value):
        self.owner=value;mx.eval(value);self.mv=memoryview(value)
        assert self.mv.c_contiguous
        self.raw=self.mv.cast('B');self.np=np.frombuffer(self.raw,dtype=np.uint8)
        self.address=int(self.np.__array_interface__['data'][0]);self.nbytes=int(value.nbytes)

sys.path.insert(0,str(Path('/private/tmp/minivllm-engine-optimization-20261014/candidate-v17-gdn-gates-build1-root')))
from tutorial_code.miniengine.kernels.gdn_metal import gated_delta_step_metal
q=mx.sin(mx.arange(6144,dtype=mx.float32)).reshape(1,48,128)*0.1
k=mx.cos(mx.arange(6144,dtype=mx.float32)).reshape(1,48,128)*0.1
v=mx.sin(mx.arange(6144,dtype=mx.float32)*0.1).reshape(1,48,128)
g=mx.full((1,48),-0.05,dtype=mx.float32);beta=mx.full((1,48),0.25,dtype=mx.float32)
seeds=(mx.arange(786432,dtype=mx.float32).reshape(1,48,128,128)/1000000,mx.array([2],mx.int32),mx.array([3],mx.int32),mx.array([4],mx.int32))
mx.eval(q,k,v,g,beta,*seeds);mx.synchronize()
weights=[Shared(a) for a in (q,k,v,g,beta)];inputs=[Shared(a) for a in seeds]
unchanged=[bytes(h.raw) for h in weights+inputs]
for i,h in enumerate(weights):call('v33_pin',h.address,h.nbytes,f'weight{i}'.encode())

def graph(state,token,position,lens):
    o,new_state=gated_delta_step_metal(q,k,v,g,beta,state)
    logits=o.reshape(-1)[:16]+mx.arange(16,dtype=mx.float32)*token.astype(mx.float32)*0.01
    pred=mx.argmax(logits,keepdims=True).astype(mx.int32)
    return logits,new_state,pred,position+1,lens+1

call('v33_begin');captured=graph(*seeds);mx.eval(*captured);mx.synchronize();call('v33_end_build')
outputs=[Shared(v) for v in captured]
captured_raw=[bytes(h.raw) for h in outputs]
for i,h in enumerate(inputs):call('v34_register_input',h.address,h.nbytes,i,f'input{i}'.encode())
for i,h in enumerate(outputs):call('v33_register_output',h.address,h.nbytes,f'output{i}'.encode())
for i,j in enumerate([1,2,3,4]):call('v34_link',inputs[i].address,outputs[j].address,inputs[i].nbytes)
call('v36_register_pingpong',inputs[0].address,outputs[1].address,inputs[0].nbytes)
call('v34_seal_feedback',1,256)
pp=report('native-sealed')['feedback']['pingpong']
assert pp['eligible_count']==1 and pp['staged_link_count']==3, pp
requests=[]
for request in range(3):
    call('v34_reset')
    reference=seeds
    seed_overrides=[]
    if request==2:
        reference=(seeds[0]+2,mx.array([7],mx.int32),mx.array([8],mx.int32),mx.array([9],mx.int32))
        seed_overrides=[Shared(v) for v in reference]
        for target,source in zip(inputs,seed_overrides):call('v34_write_input',target.address,target.nbytes,ctypes.c_void_p(source.address))
    tokens=[]
    for step in range(5):
        expected=graph(*reference);mx.eval(*expected)
        refs=[Shared(v) for v in expected]
        call('v34_replay_step')
        for original,actual in zip(outputs,refs):call('v34_compare_output',original.address,original.nbytes,ctypes.c_void_p(actual.address))
        token=ctypes.c_int32();call('v34_copy_output',outputs[2].address,4,ctypes.byref(token))
        assert token.value==int(expected[2].item());tokens.append(token.value)
        reference=(expected[1],expected[2],expected[3],expected[4])
    assert [bytes(h.raw) for h in weights+inputs]==unchanged
    assert [bytes(h.raw) for h in outputs]==captured_raw
    requests.append(tokens)
assert requests[0]==requests[1] # Changed S bytes are checked every step even if greedy token is unchanged.
native=report('native-final')
r={'status':'PASS_TINY_GDN_PINGPONG_3REQUESTS_15STEPS',
   'tokens':requests,'strict_output_checks':75,'model_loads':0,'performance_point':False,
   'perstep_CPU_seed_writes':0,'requests':3,'changed_seed':True,'native':native}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='native'}),flush=True)
