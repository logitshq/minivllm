"""V117 numerical qualification support. Import is stdlib-only."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import sys
D=Path(__file__).resolve().parent
P=D.parent
ROOT=P/'candidate-v116-bounded-draftloader-build1-root'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def identity():
    pins=json.loads((D/'source-pins.json').read_text())
    for path,h in pins.items():assert sha(path)==h,('pinned source changed',path)
    for row in json.loads((D/'header-pins.json').read_text()):
        path=Path(row['path']);assert path.stat().st_size==row['file_bytes']
        with path.open('rb') as f:
            prefix=f.read(8);n=struct.unpack('<Q',prefix)[0];assert 0<n<32*1024*1024
            body=f.read(n)
        assert hashlib.sha256(prefix+body).hexdigest()==row['prefix_header_SHA256']
    return pins

def runtime(mx):
    versions={name:importlib.metadata.version(name) for name in ('mlx','mlx-metal')}
    assert versions=={'mlx':'0.32.3','mlx-metal':'0.32.3'}
    controls={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
    assert all(os.environ.get(k)==v for k,v in controls.items())
    assert not os.environ.get('DYLD_INSERT_LIBRARIES'), 'V114 does not use native interposition'
    info=mx.device_info();assert info['device_name']=='Apple M4 Max' and info['architecture']=='applegpu_g16s'
    return {'versions':versions,'controls':controls,'device':info}

def provenance(pins):
    observed={}
    for name,module in tuple(sys.modules.items()):
        if name.startswith('tutorial_code.miniengine') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve())
            assert Path(path).is_relative_to(ROOT) and path in pins and sha(path)==pins[path],('wrong candidate import',name,path)
            observed[name]=path
    return observed

def signature(mx,np,value):
    if value is None:return None
    mx.eval(value)
    if value.dtype==mx.bfloat16:
        host=np.ascontiguousarray(np.asarray(value.view(mx.uint16)),dtype='<u2')
        assert not np.any((host&0x7f80)==0x7f80)
    else:
        host=np.ascontiguousarray(np.asarray(value))
        if value.dtype in (mx.float32,mx.float16):assert np.isfinite(host).all()
    return {'shape':list(value.shape),'dtype':str(value.dtype),'bytes':host.nbytes,'sha256':hashlib.sha256(memoryview(host).cast('B')).hexdigest()}

def cache_signatures(mx,np,transaction,keep=None):
    result=[]
    for trace in transaction.traces:
        values,offset=trace.witness(keep) if keep is not None else ((trace.cache.keys,trace.cache.values),trace.cache.offset) if trace.kind=='attention' else ((trace.cache.conv_state,trace.cache.S),trace.cache.offset)
        result.append({'kind':trace.kind,'offset':offset,'values':[signature(mx,np,v) for v in values]})
    return result

class Observer:
    def __init__(self,mx,np,rows,save):self.mx,self.np,self.rows,self.save,self.pending=mx,np,rows,save,{}
    def __call__(self,phase,transaction,info):
        mx,np=self.mx,self.np
        if phase=='begin':
            self.pending[transaction.id]=[{'kind':trace.kind,'offset':trace.start_offset,'values':[signature(mx,np,a) for a in trace.first]} for trace in transaction.traces]
        elif phase=='before_commit':
            initial=self.pending.pop(transaction.id)
            expected=cache_signatures(mx,np,transaction,info['committed_inputs'])
            histories=[[{'offset':trace.start_offset+i,'values':[signature(mx,np,a) for a in values]} for i,values in enumerate(trace.history)] for trace in transaction.traces]
            self.pending[transaction.id]=(expected,histories,initial)
        else:
            expected,before,initial=self.pending.pop(transaction.id)
            actual=cache_signatures(mx,np,transaction)
            after=[[{'offset':trace.start_offset+i,'values':[signature(mx,np,a) for a in values]} for i,values in enumerate(trace.history)] for trace in transaction.traces]
            row={**info,'same_invocation_prefix_strict':actual==expected,'literal_history_readonly':before==after,'expected':expected,'actual':actual,'prefix_history':before,
                 'invocation_inputs_readonly':initial==[{'kind':trace.kind,**history[0]} for trace,history in zip(transaction.traces,after)],
                 'producer_owners_retained':sum(len(trace.producer_owners) for trace in transaction.traces)}
            self.rows.append(row);self.save()
            assert actual==expected and before==after and row['invocation_inputs_readonly'],'same-invocation prefix/history bytes differ'
            assert all(getattr(trace.cache,'_verify108',None) is None for trace in transaction.traces)

def memory(mx):
    return {name:getattr(mx,name)() for name in ('get_active_memory','get_cache_memory','get_peak_memory')}
