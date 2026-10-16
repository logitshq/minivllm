"""Pure CPU source/metadata/mapping checks. No numerical surrogate tests."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('v66_helper_cpu',D/'four_projection.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
# Verify the exact native function and native qdot/helper text, not a rewritten equivalent.
up=Path('/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/fp_quantized.h');lines=up.read_text().splitlines(keepends=True);header=(D/'kernel_header.metal').read_text()
for start,end in ((20,102),(333,408)):
 text=''.join(lines[start-1:end-1]);assert text in header,(start,end)
assert (D/'LICENSE-MLX.txt').read_bytes()==(P/'v22-nvfp4-qmv-r2-build1/LICENSE-MLX.txt').read_bytes()
assert header==(P/'v22-nvfp4-qmv-r2-build1/kernel_header.metal').read_text().split('constant int v22_in_vec_size')[0]
# Exhaustively check row ownership/tile remap, never compute a fake dot product.
bounds=(0,1280,2048,2054,2060);owners=[]
for section,n in enumerate(h.ROWS):
 rows=[(tile-bounds[section])*8+sg*4+r for tile in range(bounds[section],bounds[section+1]) for sg in range(2) for r in range(4)]
 assert rows==list(range(n));owners.extend((section,row) for row in rows)
assert len(owners)==len(set(owners))==16480 and sum(h.ROWS)//8==2060
assert len(h.INPUT_NAMES)==9 and len(h.OUTPUT_NAMES)==4 and h.MAIN_BUFFER_ARGUMENTS==13<=31
body=(D/'kernel_body.metal').read_text();assert body.count('fp_qmv_fast_impl<T, 16, 4, false, 4>')==4
assert not any(x in (D/'four_projection.py').read_text() for x in ('mx.concatenate','mx.array(','mx.contiguous','quantized_matmul','compile_options='))
# Metadata guard only: no shader factory, arrays, model, native library or output simulation.
class A:
 def __init__(self,shape,dtype):self.shape,self.dtype=shape,dtype
class Backend:
 gpu='gpu';bfloat16='BF16';uint32='U32';uint8='U8'
 def default_device(self):return self.gpu
mx=Backend();ops=tuple(v for n in h.ROWS for v in (A((n,640),'U32'),A((n,320),'U8')));strides=((640,1),(320,1))*4
kw=dict(observed_strides=strides,runtime_version='0.32.3',device_name='Apple M4 Max',architecture='applegpu_g16s')
f=h.prepare(mx,ops,**kw);assert all(a is b for a,b in zip(f.operands,ops)) and f.kernel is None
failures=0
for bad in ({'runtime_version':'0.32.2'},{'observed_strides':((1,640),)*8},{'device_name':'other'}):
 try:h.prepare(mx,ops,**(kw|bad))
 except ValueError:failures+=1
 else:raise AssertionError('metadata guard accepted unsupported input')
try:f.project(A((2,5120),'BF16'),x_strides=(5120,1))
except ValueError:failures+=1
else:raise AssertionError('B2 accepted')
for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
tree=ast.parse((D/'numeric.py').read_text());assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name.startswith(('mlx','numpy')) for a in getattr(n,'names',[])) for n in tree.body)
assert not any(n=='numpy' or n=='mlx' or n.startswith('mlx.') for n in sys.modules)
print(json.dumps({'status':'PASS_CPU_SOURCE_METADATA_ONLY','verbatim_native_math':True,'R4':4,'SG':2,'threads':64,'unique_output_rows':len(owners),'original_handle_metadata_check':True,'unsupported_metadata_rejections':failures,'main_buffer_arguments':13,'MLX_imported':False,'GPU_used':False}))
