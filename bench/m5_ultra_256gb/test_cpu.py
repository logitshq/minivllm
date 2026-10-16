"""Stdlib CPU contract tests. Never load MLX or execute a resource probe."""
import importlib.util
from pathlib import Path
import tempfile,json,types,contextlib,io,ast,copy
import unittest,os
D=Path(__file__).resolve().parent
P=Path(os.environ['MINIVLLM_EXPERIMENT_DOMAIN'])

def module(name):
 spec=importlib.util.spec_from_file_location(name,D/(name+'.py'))
 value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

class Contracts(unittest.TestCase):
 def test_default_and_help_and_bad_go_do_not_enter_runtime(self):
  d=module('driver')
  def forbidden(*a,**k):raise AssertionError('runtime reached')
  d.execute=forbidden
  for argv in ([],['--help'],['--go','/missing'],['--frozen-sha256','wrong']):
   with self.subTest(argv=argv):
    with contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
     try:d.main(argv)
     except SystemExit as e:self.assertEqual(e.code,0 if argv==['--help'] else 2)
  self.assertFalse((D/'attempt1.consumed.json').exists())
 def test_complete_wrong_go_remains_inert(self):
  d=module('driver')
  def forbidden(*a,**k):raise AssertionError('resource/runtime reached')
  d.Resources=d.execute=forbidden
  with tempfile.TemporaryDirectory() as t:
   go=Path(t)/'wrong.json';go.write_text('{"schema":"wrong"}');out=Path(t)/'must-not-exist'
   with contextlib.redirect_stderr(io.StringIO()):
    with self.assertRaises(SystemExit) as caught:d.main(['--go',str(go),'--frozen-sha256',d.digest(P/'frozen.json'),'--output-dir',str(out)])
   self.assertEqual(caught.exception.code,2);self.assertFalse(out.exists())
 def test_binary_difference_including_signed_zero(self):
  r=module('replay')
  with tempfile.TemporaryDirectory() as t:
   a=Path(t)/'a';b=Path(t)/'b';a.write_bytes(bytes(4));b.write_bytes(bytes([0,0,0,128]))
   with self.assertRaises(ValueError):r.same_bytes(a,b)
 def test_guard_derivation_does_not_mutate_raw(self):
  d=module('driver')
  raw={'known_inference_processes':[],'process_scope':'raw'}
  got=d.derived_snapshot(raw,100,True)
  self.assertEqual(raw,{'known_inference_processes':[],'process_scope':'raw'})
  self.assertEqual(got['known_inference_process_count'],1)
 def test_source_identity(self):
  d=module('driver');f=d.verify_frozen()
  self.assertEqual(len(f['candidate_sources']),26);self.assertEqual(len(f['baseline_sources']),25)
 def test_signature_contract(self):
  r=module('replay');sig={'shape':[1,2],'dtype':'mlx.core.float32','sha256':'0'*64,'canonical_bytes':8};r.signature(sig)
  for key,value in [('canonical_bytes',4),('sha256','G'*64),('dtype','float16')]:
   with self.subTest(key=key):
    bad=dict(sig);bad[key]=value
    with self.assertRaises(ValueError):r.signature(bad)
 def test_norm_hooks_delegate_actual_public_prepare_state_and_dispatch(self):
  w=module('worker');called=[];restores=[]
  norm=types.SimpleNamespace(_prework_kernel=None,_state_kernel=None,_norm_probe_kernel=None)
  def factory(mx,name,*a,**kw):
   called.append('factory-'+name)
   def kernel(*a,**kw):called.append('dispatch-'+name);return ('value',)
   return kernel
  norm._make_kernel=factory
  def prepare(*a,**kw):called.append('prepare');return norm._make_kernel(None,'gdn_norm_reuse_prework_build1')()
  def state(*a,**kw):called.append('state');return norm._make_kernel(None,'gdn_norm_reuse_state_build1')()
  norm._prepare=prepare;norm._state=state
  def public(*a,**kw):called.append('public');norm._prepare();return norm._state()
  norm.gated_delta_step_norm_reuse_metal=public
  class Layer:
   def _decode(self,fail=False):
    called.append('decode')
    if fail:raise RuntimeError('sentinel')
    return self._decode_norm_reuse()
   def _decode_norm_reuse(self):called.append('decode-norm');return norm.gated_delta_step_norm_reuse_metal()
  layer=Layer();gdn=types.SimpleNamespace(GatedDeltaNet=Layer,_gates_explicit=lambda:called.append('compiled-original'))
  legacy=types.SimpleNamespace(gated_delta_step_metal=lambda:called.append('legacy-original'))
  mx=types.SimpleNamespace(compile=lambda fn,*a,**kw:fn)
  originals={name:getattr(norm,name) for name in ('_make_kernel','_prepare','_state','gated_delta_step_norm_reuse_metal')}
  counts,reset=w.install_hooks(gdn,legacy,norm,mx,{0:layer},restores);reset()
  self.assertEqual(layer._decode(),('value',));legacy.gated_delta_step_metal();mx.compile(gdn._gates_explicit)()
  for key in ('norm_public','decode_norm','prepare','state','prework_dispatch','state_dispatch'):self.assertEqual(counts[key],{'0':1})
  self.assertEqual(counts['kernel_creations'],{'prework':1,'state':1});self.assertIn('legacy-original',called);self.assertIn('compiled-original',called)
  with self.assertRaisesRegex(RuntimeError,'sentinel'):layer._decode(True)
  legacy.gated_delta_step_metal();self.assertEqual(counts['legacy'],{'None':2})
  for obj,name,old in reversed(restores):setattr(obj,name,old)
  self.assertTrue(all(getattr(obj,name) is old for obj,name,old in restores))
  self.assertTrue(all(getattr(norm,key) is value for key,value in originals.items()))
 def test_raw_saved_before_guard_and_finally_error_scope(self):
  d=module('driver');fake=types.SimpleNamespace(snapshot=lambda *a:{'known_inference_processes':[]},deltas=lambda h:None,guard=lambda h:(_ for _ in ()).throw(ValueError('guard-stop')))
  with tempfile.TemporaryDirectory() as t:
   old=d.pinned;d.pinned=lambda *a:fake
   try:
    r=d.Resources(Path(t),{},{} )
    with self.assertRaisesRegex(ValueError,'guard-stop'):r.sample('fake')
    self.assertEqual(json.loads((Path(t)/'resource-00-raw.json').read_text()),{'known_inference_processes':[]})
   finally:d.pinned=old
 def test_go_all_fields_strict(self):
  d=module('driver');f=json.loads((P/'frozen.json').read_text());go=copy.deepcopy(f['GO_schema']);go.update(frozen_manifest_sha256='a'*64,expected_power='AC',authorizing_owner_pid=5,output_dir='/private/tmp/qual-cpu-inert')
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'GO.json';args=types.SimpleNamespace(go=path,frozen_sha256='a'*64,output_dir=Path(go['output_dir']))
   path.write_text(json.dumps(go));d.validate_go(args,f)
   for key in go:
    bad=copy.deepcopy(go);bad[key]=None;path.write_text(json.dumps(bad))
    with self.subTest(key=key):
     with self.assertRaises((ValueError,TypeError)):d.validate_go(args,f)
   extra=dict(go,extra_unknown=True);path.write_text(json.dumps(extra))
   with self.assertRaisesRegex(ValueError,'field set'):d.validate_go(args,f)
   relative=dict(go,output_dir='relative-output');path.write_text(json.dumps(relative));args.output_dir=Path('relative-output')
   with self.assertRaisesRegex(ValueError,'absolute'):d.validate_go(args,f)
 def test_canonical_loop_checks_entire_array(self):
  body=ast.parse((D/'worker.py').read_text());fn=next(n for n in ast.walk(body) if isinstance(n,ast.FunctionDef) and n.name=='canonical')
  text=ast.unparse(fn)
  self.assertIn('range(0, flat.size, 1 << 18)',text);self.assertIn('np.isfinite(flat[start:start + (1 << 18)]).all()',text)
 def test_nonfinite_last_chunk_is_rejected(self):
  import math
  body=ast.parse((D/'worker.py').read_text());fn=next(n for n in ast.walk(body) if isinstance(n,ast.FunctionDef) and n.name=='canonical')
  seen=[]
  class Value:
   size=(1<<18)+1
   def reshape(self,*a):return self
   def __getitem__(self,key):seen.append((key.start,key.stop));return [float('nan')] if key.start==1<<18 else [1.0]
  value=Value()
  class Finite:
   def __init__(self,part):self.part=part
   def all(self):return all(math.isfinite(x) for x in self.part)
  np=types.SimpleNamespace(ascontiguousarray=lambda x:x,asarray=lambda x,**kw:value,isfinite=Finite)
  mx=types.SimpleNamespace(float32='fp32');x=types.SimpleNamespace(astype=lambda dtype:None)
  ns={'np':np,'mx':mx};exec(compile(ast.Module(body=[fn],type_ignores=[]),'canonical-actual-AST','exec'),ns)
  with self.assertRaisesRegex(ValueError,'nonfinite'):ns['canonical'](x)
  self.assertEqual([a for a,b in seen],[0,1<<18])
 def test_replay_strict_synthetic_schema_and_tamper_rejection(self):
  # Tiny ephemeral fixture only tests the parser; it is never model qualification evidence.
  import hashlib,struct
  replay=module('replay');frozen=module('driver').digest(P/'frozen.json');layers=[str(i) for i in range(64) if i%4!=3];want={i:31 for i in layers}
  ids=json.loads((P/'chapter15-smoke-expected-v1.json').read_text())['expected_output_ids'];raw=struct.pack('<f',1.0)
  sig={'shape':[1],'dtype':'mlx.core.float32','sha256':hashlib.sha256(raw).hexdigest(),'canonical_bytes':4}
  def save(path,v):path.write_text(json.dumps(v))
  with tempfile.TemporaryDirectory(prefix='v19-replay-synthetic-') as t:
   out=Path(t);auth={'expected_power':'AC','fixture_only':True};go=json.dumps(auth).encode();go_sha=hashlib.sha256(go).hexdigest();baseline=None
   for role,names in [('accepted',['accepted']),('candidate',['candidate-off','candidate-on','candidate-reset-off'])]:
    root=out/role;root.mkdir();(root/'ROOT-GO.raw.json').write_bytes(go);rounds=[];raws=[]
    for name in names:
     (root/name).mkdir();on=name=='candidate-on';observations=[]
     for step in range(32):
      x={'mode':'prefill' if step==0 else 'decode','positions':[18+step],'greedy':[ids[step]],'logits':copy.deepcopy(sig),'states':{str(k):copy.deepcopy(sig) for k in range(128)}};observations.append(x)
      file=name+('/%02d-logits.float32le.bin'%step);(root/file).write_bytes(raw);raws.append({'file':file,'signature':copy.deepcopy(sig)})
     save(root/(name+'-observations.json'),observations)
     counts={k:(want if on else {}) for k in ('norm_public','decode_norm','prepare','state','prework_dispatch','state_dispatch')};counts.update(legacy={} if on else want,compiled_gates=want,compile_creations=48,kernel_creations={'prework':1,'state':1} if on else {})
     rounds.append({'name':name,'output_ids':ids,'canonical_array_hashes':4128,'fresh_cache_idle':True,'observations_file':name+'-observations.json','routes':counts,'committed_positions':[]})
    resources=[]
    for i in range(5 if role=='accepted' else 11):
     path=root/('resource-%02d.json'%i);save(path,{'power_metadata':{'status':'known','source':'AC Power'},'processinfo_thermal_metadata':{'status':'known','code':0},'pressure_level_value':1});resources.append({'file':path.name,'sha256':replay.digest(path)})
    r={'GO_sha256':go_sha,'GO':auth,'status':'PASS_WORKER_BOUNDED_SCOPE','frozen_sha256':frozen,'model_load_attempts':1,'default_off_checked':True,'hooks_restored':True,'cleanup_errors':[],'rounds':rounds,'parameters':{'initial':[None]*1349,'unchanged_at':[n+'-'+side for n in names for side in ('before','after')]+['final-reset-off']},'raw_resources':resources,'raw_logits':raws}
    if role=='candidate':
     r['fallbacks']=[{'name':str(i),'strict_equal':True,'signatures':[{},{}],'routes':[{k:{} for k in ('norm_public','decode_norm','prepare','state','prework_dispatch','state_dispatch','kernel_creations')}]*2} for i in range(18)]
     r['pipeline_lifecycle']=[{'round':n,'prework_present':n!='candidate-off','state_present':n!='candidate-off','probe_absent':True,'reset_preserves_factories':n=='candidate-reset-off'} for n in names]
    save(root/'receipt.json',r)
   got=replay.verify(out,include_owner=False);self.assertEqual((got['raw_logits_files'],got['state_signatures']),(128,16384))
   target=out/'candidate/candidate-on/31-logits.float32le.bin';target.write_bytes(struct.pack('<f',-0.0))
   with self.assertRaisesRegex(ValueError,'raw logits'):replay.verify(out,include_owner=False)
   target.write_bytes(raw);rp=out/'candidate/receipt.json';r=json.loads(rp.read_text());r['rounds'][1]['routes']['state_dispatch']['0']=30;save(rp,r)
   with self.assertRaisesRegex(ValueError,'dispatch'):replay.verify(out,include_owner=False)
   r['rounds'][1]['routes']['state_dispatch']['0']=31;save(rp,r);op=out/'candidate/candidate-on-observations.json';obs=json.loads(op.read_text());obs[31]['states']['127']['sha256']='0'*64;save(op,obs)
   with self.assertRaisesRegex(ValueError,'signature/cache'):replay.verify(out,include_owner=False)
 def test_modules_are_stdlib_inert(self):
  import sys
  before=set(sys.modules)
  for name in ('driver','worker','fallback','replay'):module(name)
  self.assertFalse(any(k=='mlx' or k.startswith('mlx.') for k in set(sys.modules)-before))

if __name__=='__main__':unittest.main()
