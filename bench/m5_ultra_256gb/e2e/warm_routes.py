"""Warm-only actual delegates; preserves warmed raw cached callable identities."""
def expected_counts(enabled):
 return {'decode':4464,'decode_norm_reuse':4464 if enabled else 0,'decode_gates':4464,'compiled_gates':4464,**{k:4464 if enabled else 0 for k in ('norm_public','prepare','state','prework_dispatch','state_dispatch')}}
def validate_metadata(row,enabled):
 assert row['hooks_restored_before_measure'] is True
 assert row['counts']==expected_counts(enabled),'actual warm delegate count differs'
 assert row['kernel_creations']==({'prework':1,'state':1} if enabled else {}) and row['compile_creations']==48
 assert row['kernel_cached_callable_count']==(2 if enabled else 0) and row['probe_absent'] is True
 assert row['warmed_raw_cached_callable_identity_preserved'] is True and row['cached_wrappers_remaining']==0
class WarmRoutes:
 def __init__(self,gdn,norm,mx,gdns,BatchState,enabled):
  self.gdn,self.norm,self.mx,self.gdns,self.enabled=gdn,norm,mx,gdns,enabled;self.restores=[];self.raw={};self.counts={k:0 for k in expected_counts(enabled)};self.creations={};self.compiles=0;self.done=False;self.shapes={}
  self.start_cache={k:getattr(norm,k) for k in ('_prework_kernel','_state_kernel','_norm_probe_kernel')}
  assert all(v is None for v in self.start_cache.values()),'fresh child N2 cache must be empty'
  def replace(obj,name,fn):
   old=getattr(obj,name);self.restores.append((obj,name,old));setattr(obj,name,fn)
  def hit(kind):self.counts[kind]+=1
  cls=gdn.GatedDeltaNet;decode0=cls._decode
  def decode(layer,x,st,**kw):
   assert x.shape==(1,5120) and x.dtype==mx.bfloat16 and type(st) is BatchState and set(vars(st))=={'conv','S'} and kw.get('_compile_gates') is True and kw.get('_norm_reuse_normal') is True
   assert st.conv.shape==(1,3,10240) and st.conv.dtype==mx.bfloat16 and st.S.shape==(1,48,128,128) and st.S.dtype==mx.float32
   hit('decode');self.shapes['normal_decode']={'x':[1,5120],'conv':[1,3,10240],'S':[1,48,128,128],'normal':True,'compile_gates':True}
   return decode0(layer,x,st,**kw)
  replace(cls,'_decode',decode)
  for name,kind in (('_decode_norm_reuse','decode_norm_reuse'),('_decode_gates','decode_gates')):
   original=getattr(cls,name)
   def called(*a,_fn=original,_kind=kind,**kw):hit(_kind);return _fn(*a,**kw)
   replace(cls,name,called)
  original_public=norm.gated_delta_step_norm_reuse_metal
  def public(u,g,beta,S):
   assert u.shape==(1,10240) and u.dtype==mx.bfloat16 and g.shape==beta.shape==(1,48) and g.dtype==beta.dtype==S.dtype==mx.float32 and S.shape==(1,48,128,128)
   hit('norm_public');self.shapes['norm_public']={'u':[1,10240],'u_dtype':'bfloat16','g_beta':[1,48],'S':[1,48,128,128],'outputs_dtype':'float32'}
   return original_public(u,g,beta,S)
  replace(norm,'gated_delta_step_norm_reuse_metal',public)
  for name,kind in (('_prepare','prepare'),('_state','state')):
   original=getattr(norm,name)
   def called(*a,_fn=original,_kind=kind,**kw):hit(_kind);return _fn(*a,**kw)
   replace(norm,name,called)
  factory0=norm._make_kernel
  def factory(mx,name,*a,**kw):
   kind={'gdn_norm_reuse_prework_build1':'prework','gdn_norm_reuse_state_build1':'state'}.get(name)
   assert kind is not None,'unexpected N2 factory'
   raw=factory0(mx,name,*a,**kw);assert callable(raw);self.creations[kind]=self.creations.get(kind,0)+1
   def dispatched(*a,**kw):hit(kind+'_dispatch');return raw(*a,**kw)
   self.raw[dispatched]=raw;return dispatched
  replace(norm,'_make_kernel',factory)
  compile0=mx.compile
  def compile(fn,*a,**kw):
   raw=compile0(fn,*a,**kw)
   if fn is not gdn._gates_explicit:return raw
   self.compiles+=1
   def invoked(*a,**kw):hit('compiled_gates');return raw(*a,**kw)
   self.raw[invoked]=raw;return invoked
  replace(mx,'compile',compile)
 def restore(self,validate=True):
  if self.done:
   if validate:validate_metadata(self.metadata,self.enabled)
   return self.metadata
  # Capture warmed raw objects, unwrap the caches, and keep them for measurement.
  cached=[(self.norm,k) for k in ('_prework_kernel','_state_kernel')]+[(g,'_compiled_decode_gates') for g in self.gdns]
  identities=[]
  for obj,key in cached:
   value=getattr(obj,key);raw=self.raw.get(value,value)
   if value is not raw:setattr(obj,key,raw)
   assert getattr(obj,key) is raw
   identities.append({'cache':key,'raw_callable_id':id(raw) if raw is not None else None,'restored_callable_id':id(getattr(obj,key)) if raw is not None else None,'same_raw_object':True})
  for obj,key,fn in reversed(self.restores):setattr(obj,key,fn)
  assert all(getattr(obj,key) is fn for obj,key,fn in self.restores)
  assert not any(getattr(obj,key) in self.raw for obj,key in cached)
  self.done=True
  self.metadata={'scope':'three warm requests only; actual delegates restored before measurement','counts':dict(self.counts),'kernel_creations':dict(self.creations),'compile_creations':self.compiles,'actual_shapes':self.shapes,'hooks_restored_before_measure':True,'kernel_cached_callable_count':sum(callable(getattr(self.norm,k)) for k in ('_prework_kernel','_state_kernel')),'probe_absent':self.norm._norm_probe_kernel is None,'warmed_raw_cached_callable_identity_preserved':True,'cached_wrappers_remaining':0,'raw_cached_callable_identities':identities}
  if validate:validate_metadata(self.metadata,self.enabled)
  return self.metadata
