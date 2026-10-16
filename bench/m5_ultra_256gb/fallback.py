"""Actual tensor fallback coverage. Spec flags are rejection tests, not drafter inference."""
import gc,json

def run(model,gates,e,w,mx,BatchState,ForwardContext,make_cache,DecodeBatch,toggle,reset_count,count,canonical,state_arrays,result):
 guards=[];result['fallbacks']=guards;layer=gates[0]
 def snapshot():return json.loads(json.dumps(count))
 def state(B):return BatchState(mx.zeros((B,3,10240),mx.bfloat16),mx.zeros((B,48,128,128),mx.float32))
 def one(name,B=1,context_flag=None,state_flag=None,direct=False):
  sigs=[];routes=[]
  for enabled in (False,True):
   toggle(enabled);reset_count();st=state(B);x=mx.zeros((B,5120),mx.bfloat16);ctx=ForwardContext('decode',[st])
   if context_flag:setattr(ctx,context_flag,True)
   if state_flag:setattr(st,state_flag,True)
   # Normal evidence is deliberately absent for the direct entry. For context/state
   # exclusions, pass normal=True to exercise the production rejection itself.
   output=layer._decode(x,st) if direct else layer(x,ctx,0,_norm_reuse_normal=True)
   sigs.append({'output':canonical(output),'conv':canonical(st.conv),'S':canonical(st.S)});routes.append(snapshot())
   if any(count[k] for k in ('norm_public','decode_norm','prepare','state','prework_dispatch','state_dispatch','kernel_creations')) or count['legacy']!={'0':1}:raise ValueError('fallback dispatch changed '+name)
   output=st=ctx=x=None;gc.collect();mx.clear_cache()
  if sigs[0]!=sigs[1]:raise ValueError('fallback strict signature differs '+name)
  guards.append({'name':name,'signatures':sigs,'routes':routes,'strict_equal':True})
 for B in (2,4):one('decode-B%d-fallback'%B,B=B)
 one('direct-decode-default-fallback',direct=True)
 for flag in ('record','capture','speculative','verify'):
  one('context-'+flag+'-fallback',context_flag=flag);one('state-'+flag+'-fallback',state_flag=flag)
 one('unknown-state-metadata-fallback',state_flag='unknown_metadata')
 # Unknown context metadata is filtered by Qwen, so this boundary must use the
 # real model rather than asking the layer to infer Qwen's context-key policy.
 for name,record,capture,decode,unknown in [('prefill',False,(),False,False),('recorded-prefill',True,(),False,False),('captured-prefill',False,(0,),False,False),('captured-decode',False,(0,),True,False),('unknown-context-model-decode',False,(),True,True),('subclass-context-model-decode',False,(),True,False)]:
  sigs=[];routes=[]
  for enabled in (False,True):
   toggle(enabled);cs=make_cache(model)
   if record:
    for c in cs:
     if hasattr(c,'record'):c.record=True
   reset_count()
   if decode:
    seed=model(mx.array(w['prompt_ids']),ForwardContext('prefill',cs),last_only=True);mx.eval(seed)
    batch=DecodeBatch(model);batch.rebuild([],[('guard',cs,64)]);ctx=ForwardContext('decode',batch.layers,positions=mx.array([19]),lens=mx.array([20]),max_len=20)
    if name=='subclass-context-model-decode':
     class UnknownContext(ForwardContext):pass
     ctx=UnknownContext('decode',batch.layers,positions=mx.array([19]),lens=mx.array([20]),max_len=20)
    if unknown:ctx.unknown_metadata=True
    reset_count();output=model(mx.array([w['continuation_ids'][0]]),ctx,capture=capture);mode='decode';length=20;caches=batch.layers
   else:
    output=model(mx.array(w['prompt_ids']),ForwardContext('prefill',cs),last_only=True,capture=capture);mode='prefill';length=19;caches=cs
   values=output if isinstance(output,tuple) else (output,);sig={'outputs':[canonical(v) for v in values],'states':{}}
   for label,arr in state_arrays(mode,caches,length):sig['states'][label]=canonical(arr)
   if len(sig['states'])!=128 or any(count[k] for k in ('norm_public','decode_norm','prepare','state','prework_dispatch','state_dispatch','kernel_creations')):raise ValueError('model fallback routed norm reuse '+name)
   if decode and count['legacy']!={str(i):1 for i in gates}:raise ValueError('model decode legacy count differs '+name)
   if not decode and count['legacy']:raise ValueError('prefill entered decode kernel '+name)
   if record and not all(c.tape is not None for c in cs if hasattr(c,'record')):raise ValueError('record tape missing')
   sigs.append(sig);routes.append(snapshot());output=values=cs=arr=caches=None
   if decode:batch.clear();batch=ctx=seed=None
   gc.collect();mx.clear_cache()
  if sigs[0]!=sigs[1]:raise ValueError('model fallback strict mismatch '+name)
  guards.append({'name':name,'signatures':sigs,'routes':routes,'strict_equal':True})
 result['fallback_scope']='18 actual exclusions: single-layer B2/B4/direct/flags/state unknown; full-model prefill/record/capture/unknown context. No speculative drafter, reload, batch scheduler or performance qualification.'
