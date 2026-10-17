"""V114 independent live ordinary-M1 reference. Hash comparisons only, never B input.

One own prefill/cache per request, then own greedy token feedback. Reference is
advanced on demand to B's computed frontier; no future arrays are retained.
No MLX/NumPy imports before explicit root execution.
"""
import hashlib
import struct


class OrdinaryM1Reference:
    def __init__(self, mx, np, model, prompt, capacity, signature):
        from tutorial_code.miniengine.cache import make_cache, DecodeBatch
        from tutorial_code.miniengine.context import ForwardContext
        self.mx,self.np,self.model,self.signature=mx,np,model,signature
        self.ForwardContext=ForwardContext
        self.capacity=capacity
        self.prompt=list(prompt)
        caches=make_cache(model)
        self.logits=model(mx.array(prompt,dtype=mx.int32),ForwardContext('prefill',caches),last_only=True)
        mx.eval(self.logits,*[a for c in caches for a in ((c.keys,c.values) if hasattr(c,'keys') else (c.conv_state,c.S))])
        self.token=int(mx.argmax(self.logits,axis=-1).item())
        self.tokens=[self.token]
        self.batch=DecodeBatch(model)
        self.batch.rebuild([], [('V114.independentM1',caches,capacity)])
        self.frontier=len(prompt)
        self.steps=0
        self.prefill_logits_signature=signature(mx,np,self.logits)

    def _flags_clear(self):
        return all(getattr(op,'_verify_m1_scope114',None) is None for name,op in self.model.named_modules())

    def advance_to(self, frontier):
        if type(frontier) is not int or not self.frontier <= frontier <= self.capacity:
            raise ValueError('reference must advance to a bounded actual computed frontier')
        assert self._flags_clear(),'ordinary reference entered a target broadcast verify scope'
        mx=self.mx
        while self.frontier<frontier:
            pos=self.frontier
            self.logits=self.model(mx.array([self.token],dtype=mx.int32),self.ForwardContext('decode',self.batch.layers,
                                   positions=mx.array([pos],dtype=mx.int32),lens=mx.array([pos+1],dtype=mx.int32),max_len=pos+1))
            mx.eval(self.logits,*self.batch.arrays())
            self.token=int(mx.argmax(self.logits,axis=-1).item())
            self.tokens.append(self.token)
            self.frontier+=1;self.steps+=1;self.batch.lens[0]=self.frontier
        assert self._flags_clear()

    def compare(self, a, b):
        mx,np=self.mx,self.np
        A=self.signature(mx,np,a);B=self.signature(mx,np,b)
        row={'A':A,'B':B,'strict_equal':A==B,'first_difference':None}
        if A==B:return row
        if a.shape!=b.shape or a.dtype!=b.dtype:
            row['first_difference']={'kind':'shape_or_dtype'};return row
        unsigned=mx.uint16 if a.dtype in (mx.bfloat16,mx.float16) else mx.uint32
        aa=np.ascontiguousarray(np.asarray(a.view(unsigned)));bb=np.ascontiguousarray(np.asarray(b.view(unsigned)))
        indices=np.flatnonzero(aa.reshape(-1)!=bb.reshape(-1))
        if indices.size:
            flat=int(indices[0]);index=tuple(int(v) for v in np.unravel_index(flat,aa.shape))
            left,right=int(aa[index]),int(bb[index])
            def number(bits):
                if a.dtype==mx.bfloat16:return struct.unpack('<f',struct.pack('<I',bits<<16))[0]
                if a.dtype==mx.float32:return struct.unpack('<f',struct.pack('<I',bits))[0]
                return struct.unpack('<e',struct.pack('<H',bits))[0]
            row['first_difference']={'index':list(index),'A_bits':left,'B_bits':right,
                                     'A_value':number(left),'B_value':number(right),'different_elements':int(indices.size)}
        return row

    def state_compare(self, transaction, keep):
        frontier=transaction.traces[0].start_offset+keep
        assert self.frontier==frontier
        states=[];tail=[]
        for layer,(trace,reference) in enumerate(zip(transaction.traces,self.batch.layers)):
            B,_=trace.witness(keep) if keep else (trace.first,trace.start_offset)
            A=(reference.K,reference.V) if trace.kind=='attention' else (reference.conv[0],reference.S[0])
            for component,(a,b) in enumerate(zip(A,B)):
                label=f'state.{2*layer+component}'
                if trace.kind=='attention':
                    assert a.shape==b.shape==(1,4,self.capacity,256)
                    used_a,used_b=a[:,:,:frontier,:],b[:,:,:frontier,:]
                    row=self.compare(used_a,used_b)
                    bits_a=self.np.asarray(a.view(self.mx.uint16))[:,:,frontier:,:]
                    bits_b=self.np.asarray(b.view(self.mx.uint16))[:,:,frontier:,:]
                    tail.append({'label':label,'A_tail_zero':bool(self.np.all(bits_a==0)),
                                 'B_tail_zero':bool(self.np.all(bits_b==0)),
                                 'physical_shape':list(a.shape),'used_prefix':frontier})
                else:row=self.compare(a,b)
                states.append({'label':label,**row})
        assert len(states)==128 and len(tail)==32
        return {'computed_frontier':frontier,'cache_components':states,'attention_tail_checks':tail,
                'all128_strict_equal':all(s['strict_equal'] for s in states),
                'all32_attention_tails_zero':all(t['A_tail_zero'] and t['B_tail_zero'] for t in tail)}

    def compare_begin(self, transaction):
        if self.frontier!=transaction.traces[0].start_offset:
            raise RuntimeError('independent A and next B invocation begin at different frontiers')
        return {'phase':'before_verify_initial_cache',**self.state_compare(transaction,0)}

    def compare_commit(self, logits, transaction, info):
        frontier=info['start']+info['committed_inputs']
        self.advance_to(frontier)
        keep=info['committed_inputs']
        state=self.state_compare(transaction,keep)
        score=self.compare(self.logits,logits[keep-1:keep])
        return {'phase':'selected_commit_before_apply','request':info['request'],'transaction':info['transaction'],
                'keep':keep,'A_ordinary_decode_steps':self.steps,**state,
                'committed_last_logits':score,
                'hard_gate_pass':state['all128_strict_equal'] and state['all32_attention_tails_zero'] and score['strict_equal'],
                'B_received_A_IDs_or_hashes_as_input':False}
