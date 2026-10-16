"""Standalone pure131→132 factory. Backend/types injected; no MLX import.

Only fixed-cap B1 original candidate44 is intended. Weights/configuration are
held fixed by the owner. No caller seed/context/token/cache arrays are closed over.
"""
def make_pure(model, *, backend, kv_type, state_type, context_type, max_len,
              on_trace=None):
    kinds=tuple(layer.is_attn for layer in model.layers)
    if len(kinds)!=64 or max_len!=256:
        raise ValueError('only the qualified64layer fixed-cap256 graph is intended')
    observer=[on_trace]
    def pure(tokens, positions, lens, *flat):
        if observer[0] is not None:observer[0]()
        if len(flat)!=128:raise ValueError('exact128 dynamic state arrays required')
        caches=[kv_type(flat[2*i][:],flat[2*i+1][:]) if attn
                else state_type(flat[2*i],flat[2*i+1])
                for i,attn in enumerate(kinds)]
        ctx=context_type('decode',caches,positions=positions,lens=lens,max_len=max_len)
        logits=model(tokens,ctx)
        states=tuple(v for attn,c in zip(kinds,caches)
                     for v in ((c.K,c.V) if attn else (c.conv,c.S)))
        return (logits,*states,backend.argmax(logits,axis=-1).astype(backend.int32),
                positions+1,lens+1)
    def detach_observer():observer[0]=None
    return pure,detach_observer
