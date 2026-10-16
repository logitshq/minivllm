def state_arrays(mode,caches,length):
 """Yield one valid array view at a time; never retain an all-layer view list."""
 for i,c in enumerate(caches):
  if mode=='prefill':
   assert c.offset==length,('cache offset',i,c.offset,length)
   if hasattr(c,'keys'):
    yield f'L{i}.K',c.keys[:,:,:length]
    yield f'L{i}.V',c.values[:,:,:length]
   else:
    yield f'L{i}.conv',c.conv_state
    yield f'L{i}.S',c.S
  else:
   if hasattr(c,'K'):
    yield f'L{i}.K',c.K[:,:,:length]
    yield f'L{i}.V',c.V[:,:,:length]
   else:
    yield f'L{i}.conv',c.conv
    yield f'L{i}.S',c.S
