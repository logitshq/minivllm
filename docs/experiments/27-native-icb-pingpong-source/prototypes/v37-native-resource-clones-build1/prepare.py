from pathlib import Path
import hashlib,shutil
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).parent;parent=P/'v34-native-replay-build1/replay.mm'
s=parent.read_text();assert hashlib.sha256(parent.read_bytes()).hexdigest()=='da7df9ceee5d042bbac73489bfa309cbd4188cb4ba1be8f4628ffb4f21438a93'
def change(a,b):
 global s
 assert s.count(a)==1,(a[:90],s.count(a));s=s.replace(a,b)
change('id<MTLBuffer> b=nil; NSData *initial=nil; size_t arena=0;', 'id<MTLBuffer> b=nil,owned=nil; NSData *initial=nil;')
change('id<MTLBuffer> arena=nil;', '')
change('size_t snapshotBytes=0,zeroBytes=0,arenaBytes=0;', 'size_t snapshotBytes=0,zeroBytes=0,cloneBytes=0;unsigned cloneCount=0,lastScopeEmitted=0,lastScopeCoalesced=0,lastResourceBarriers=0;')
change('bool built=false;NSUInteger arenaOptions=0;bool haveArenaOptions=false;', 'bool built=false;')
change('return x.zero?x.b:S().arena;', 'return x.zero?x.b:x.owned;')
change('return x.zero?o:x.arena+o;', 'return o;')
a=next(x for x in s.splitlines() if x.startswith('static void encodeGraph('))
b='''// Coalesce only adjacent full-buffer barriers. No dispatch/resource/other-scope command is crossed.
static void encodeGraph(id<MTLComputeCommandEncoder>e){
 bool priorFull=false;S().lastScopeEmitted=S().lastScopeCoalesced=S().lastResourceBarriers=0;
 for(auto&cmd:S().commands){
  bool scopeCommand=cmd.kind==3||(cmd.kind>=5&&cmd.kind<=8);
  MTLBarrierScope scope=cmd.kind==3?cmd.scope:MTLBarrierScopeBuffers;
  bool full=scopeCommand&&scope==MTLBarrierScopeBuffers;
  if(full){if(priorFull){++S().lastScopeCoalesced;continue;}[e memoryBarrierWithScope:scope];++S().lastScopeEmitted;priorFull=true;continue;}
  priorFull=false;
  if(scopeCommand){[e memoryBarrierWithScope:scope];++S().lastScopeEmitted;continue;}
  if(cmd.kind==4){std::vector<id<MTLResource>>r;for(auto x:cmd.resources)r.push_back(mapped(x));if(!r.empty()){[e memoryBarrierWithResources:r.data() count:r.size()];++S().lastResourceBarriers;}continue;}
  [e setComputePipelineState:cmd.p];
  for(auto&arg:cmd.bindings){auto&x=arg.second;if(x.bytes)[e setBytes:x.bytes.bytes length:x.bytes.length atIndex:arg.first];else[e setBuffer:mapped(x.b) offset:mappedOffset(x.b,x.offset) atIndex:arg.first];}
  for(auto&x:cmd.tg)[e setThreadgroupMemoryLength:x.second atIndex:x.first];
  if(cmd.kind==1)[e dispatchThreads:cmd.grid threadsPerThreadgroup:cmd.group];else[e dispatchThreadgroups:cmd.grid threadsPerThreadgroup:cmd.group];
 }
}'''
change(a,b)
change('memcpy((char*)S().arena.contents+x.arena,x.initial.bytes,x.b.length);', 'memcpy(x.owned.contents,x.initial.bytes,x.b.length);') if s.count('memcpy((char*)S().arena.contents+x.arena,x.initial.bytes,x.b.length);')==1 else None
# Same reset copy also appears in the legacy v33 API; both route to their own clone.
s=s.replace('memcpy((char*)S().arena.contents+x.arena,x.initial.bytes,x.b.length);','memcpy(x.owned.contents,x.initial.bytes,x.b.length);')
a='else{NSUInteger options=x.b.resourceOptions;if(!S().haveArenaOptions){S().arenaOptions=options;S().haveArenaOptions=true;}else if(options!=S().arenaOptions){fail("mixed mutable resource options unsupported");return 1;}x.arena=(S().arenaBytes+255)&~(size_t)255;S().arenaBytes=x.arena+x.b.length;}'
b='''else{
 if(x.b.length>S().device.maxBufferLength||x.initial.length!=x.b.length){fail("invalid per-resource clone size/snapshot");return 1;}
 x.owned=[S().device newBufferWithLength:x.b.length options:x.b.resourceOptions];
 if(!x.owned||!x.owned.contents||x.owned.length!=x.b.length||x.owned.resourceOptions!=x.b.resourceOptions||x.owned==x.b){fail("owned resource clone allocation/options mismatch");return 1;}
 S().cloneBytes+=x.owned.length;++S().cloneCount;
}'''
change(a,b)
change('if(S().arenaBytes>S().device.maxBufferLength){fail("arena exceeds actual maxBufferLength");return 1;}S().arena=[S().device newBufferWithLength:S().arenaBytes options:S().arenaOptions];if(!S().arena){fail("owned fixed arena allocation failed");return 1;}S().built=true;return 0;}}', '''std::set<uintptr_t>clones;for(auto&kv:S().buffers)if(kv.second.used&&!kv.second.zero){auto&x=kv.second;if(!clones.insert(key(x.owned)).second){fail("distinct original buffers share one clone");return 1;}}
 S().built=true;return 0;}}''')
change('memcpy(dst,(char*)S().arena.contents+b.arena+x.offset,n);','memcpy(dst,(char*)b.owned.contents+x.offset,n);')
change('@"arena_offset":@(b.arena),','@"owned_clone":@(key(b.owned)),@"clone_contents":@((uintptr_t)b.owned.contents),@"clone_length":@(b.owned.length),@"clone_resource_options":@(b.owned.resourceOptions),')
change('@"arena_bytes":@(S().arenaBytes),','@"arena_bytes":@0,@"mutable_clone_bytes":@(S().cloneBytes),@"mutable_clone_count":@(S().cloneCount),@"allocation_policy":@"one stable clone per original MTLBuffer, same offset and resourceOptions; readonly parameters zero-copy",@"last_full_scope_barriers_encoded":@(S().lastScopeEmitted),@"last_adjacent_full_scope_barriers_coalesced":@(S().lastScopeCoalesced),@"last_resource_barriers_encoded":@(S().lastResourceBarriers),')
change('static size_t arenaInput(const InputRange&x){return S().buffers.at(x.buffer).arena+x.offset;}','static size_t inputOffset(const InputRange&x){return x.offset;}\nstatic id<MTLBuffer>inputBuffer(const InputRange&x){return S().buffers.at(x.buffer).owned;}')
change('static size_t arenaOutput(const Output&x){return S().buffers.at(x.buffer).arena+x.offset;}','static size_t outputOffset(const Output&x){return x.offset;}\nstatic id<MTLBuffer>outputBuffer(const Output&x){return S().buffers.at(x.buffer).owned;}')
change('overlap(arenaInput(a),a.size,arenaInput(b),b.size)','a.buffer==b.buffer&&overlap(inputOffset(a),a.size,inputOffset(b),b.size)')
change('arenaOutput(x)%4','outputOffset(x)%4')
change('memcpy((char*)S().arena.contents+arenaInput(*x),src,n);','memcpy((char*)inputBuffer(*x).contents+inputOffset(*x),src,n);')
change('auto*p=(const char*)S().arena.contents+arenaInput(x);','auto*p=(const char*)inputBuffer(x).contents+inputOffset(x);')
change('[capture copyFromBuffer:S().arena sourceOffset:arenaOutput(x)', '[capture copyFromBuffer:outputBuffer(x) sourceOffset:outputOffset(x)')
change('toBuffer:S().arena destinationOffset:arenaInput(*i)', 'toBuffer:inputBuffer(*i) destinationOffset:inputOffset(*i)')
change('@"arena_offset":@(arenaInput(x)),','@"owned_clone":@(key(inputBuffer(x))),@"clone_contents":@((uintptr_t)inputBuffer(x).contents),@"clone_offset":@(inputOffset(x)),')
change('overlap(arenaInput(i),i.size,arenaOutput(o),o.size)','i.buffer==o.buffer&&overlap(inputOffset(i),i.size,outputOffset(o),o.size)')
# Add exact original->mapped resource identities to the diagnostic report.
change('NSDictionary*r=@{@"status":', '''NSMutableArray*resourceMaps=[NSMutableArray array];size_t commandIndex=0;for(auto&cmd:S().commands){if(cmd.kind==4){NSMutableArray*entries=[NSMutableArray array];for(auto b:cmd.resources){auto&bi=S().buffers.at(key(b));[entries addObject:@{@"original_buffer":@(key(b)),@"mapped_buffer":@(key(mapped(b))),@"zero_copy":@(bi.zero),@"length":@(b.length)}];}[resourceMaps addObject:@{@"command_index":@(commandIndex),@"resources":entries}];}++commandIndex;}NSDictionary*r=@{@"resource_barrier_mappings":resourceMaps,@"status":''')
assert 'S().arena' not in s and '.arena+' not in s and 'arenaInput' not in s and 'arenaOutput' not in s
(D/'replay.mm').write_text(s)
shutil.copyfile(P/'v34-native-replay-build1/ABI.h',D/'ABI.h')
print('source written',len(s),'bytes')
