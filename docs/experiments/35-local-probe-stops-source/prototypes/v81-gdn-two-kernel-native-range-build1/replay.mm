// Frozen capture/arena donor is included verbatim; donor exports are unchanged.
#include "/private/tmp/minivllm-engine-optimization-20261014/v40-native-icb-build2/replay.mm"
#include "range_contract.h"
#include <algorithm>
#include <cmath>
struct V81Slot {uint64_t address[10]{},size[10]{};uintptr_t allocation[10]{};bool registered=false;};
struct V81State {
 bool enabled=false,restored=false,valid=false,completed=false;unsigned lastArm=0,recreated=0;
 uint64_t lastGroups=0,lastEncoded=0,lastWall=0,loops=0,totalGroups=0,totalEncoded=0;
 double gpuStart=0,gpuEnd=0;size_t constantBytes=0;NSUInteger maxBind=0;
 V81Slot slots[30];id<MTLIndirectCommandBuffer>icb=nil;id<MTLBuffer>constants=nil;
 std::vector<Command*> dispatches;std::vector<unsigned> capturedBefore;
 std::map<uintptr_t,size_t> constantOffsets;std::map<uintptr_t,id<MTLComputePipelineState>>psos;
 std::vector<id<MTLResource>>readonlyResources;std::set<uintptr_t>immutableAllocations,shapeAllocations;
 std::string report;
};
static V81State&V81(){static V81State x;return x;}
static int v81_stop(const std::string&reason){V81().valid=false;fail("V81: "+reason);return 1;}
static const uint64_t v81_sizes[10]={20480,192,192,3145728,8192,8192,24576,128,24576,3145728};
extern "C" int v81_register_bank(uint32_t arm,uint32_t bank,const uint64_t*addresses,const uint64_t*sizes){
 std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V81();
 if(!addresses||!sizes||arm>1||bank>=15||!S().built||S().armed||v.enabled||!S().error.empty())return v81_stop("invalid/late bank registration");
 auto&slot=v.slots[arm*15+bank];if(slot.registered)return v81_stop("duplicate bank");
 for(unsigned a=0;a<10;++a){if(!addresses[a]||sizes[a]!=v81_sizes[a]||addresses[a]+sizes[a]<addresses[a])return v81_stop("fixed array byte contract mismatch");slot.address[a]=addresses[a];slot.size[a]=sizes[a];}
 slot.registered=true;return 0;
}
static bool v81_resolve(V81Slot&slot,unsigned a,const Binding&binding,bool immutable){
 if(binding.bytes||!binding.b||!binding.b.contents)return false;auto id=key(binding.b);auto&bi=S().buffers.at(id);
 uintptr_t base=(uintptr_t)binding.b.contents,addr=base+binding.offset;
 if(binding.offset||addr!=base||addr!=slot.address[a]||slot.size[a]>binding.b.length||bi.zero!=immutable)return false;
 if(slot.allocation[a]&&slot.allocation[a]!=id)return false;slot.allocation[a]=id;
 if(immutable){auto*p=parameter(addr);return bi.pinCandidate&&p&&p->addr==addr&&p->size==slot.size[a]&&bi.writeStarts.empty();}
 if(bi.pinCandidate||!bi.initial||bi.initial.length!=binding.b.length)return false;
 return std::count_if(S().outputs.begin(),S().outputs.end(),[&](const Output&o){return o.addr==addr&&o.size==slot.size[a]&&o.buffer==id&&o.offset==0;})==1;
}
static bool v81_disjoint_allocations(){
 std::vector<std::pair<uintptr_t,uintptr_t>>original,arena;
 for(auto&pair:S().buffers){auto&b=pair.second;if(!b.used)continue;uintptr_t start=(uintptr_t)b.b.contents;size_t n=b.b.length;
  if(!start||!n||n>UINTPTR_MAX-start)return false;original.push_back({start,start+n});
  if(!b.zero){if(!b.initial||b.initial.length!=n||b.arena>S().arenaBytes||n>S().arenaBytes-b.arena)return false;arena.push_back({b.arena,b.arena+n});}
 }
 auto clean=[](std::vector<std::pair<uintptr_t,uintptr_t>>&ranges){std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();++i)if(ranges[i].first<ranges[i-1].second)return false;return true;};
 return clean(original)&&clean(arena);
}
extern "C" int v81_build_ranges(){@autoreleasepool{
 std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V81();
 if(!S().built||S().armed||!S().error.empty()||F().sealed||I().enabled||!PP().pairs.empty()||v.enabled||S().replays)return v81_stop("requires fresh captured graph without legacy replay/feedback");
 if(S().pins.size()!=60||S().dispatches!=60||S().outputs.size()!=180)return v81_stop("requires60 immutable pins/60 dispatches/180 registered mutable outputs");
 if(!v81_disjoint_allocations())return v81_stop("original or arena allocations overlap or lack exact snapshots");
 unsigned before=0;for(auto&c:S().commands){if(c.kind>=3&&c.kind<=8){++before;continue;}if(c.kind!=1&&c.kind!=2)return v81_stop("unknown captured command kind");v.dispatches.push_back(&c);v.capturedBefore.push_back(before);before=0;}
 if(v.dispatches.size()!=60)return v81_stop("dispatch extraction mismatch");
 std::vector<v81::Access>accesses;std::vector<v81::Bank>banks;NSUInteger maxTG=0;bool threads=false,groups=false;
 for(unsigned ordinal=0;ordinal<60;++ordinal){unsigned arm=ordinal/30,stage=ordinal%2,slotIndex=ordinal/2;auto&slot=v.slots[slotIndex];auto&c=*v.dispatches[ordinal];
  if(!slot.registered)return v81_stop("unregistered arm-bank slot");auto p=S().psos.find(key(c.p));
  if(p==S().psos.end()||!p->second.reflection||!p->second.function||!p->second.descriptor)return v81_stop("actual reflection/function/descriptor unavailable");auto&info=p->second;
  if(info.name!=v81::function(arm,stage))return v81_stop("function outside exact four-PSO whitelist: "+info.name);
  if(!v81::geometry(stage,c.kind,{c.grid.width,c.grid.height,c.grid.depth},{c.group.width,c.group.height,c.group.depth}))return v81_stop("actual geometry differs from frozen helper");
  if(stage==1&&!v.capturedBefore[ordinal])return v81_stop("no captured prework-to-state synchronization boundary");
  for(auto&b:c.bindings)v.maxBind=std::max(v.maxBind,b.first+1);
  v81::Access use;std::set<unsigned>seenArrays;bool shapeSeen=false;
  for(auto&a:info.args){if(a.type==MTLArgumentTypeThreadgroupMemory){if(!c.tg.count(a.index))return v81_stop("missing dynamic TG binding");continue;}
   if(a.type!=MTLArgumentTypeBuffer)return v81_stop("unknown active reflection type");auto binding=c.bindings.find(a.index);if(binding==c.bindings.end())return v81_stop("missing active binding");auto&b=binding->second;
   int array=v81::arrayIndex(stage,a.index,a.name);if(array==-1)return v81_stop("argument outside exact index/name whitelist");
   if(array==-2){if(a.access!=MTLArgumentAccessReadOnly)return v81_stop("S_shape not readonly");shapeSeen=true;
    if(b.bytes){if(!v.constantOffsets.count(key(b.bytes))){size_t off=(v.constantBytes+255)&~size_t(255);v.constantOffsets[key(b.bytes)]=off;v.constantBytes=off+b.bytes.length;}}
    else{if(!b.b||!b.b.contents||b.offset||b.b.length>256)return v81_stop("unexpected S_shape buffer extent");auto id=key(b.b);auto&bi=S().buffers.at(id);if(bi.zero||bi.pinCandidate||!bi.initial||!bi.writeStarts.empty())return v81_stop("S_shape buffer aliases mutable or pinned allocation");v.shapeAllocations.insert(id);}
    continue;
   }
   if(!seenArrays.insert(array).second)return v81_stop("duplicate active array binding");
   bool out=v81::output(stage,a.index,a.name);
   if(out){if(a.access!=MTLArgumentAccessWriteOnly&&!v81::storeOnly(arm,stage,info.name,a.index,a.name,(unsigned)a.access))return v81_stop("output requires WriteOnly or exact source-store-only RW exception");}
   else if(a.access!=MTLArgumentAccessReadOnly)return v81_stop("input/S requires actual ReadOnly reflection; no input RW exception");
   if(!v81_resolve(slot,array,b,array<4))return v81_stop("exact array/range/pin/output resolution failed for "+a.name);
   auto id=key(b.b);if(out)use.writes.insert(id);else use.reads.insert(id);
   if(array<4)v.immutableAllocations.insert(id);
  }
  if(stage==0){if(seenArrays!=std::set<unsigned>{0,4,5,6,7})return v81_stop("prework active arrays incomplete");}
  else{std::set<unsigned>required={1,2,3,4,5,6,8,9};if(arm==0)required.insert(7);auto actual=seenArrays;if(arm==1)actual.erase(7);if(actual!=required||!shapeSeen)return v81_stop("state active arrays/shape incomplete");}
  accesses.push_back(use);for(auto&t:c.tg)maxTG=std::max(maxTG,t.first+1);threads|=c.kind==1;groups|=c.kind==2;
  if(!v.psos.count(key(c.p))){id<MTLComputePipelineState>chosen=info.p;if(!chosen.supportIndirectCommandBuffers){if(!oldPD)return v81_stop("ICB descriptor creator unavailable");MTLComputePipelineDescriptor*d=[info.descriptor copy];d.supportIndirectCommandBuffers=YES;NSError*error=nil;MTLComputePipelineReflection*r=nil;chosen=oldPD(S().device,@selector(newComputePipelineStateWithDescriptor:options:reflection:error:),d,info.options,&r,&error);[d release];if(!chosen||error)return v81_stop("ICB-only PSO recreation failed: "+text(error.localizedDescription));++v.recreated;}if(!chosen.supportIndirectCommandBuffers||c.group.width*c.group.height*c.group.depth>chosen.maxTotalThreadsPerThreadgroup)return v81_stop("ICB support/geometry rejected");v.psos[key(c.p)]=chosen;}
 }
 for(auto&slot:v.slots){v81::Bank bank{};for(unsigned a=0;a<10;++a)bank[a]=slot.allocation[a];banks.push_back(bank);}
 auto reason=v81::dependencies(accesses,banks);if(!reason.empty())return v81_stop(reason);
 if(v.immutableAllocations.size()!=60)return v81_stop("immutable allocation count mismatch");
 std::set<uintptr_t>mutableIds;for(auto&slot:v.slots)for(unsigned a=4;a<10;++a)mutableIds.insert(slot.allocation[a]);
 for(auto id:v.shapeAllocations)if(mutableIds.count(id)||v.immutableAllocations.count(id))return v81_stop("shape allocation aliases declared arrays");
 for(auto id:v.immutableAllocations){auto&b=S().buffers.at(id);if(!b.zero||!b.writeStarts.empty())return v81_stop("immutable input written in capture");v.readonlyResources.push_back(b.b);}
 size_t n=std::max(size_t(256),(v.constantBytes+255)&~size_t(255));v.constants=[S().device newBufferWithLength:n options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked];if(!v.constants)return v81_stop("constant allocation failed");
 for(auto*c:v.dispatches)for(auto&b:c->bindings)if(b.second.bytes){auto*d=b.second.bytes;memcpy((char*)v.constants.contents+v.constantOffsets.at(key(d)),d.bytes,d.length);}v.readonlyResources.push_back(v.constants);
 auto*d=[MTLIndirectCommandBufferDescriptor new];d.commandTypes=(threads?MTLIndirectCommandTypeConcurrentDispatchThreads:0)|(groups?MTLIndirectCommandTypeConcurrentDispatch:0);d.inheritBuffers=NO;d.inheritPipelineState=NO;d.maxKernelBufferBindCount=v.maxBind;d.maxKernelThreadgroupMemoryBindCount=maxTG;
 v.icb=[S().device newIndirectCommandBufferWithDescriptor:d maxCommandCount:60 options:0];[d release];if(!v.icb)return v81_stop("ICB allocation failed");[v.icb resetWithRange:NSMakeRange(0,60)];
 for(unsigned i=0;i<60;++i){auto&c=*v.dispatches[i];auto cmd=[v.icb indirectComputeCommandAtIndex:i];if(i%2)[cmd setBarrier];else[cmd clearBarrier];[cmd setComputePipelineState:v.psos.at(key(c.p))];
  for(auto&pair:c.bindings){auto&b=pair.second;id<MTLBuffer>buffer=b.bytes?v.constants:mapped(b.b);size_t off=b.bytes?v.constantOffsets.at(key(b.bytes)):mappedOffset(b.b,b.offset);if(off>UINT32_MAX)return v81_stop("ICB offset exceeds supported32bit range");[cmd setKernelBuffer:buffer offset:off atIndex:pair.first];}
  for(auto&t:c.tg)[cmd setThreadgroupMemoryLength:t.second atIndex:t.first];if(c.kind==1)[cmd concurrentDispatchThreads:c.grid threadsPerThreadgroup:c.group];else[cmd concurrentDispatchThreadgroups:c.grid threadsPerThreadgroup:c.group];
 }
 restoreArena();v.restored=true;v.enabled=true;return 0;
}}
extern "C" int v81_loop_range(uint32_t arm,uint32_t groups){@autoreleasepool{
 std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V81();v.valid=false;v.completed=false;
 if(!v.enabled||!v.restored||!v81::groups(arm,groups)||!S().error.empty()||S().replays)return v81_stop("invalid loop state/range/count");
 auto start=std::chrono::steady_clock::now();auto b=[S().queue commandBuffer];auto e=[b computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
 [e useResource:S().arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];[e useResources:v.readonlyResources.data() count:v.readonlyResources.size() usage:MTLResourceUsageRead];[e useResource:v.icb usage:MTLResourceUsageRead];
 uint64_t encoded=0;for(unsigned i=0;i<groups;++i){[e executeCommandsInBuffer:v.icb withRange:NSMakeRange(arm*30,30)];encoded+=30;[e memoryBarrierWithScope:MTLBarrierScopeBuffers];}
 [e endEncoding];[b commit];[b waitUntilCompleted];auto end=std::chrono::steady_clock::now();
 if(b.status!=MTLCommandBufferStatusCompleted||b.error)return v81_stop("CB failed: "+text(b.error.localizedDescription));
 if(encoded!=uint64_t(groups)*30)return v81_stop("encoded count mismatch");
 v.lastArm=arm;v.lastGroups=groups;v.lastEncoded=encoded;v.lastWall=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();v.gpuStart=b.GPUStartTime;v.gpuEnd=b.GPUEndTime;v.completed=true;v.valid=true;++v.loops;v.totalGroups+=groups;v.totalEncoded+=encoded;return 0;
}}
extern "C" int v81_copy_output(uint32_t arm,uint32_t bank,uint32_t output,uint64_t size,void*destination){
 std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V81();if(!destination||!v.valid||!v.completed||!S().error.empty()||arm!=v.lastArm||bank>=15||output>1||S().replays)return v81_stop("output requires last completed selected range");auto&slot=v.slots[arm*15+bank];unsigned a=8+output;if(size!=slot.size[a])return v81_stop("terminal output size mismatch");auto&b=S().buffers.at(slot.allocation[a]);if(b.zero||b.arena>S().arenaBytes||size>S().arenaBytes-b.arena)return v81_stop("terminal arena mapping invalid");memcpy(destination,(char*)S().arena.contents+b.arena,size);return 0;
}
extern "C" int v81_verify_readonly(){
 std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V81();if(!v.enabled||!S().error.empty())return v81_stop("readonly verification before valid build");
 for(auto id:v.shapeAllocations){auto&b=S().buffers.at(id);if(memcmp((char*)S().arena.contents+b.arena,b.initial.bytes,b.b.length))return v81_stop("readonly S_shape arena buffer changed");}
 for(auto*c:v.dispatches)for(auto&pair:c->bindings)if(pair.second.bytes){auto*d=pair.second.bytes;if(memcmp((char*)v.constants.contents+v.constantOffsets.at(key(d)),d.bytes,d.length))return v81_stop("immutable constant payload changed");}return 0;
}
extern "C" const char*v81_report_json(){@autoreleasepool{
 std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V81();NSMutableArray*records=[NSMutableArray array],*psos=[NSMutableArray array],*mappings=[NSMutableArray array];
 for(unsigned i=0;i<v.dispatches.size();++i){auto&c=*v.dispatches[i];auto&pi=S().psos.at(key(c.p));NSMutableArray*args=[NSMutableArray array];
  for(auto&a:pi.args){if(a.type!=MTLArgumentTypeBuffer)continue;auto&b=c.bindings.at(a.index);if(b.bytes)[args addObject:@{@"index":@(a.index),@"name":[NSString stringWithUTF8String:a.name.c_str()],@"reflection_access":@(a.access),@"kind":@"setBytes",@"bytes":@(b.bytes.length)}];
   else{auto&bi=S().buffers.at(key(b.b));[args addObject:@{@"index":@(a.index),@"name":[NSString stringWithUTF8String:a.name.c_str()],@"reflection_access":@(a.access),@"source_store_only_RW_exception":@(v81::storeOnly(i/30,i%2,pi.name,a.index,a.name,(unsigned)a.access)),@"kind":@"actual_MTLBuffer",@"original_buffer":@(key(b.b)),@"original_address":@((uintptr_t)b.b.contents+b.offset),@"original_offset":@(b.offset),@"allocation_bytes":@(b.b.length),@"zero_copy":@(bi.zero),@"mapped_offset":@(mappedOffset(b.b,b.offset))}];}}
  [records addObject:@{@"ordinal":@(i),@"arm":@(i/30),@"bank":@((i%30)/2),@"stage":@(i%2),@"function":[NSString stringWithUTF8String:pi.name.c_str()],@"dispatch_kind":@(c.kind),@"grid":@[@(c.grid.width),@(c.grid.height),@(c.grid.depth)],@"threadgroup":@[@(c.group.width),@(c.group.height),@(c.group.depth)],@"captured_boundaries_before":@(v.capturedBefore[i]),@"ICB_barrier_before":@(i%2==1),@"actual_active_arguments":args}];
 }
 for(auto&p:v.psos){auto&pi=S().psos.at(p.first);[psos addObject:@{@"function":[NSString stringWithUTF8String:pi.name.c_str()],@"supports_ICB":@(p.second.supportIndirectCommandBuffers),@"recreated":@(p.second!=pi.p),@"descriptor_change":@"only supportsIndirectCommandBuffers=YES when needed"}];}
 for(unsigned i=0;i<30;++i)if(v.slots[i].registered){NSMutableArray*arrays=[NSMutableArray array];for(unsigned a=0;a<10;++a){auto&slot=v.slots[i];auto it=S().buffers.find(slot.allocation[a]);[arrays addObject:@{@"array":@(a),@"logical_address":@(slot.address[a]),@"logical_bytes":@(slot.size[a]),@"original_allocation":@(slot.allocation[a]),@"allocation_bytes":@(it==S().buffers.end()?0:it->second.b.length),@"arena_offset":@(it==S().buffers.end()?0:it->second.arena)}];}[mappings addObject:@{@"arm":@(i/15),@"bank":@(i%15),@"arrays":arrays}];}
 bool timestamps=std::isfinite(v.gpuStart)&&std::isfinite(v.gpuEnd)&&v.gpuStart>0&&v.gpuEnd>=v.gpuStart;
 NSDictionary*r=@{@"status":S().error.empty()?@"OK":@"STOP",@"error":[NSString stringWithUTF8String:S().error.c_str()],@"enabled":@(v.enabled),@"capture_dispatches":@(S().dispatches),@"ICB_commands":@(v.enabled?60:0),@"ranges":@[@[@0,@30],@[@30,@60]],@"fixed_saved_banks":@15,@"immutable_registered_ranges":@(S().pins.size()),@"immutable_allocations":@(v.immutableAllocations.size()),@"mutable_registered_outputs":@(S().outputs.size()),@"source_access_policy":@"exact four functions and index/name store-only output RW exception; all inputs including S require actual ReadOnly",@"producer_consumer_edges_verified":@(v.enabled),@"original_and_arena_ranges_disjoint":@(v.enabled),@"internal_state_barriers_per_range":@15,@"trailing_Buffers_barriers_last_loop":@(v.lastGroups),@"last_arm":@(v.lastArm),@"last_groups":@(v.lastGroups),@"observed_encoded_dispatches":@(v.lastEncoded),@"last_CB_completed":@(v.completed),@"last_native_wall_ns":@(v.lastWall),@"CBGPUStartTime_seconds":@(v.gpuStart),@"CBGPUEndTime_seconds":@(v.gpuEnd),@"CBGPU_timestamps_valid":@(timestamps),@"loops":@(v.loops),@"total_groups":@(v.totalGroups),@"total_encoded_dispatches":@(v.totalEncoded),@"arena_restore_once":@(v.restored),@"arena_bytes":@(S().arenaBytes),@"snapshot_bytes":@(S().snapshotBytes),@"zero_copy_bytes":@(S().zeroBytes),@"constant_bytes":@(v.constantBytes),@"max_buffer_bind_count":@(v.maxBind),@"PSOs":psos,@"dispatch_records":records,@"bank_mappings":mappings,@"legacy_feedback_enabled":@(F().sealed),@"legacy_ICB_enabled":@(I().enabled)};
 NSData*d=[NSJSONSerialization dataWithJSONObject:r options:0 error:nil];v.report.assign((const char*)d.bytes,d.length);return v.report.c_str();
}}
