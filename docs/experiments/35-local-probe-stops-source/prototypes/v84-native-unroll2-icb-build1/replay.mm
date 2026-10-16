// Frozen base is included verbatim. Existing exports/semantics are unchanged.
#include "/private/tmp/minivllm-engine-optimization-20261014/v40-native-icb-build2/replay.mm"
#include "range_contract.h"
#include <cmath>
#include <algorithm>
struct V84Slot { uint64_t a[8]{}; bool registered=false; };
struct V84State {
 bool enabled=false,restored=false,lastValid=false;unsigned lastArm=0,recreated=0;uint64_t loops=0,groups=0,encoded=0;
 V84Slot slots[16];id<MTLIndirectCommandBuffer> icb=nil;id<MTLBuffer> constants=nil;size_t constantBytes=0;
 std::map<uintptr_t,size_t> offsets;std::map<uintptr_t,id<MTLComputePipelineState>> psos;
 std::vector<id<MTLResource>> readonly;std::vector<Command*> dispatches;std::set<uintptr_t> readonlyAllocations;
 std::string report;uint64_t lastGroups=0,lastEncoded=0,lastWall=0;double gpuStart=0,gpuEnd=0;bool completed=false;NSUInteger maxBindCount=0;
};
static V84State& V84(){static V84State x;return x;}
static int v84_stop(const std::string&why){V84().lastValid=false;fail("V84: "+why);return 1;}
extern "C" int v84_register_slot(uint32_t arm,uint32_t slot,uint64_t w,uint64_t wn,uint64_t s,uint64_t sn,uint64_t x,uint64_t xn,uint64_t y,uint64_t yn){
 std::lock_guard<std::recursive_mutex>lock(S().lock);if(arm>1||slot>7||!S().built||V84().enabled)return v84_stop("invalid/late slot");
 auto&r=V84().slots[arm*8+slot];if(r.registered)return v84_stop("duplicate slot");uint64_t a[]={w,wn,s,sn,x,xn,y,yn};
 for(int i=0;i<8;i+=2)if(!a[i]||!a[i+1]||a[i]+a[i+1]<a[i])return v84_stop("invalid range");
 std::copy(a,a+8,r.a);r.registered=true;return 0;
}
static bool v84_exact_read(const Command&c,uintptr_t addr,size_t n){
 for(auto&a:c.bindings){auto&b=a.second;if(b.bytes||!b.b)continue;auto&bi=S().buffers.at(key(b.b));uintptr_t base=(uintptr_t)b.b.contents;
  if(base+b.offset==addr&&bi.zero&&addr>=base&&n<=b.b.length&&addr-base<=b.b.length-n){auto*p=parameter(addr);if(p&&p->addr==addr&&p->size==n)return true;}}
 return false;
}
extern "C" int v84_build_ranges(){@autoreleasepool{
 std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V84();
 if(!S().built||S().armed||!S().error.empty()||F().sealed||I().enabled||!PP().pairs.empty()||v.enabled||S().replays)return v84_stop("requires fresh captured graph without feedback/legacyreplay");
 if(S().pins.size()!=24)return v84_stop("requires24 distinct original W/scales/X pins");
 for(unsigned i=0;i<8;++i)for(unsigned k=0;k<6;++k)if(!v.slots[i].registered||!v.slots[i+8].registered||v.slots[i].a[k]!=v.slots[i+8].a[k])return v84_stop("A/B bank original inputs differ");
 if(S().dispatches!=16||S().outputs.size()!=16)return v84_stop("exact16 dispatches/registeredoutputs required");
 NSUInteger maxBind=0,maxTG=0;unsigned threads=0,groups=0;std::vector<V84Access> accesses;
 for(auto&c:S().commands){if(c.kind>=3&&c.kind<=8)continue;if(c.kind!=1&&c.kind!=2)return v84_stop("unknown captured command kind");v.dispatches.push_back(&c);}
 if(v.dispatches.size()!=16)return v84_stop("dispatch extraction mismatch");
 for(unsigned i=0;i<16;++i){auto&c=*v.dispatches[i];auto&slot=v.slots[i];if(!slot.registered)return v84_stop("slot unregistered");
  auto pi=S().psos.find(key(c.p));if(pi==S().psos.end()||!pi->second.reflection||!pi->second.function||!pi->second.descriptor)return v84_stop("actual reflection/function/descriptor unavailable");
  auto&info=pi->second;std::string fn=info.name;
  if(!v84_exact_function(i/8,fn))return v84_stop("actual function differs from exact source-derived EXPECTED A/B; no RW exception granted");
  if(info.args.size()!=(i<8?6:4))return v84_stop("active argument count differs from exact source schema");
  if(i<8){if(c.kind!=2||c.grid.width!=1||c.grid.height!=2176||c.grid.depth!=1||c.group.width!=32||c.group.height!=2||c.group.depth!=1)return v84_stop("builtin original R4 geometry changed");}
  else if(c.kind!=1||c.grid.width!=64||c.grid.height!=2176||c.grid.depth!=1||c.group.width!=64||c.group.height!=1||c.group.depth!=1)return v84_stop("frozen83 R4/SG2 geometry changed");
  if(slot.a[1]!=17408ull*640*4||slot.a[3]!=17408ull*320||slot.a[5]!=10240||slot.a[7]!=34816)return v84_stop("slot shape-byte contract differs from L0gate shape");
  V84Access use;unsigned writes=0;
  for(auto&binding:c.bindings)maxBind=std::max(maxBind,binding.first+1); // every encoded binding, not only active-table max
  for(auto&a:info.args){if(a.type==MTLArgumentTypeThreadgroupMemory){if(!c.tg.count(a.index))return v84_stop("missing threadgroup bank");continue;}if(a.type!=MTLArgumentTypeBuffer)return v84_stop("unknown active reflection type");
   auto it=c.bindings.find(a.index);if(it==c.bindings.end())return v84_stop("missing active binding");auto&b=it->second;maxBind=std::max(maxBind,a.index+1);
   if(!v84_exact_argument(i/8,a.index,a.name,(unsigned)a.access,b.bytes!=nil,b.bytes?b.bytes.length:0))return v84_stop("active name/index/access/binding-kind schema differs");
   if(b.bytes){int value=0;memcpy(&value,b.bytes.bytes,4);if(value!=(a.index==5?5120:17408))return v84_stop("builtin K/N constant changed");}
   else if(a.access==MTLArgumentAccessReadOnly){unsigned role=a.index==0?0:a.index==1?2:4;uintptr_t base=(uintptr_t)b.b.contents;
    if(!b.b||base+b.offset!=slot.a[role])return v84_stop("original typed W/scales/X role address differs");auto*pin=parameter(slot.a[role]);if(!pin||pin->addr!=slot.a[role]||pin->size!=slot.a[role+1])return v84_stop("original typed operand full range unresolved");}
   if(b.bytes){if(a.access!=MTLArgumentAccessReadOnly)return v84_stop("writable setBytes");if(!v.offsets.count(key(b.bytes))){size_t off=(v.constantBytes+255)&~size_t(255);v.offsets[key(b.bytes)]=off;v.constantBytes=off+b.bytes.length;}continue;}
   if(!b.b||!b.b.contents)return v84_stop("unmapped active buffer");auto id=key(b.b);
   if(a.access==MTLArgumentAccessReadOnly){use.reads.insert(id);v.readonlyAllocations.insert(id);}
   else if(a.access==MTLArgumentAccessWriteOnly||v84_source_writeonly(i/8,fn,a.index,a.name,(unsigned)a.access)){++writes;use.write=id;auto&bi=S().buffers.at(id);uintptr_t addr=(uintptr_t)b.b.contents+b.offset;
    if(addr!=slot.a[6]||b.offset!=0||addr!=(uintptr_t)b.b.contents||bi.zero||bi.pinCandidate)return v84_stop("writeonly binding does not match original slot output");
    auto o=std::find_if(S().outputs.begin(),S().outputs.end(),[&](const Output&o){return o.addr==slot.a[6]&&o.size==slot.a[7]&&o.buffer==id;});if(o==S().outputs.end())return v84_stop("unregistered output binding");}
   else return v84_stop("active readwrite/unknown access forbidden");
  }
  if(writes!=1)return v84_stop("one exclusive output per dispatch required");
  for(int p=0;p<6;p+=2)if(!v84_exact_read(c,slot.a[p],slot.a[p+1]))return v84_stop("W/scales/X original readonly pin unresolved");
  for(auto&x:c.tg)maxTG=std::max(maxTG,x.first+1);accesses.push_back(use);if(c.kind==1)++threads;else ++groups;
  if(!v.psos.count(key(c.p))){id<MTLComputePipelineState>chosen=info.p;if(!chosen.supportIndirectCommandBuffers){if(!oldPD)return v84_stop("ICB descriptor creator unavailable");MTLComputePipelineDescriptor*d=[info.descriptor copy];d.supportIndirectCommandBuffers=YES;NSError*error=nil;MTLComputePipelineReflection*r=nil;chosen=oldPD(S().device,@selector(newComputePipelineStateWithDescriptor:options:reflection:error:),d,info.options,&r,&error);[d release];if(!chosen||error)return v84_stop("ICB PSO recreation failed: "+text(error.localizedDescription));++v.recreated;}if(!chosen.supportIndirectCommandBuffers)return v84_stop("ICB support missing");v.psos[key(c.p)]=chosen;}
  auto chosen=v.psos.at(key(c.p));if(c.group.width*c.group.height*c.group.depth>chosen.maxTotalThreadsPerThreadgroup)return v84_stop("original geometry unsupported by ICB PSO");
 }
 auto why=v84_independent(accesses);if(!why.empty())return v84_stop(why);
 for(auto id:v.readonlyAllocations){auto&b=S().buffers.at(id);if(!b.writeStarts.empty())return v84_stop("readonly allocation written anywhere in capture");if(b.zero)v.readonly.push_back(b.b);else if(!b.initial)return v84_stop("small readonly allocation lacks snapshot");}
 size_t allocation=std::max(size_t(256),(v.constantBytes+255)&~size_t(255));v.constants=[S().device newBufferWithLength:allocation options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked];if(!v.constants)return v84_stop("constant allocation failed");
 for(auto*c:v.dispatches)for(auto&a:c->bindings)if(a.second.bytes){auto*d=a.second.bytes;memcpy((char*)v.constants.contents+v.offsets.at(key(d)),d.bytes,d.length);}v.readonly.push_back(v.constants);
 v.maxBindCount=maxBind;auto*d=[MTLIndirectCommandBufferDescriptor new];d.commandTypes=(threads?MTLIndirectCommandTypeConcurrentDispatchThreads:0)|(groups?MTLIndirectCommandTypeConcurrentDispatch:0);d.inheritBuffers=NO;d.inheritPipelineState=NO;d.maxKernelBufferBindCount=maxBind;d.maxKernelThreadgroupMemoryBindCount=maxTG;
 v.icb=[S().device newIndirectCommandBufferWithDescriptor:d maxCommandCount:16 options:0];[d release];if(!v.icb)return v84_stop("ICB allocation failed");[v.icb resetWithRange:NSMakeRange(0,16)];
 for(unsigned i=0;i<16;++i){auto&c=*v.dispatches[i];auto cmd=[v.icb indirectComputeCommandAtIndex:i];[cmd clearBarrier];[cmd setComputePipelineState:v.psos.at(key(c.p))];
  for(auto&a:c.bindings){auto&b=a.second;id<MTLBuffer>buffer=b.bytes?v.constants:mapped(b.b);size_t off=b.bytes?v.offsets.at(key(b.bytes)):mappedOffset(b.b,b.offset);if(off>UINT32_MAX)return v84_stop("ICB offset too large");[cmd setKernelBuffer:buffer offset:off atIndex:a.first];}
  for(auto&x:c.tg)[cmd setThreadgroupMemoryLength:x.second atIndex:x.first];if(c.kind==1)[cmd concurrentDispatchThreads:c.grid threadsPerThreadgroup:c.group];else[cmd concurrentDispatchThreadgroups:c.grid threadsPerThreadgroup:c.group];
 }
 restoreArena();v.restored=true;v.enabled=true;return 0;
}}
extern "C" int v84_loop_range(uint32_t arm,uint32_t groups){@autoreleasepool{
 std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V84();v.lastValid=false;v.completed=false;
 if(S().replays||!v.enabled||!v.restored||arm>1||groups<1||groups>8192||!S().error.empty())return v84_stop("invalid range/groups/state");
 auto begin=std::chrono::steady_clock::now();id<MTLCommandBuffer>b=[S().queue commandBuffer];id<MTLComputeCommandEncoder>e=[b computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
 [e useResource:S().arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];[e useResources:v.readonly.data() count:v.readonly.size() usage:MTLResourceUsageRead];[e useResource:v.icb usage:MTLResourceUsageRead];
 uint64_t encoded=0;for(unsigned i=0;i<groups;++i){[e executeCommandsInBuffer:v.icb withRange:NSMakeRange(arm*8,8)];encoded+=8;[e memoryBarrierWithScope:MTLBarrierScopeBuffers];}
 [e endEncoding];[b commit];[b waitUntilCompleted];auto end=std::chrono::steady_clock::now();
 if(b.status!=MTLCommandBufferStatusCompleted||b.error)return v84_stop("CB failed: "+text(b.error.localizedDescription));
 if(encoded!=uint64_t(groups)*8)return v84_stop("encoded dispatch count mismatch");
 v.lastArm=arm;v.lastGroups=groups;v.lastEncoded=encoded;v.lastWall=std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();v.gpuStart=b.GPUStartTime;v.gpuEnd=b.GPUEndTime;v.completed=true;v.lastValid=true;++v.loops;v.groups+=groups;v.encoded+=encoded;return 0;
}}
extern "C" int v84_copy_output(uint32_t arm,uint32_t slot,uint64_t n,void*dst){std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V84();if(!dst||S().replays||!S().error.empty()||!v.lastValid||arm!=v.lastArm||slot>7||n!=v.slots[arm*8+slot].a[7])return v84_stop("output read requires last completed selected range");auto addr=v.slots[arm*8+slot].a[6];for(auto&o:S().outputs)if(o.addr==addr&&o.size==n){auto&b=S().buffers.at(o.buffer);memcpy(dst,(char*)S().arena.contents+b.arena+o.offset,n);return 0;}return v84_stop("output unmapped");}
extern "C" int v84_verify_readonly(){std::lock_guard<std::recursive_mutex>lock(S().lock);if(!V84().enabled)return v84_stop("readonly check before build");for(auto id:V84().readonlyAllocations){auto&b=S().buffers.at(id);if(!b.zero&&memcmp((char*)S().arena.contents+b.arena,b.initial.bytes,b.b.length))return v84_stop("small readonly arena input changed");}return 0;}
extern "C" const char*v84_report_json(){@autoreleasepool{std::lock_guard<std::recursive_mutex>lock(S().lock);auto&v=V84();NSMutableArray*psos=[NSMutableArray array];for(auto&p:v.psos){auto&o=S().psos.at(p.first);[psos addObject:@{@"function":[NSString stringWithUTF8String:o.name.c_str()],@"actual_supports_ICB":@(p.second.supportIndirectCommandBuffers),@"recreated":@(p.second!=o.p),@"descriptor_change":@"only supportIndirectCommandBuffers=YES when needed"}];}
 NSMutableArray*slots=[NSMutableArray array];for(unsigned i=0;i<v.dispatches.size();++i){auto&c=*v.dispatches[i];auto&pi=S().psos.at(key(c.p));NSMutableArray*args=[NSMutableArray array];for(auto&a:pi.args){if(a.type!=MTLArgumentTypeBuffer)continue;auto&b=c.bindings.at(a.index);if(b.bytes){[args addObject:@{@"index":@(a.index),@"name":[NSString stringWithUTF8String:a.name.c_str()],@"access":@(a.access),@"kind":@"setBytes",@"bytes":@(b.bytes.length)}];}else{auto&bi=S().buffers.at(key(b.b));[args addObject:@{@"index":@(a.index),@"name":[NSString stringWithUTF8String:a.name.c_str()],@"access":@(a.access),@"source_store_only_output_exception":@(v84_source_writeonly(i/8,pi.name,a.index,a.name,(unsigned)a.access)),@"kind":@"actual_MTLBuffer",@"original_buffer":@(key(b.b)),@"original_binding_address":@((uintptr_t)b.b.contents+b.offset),@"original_offset":@(b.offset),@"allocation_bytes":@(b.b.length),@"zero_copy":@(bi.zero),@"mapped_offset":@(mappedOffset(b.b,b.offset))}];}}[slots addObject:@{@"arm":@(i/8),@"slot":@(i%8),@"function":[NSString stringWithUTF8String:pi.name.c_str()],@"dispatch_kind":@(c.kind),@"grid":@[@(c.grid.width),@(c.grid.height),@(c.grid.depth)],@"threadgroup":@[@(c.group.width),@(c.group.height),@(c.group.depth)],@"actual_active_arguments":args}];}
 bool valid=std::isfinite(v.gpuStart)&&std::isfinite(v.gpuEnd)&&v.gpuStart>0&&v.gpuEnd>=v.gpuStart;
 NSDictionary*r=@{@"status":S().error.empty()?@"OK":@"STOP",@"error":[NSString stringWithUTF8String:S().error.c_str()],@"enabled":@(v.enabled),@"capture_dispatches":@(S().dispatches),@"commands_in_single_ICB":@(v.enabled?16:0),@"maxKernelBufferBindCount_all_recorded_bindings":@(v.maxBindCount),@"ranges":@[@[@0,@8],@[@8,@16]],@"independent_allocation_proof":@(v.enabled),@"output_allocations":@16,@"resolved_W_scale_X_bindings":@48,@"unique_registered_RO_ranges":@(S().pins.size()),@"readonly_allocations":@(v.readonlyAllocations.size()),@"captured_envelope_commands_not_replayed":@(S().commands.size()-S().dispatches),@"internal_ICB_barriers":@0,@"output_access_policy":@"reflection WriteOnly OR exact source-derived function+index+name RW output with pinned store-only proof; otherRW rejected",@"B_function_CPU_provenance":@"EXPECTED_FROM_FROZEN83_NAME_TEMPLATE_NOT_PREVIOUSLY_OBSERVED",@"actual_function_and_schema_matched":@(v.enabled),@"buffer_barriers_per_last_loop":@(v.lastGroups),@"last_arm":@(v.lastArm),@"last_groups":@(v.lastGroups),@"observed_encoded_dispatches":@(v.lastEncoded),@"last_CB_completed":@(v.completed),@"last_native_wall_ns":@(v.lastWall),@"CBGPUStartTime_seconds":@(v.gpuStart),@"CBGPUEndTime_seconds":@(v.gpuEnd),@"CBGPU_timestamps_valid":@(valid),@"loops":@(v.loops),@"total_groups":@(v.groups),@"total_encoded_dispatches":@(v.encoded),@"arena_initial_restore_once":@(v.restored),@"arena_bytes":@(S().arenaBytes),@"snapshot_bytes":@(S().snapshotBytes),@"zero_copy_buffer_bytes":@(S().zeroBytes),@"constant_bank_bytes":@(v.constantBytes),@"PSOs":psos,@"slot_records":slots,@"legacy_feedback_enabled":@(F().sealed),@"legacy_ICB_enabled":@(I().enabled)};NSData*d=[NSJSONSerialization dataWithJSONObject:r options:0 error:nil];v.report.assign((const char*)d.bytes,d.length);return v.report.c_str();}}
