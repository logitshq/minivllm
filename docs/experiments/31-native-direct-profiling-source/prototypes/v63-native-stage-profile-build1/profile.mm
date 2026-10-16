// Diagnostic extension: frozen native40 remains byte-for-byte untouched.
// Stage envelopes come from one encoder per original ICB command. The added
// fences serialize more work, so these are not production exclusive shares.
#include "/private/tmp/minivllm-engine-optimization-20261014/v40-native-icb-build2/replay.mm"
static std::string profile63;
static std::string category63(const std::string& n) {
 if(n.find("nvfp4_qmv")!=std::string::npos)return "NVFP4-QMV";
 if(n.find("gdn_norm_reuse_cached_state")!=std::string::npos)return "GDN-state";
 if(n.find("gdn_activated_norm_reuse_prework")!=std::string::npos)return "GDN-prework";
 if(n.find("gdn_conv_silu")!=std::string::npos)return "GDN-conv-SiLU";
 if(n.find("explicit_norm")!=std::string::npos)return "norm";
 if(n.find("sdpa_")!=std::string::npos)return "SDPA";
 return "other";
}
static std::vector<const Command*> commands63() {
 std::vector<const Command*> out;
 for(auto& c:S().commands)if(c.kind==1||c.kind==2)out.push_back(&c);
 return out;
}
extern "C" int v63_profile_step(uint32_t sampled){@autoreleasepool{
 std::lock_guard<std::recursive_mutex> lock(S().lock);
 if(sampled>1||!I().enabled||!F().sealed||!F().ready||!S().error.empty()||(!F().written.empty()&&F().written.size()!=F().inputs.size())){fail("profile requires valid seeded synchronous ICB graph and sampled bool");return 1;}
 int32_t token=-1,pos=-1,len=-1;
 for(auto&x:F().inputs){auto*p=(const char*)S().arena.contents+arenaInput(x);if(x.role==1)memcpy(&token,p,4);if(x.role==2)memcpy(&pos,p,4);if(x.role==3)memcpy(&len,p,4);}
 if(token<0||pos<0||len!=pos+1||len>(int)F().capacity){fail("profile scalar guard failed");return 1;}
 auto cmds=commands63();if(cmds.size()!=I().dispatches||cmds.empty()){fail("profile command ordinal map mismatch");return 1;}
 id<MTLCounterSampleBuffer> counter=nil;
 if(sampled){
  if(![S().device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]){fail("stage counters unsupported");return 1;}
  id<MTLCounterSet> set=nil;for(id<MTLCounterSet> x in S().device.counterSets)if([x.name isEqualToString:MTLCommonCounterSetTimestamp])set=x;
  if(!set){fail("timestamp counter set absent");return 1;}
  MTLCounterSampleBufferDescriptor*d=[[MTLCounterSampleBufferDescriptor alloc]init];d.counterSet=set;d.storageMode=MTLStorageModeShared;d.sampleCount=cmds.size()*2;
  NSError*err=nil;counter=[S().device newCounterSampleBufferWithDescriptor:d error:&err];[d release];
  if(!counter||err){fail("counter allocation failed: "+text(err.localizedDescription));return 1;}
 }
 id<MTLFence> chain=[S().device newFence];if(!chain){[counter release];fail("profile chain fence allocation failed");return 1;}
 uint64_t t0=nowNS();id<MTLCommandBuffer>b=[S().queue commandBuffer];
 for(size_t j=0;j<cmds.size();++j){
  MTLComputePassDescriptor*d=[MTLComputePassDescriptor computePassDescriptor];d.dispatchType=MTLDispatchTypeConcurrent;
  if(sampled){auto*a=d.sampleBufferAttachments[0];a.sampleBuffer=counter;a.startOfEncoderSampleIndex=j*2;a.endOfEncoderSampleIndex=j*2+1;}
  id<MTLComputeCommandEncoder>e=[b computeCommandEncoderWithDescriptor:d];
  if(!e){[chain release];[counter release];fail("profile encoder allocation failed before commit");return 1;}
  if(j)[e waitForFence:chain];else if(F().steps)[e waitForFence:F().feedbackDone];
  [e useResource:S().arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];
  if(!I().readonlyResources.empty())[e useResources:I().readonlyResources.data() count:I().readonlyResources.size() usage:MTLResourceUsageRead];
  [e useResource:I().banks[PP().bank] usage:MTLResourceUsageRead];
  [e executeCommandsInBuffer:I().banks[PP().bank] withRange:NSMakeRange(j,1)];
  [e updateFence:chain];if(j+1==cmds.size())[e updateFence:F().graphDone];[e endEncoding];
 }
 ++I().encoded;uint64_t tGraph=nowNS();
 id<MTLBlitCommandEncoder>capture=[b blitCommandEncoder];[capture waitForFence:F().graphDone];
 for(auto&x:S().outputs){if(pingOutput(x.addr,x.size))continue;auto*t=stage(x.addr,x.size);[capture copyFromBuffer:S().arena sourceOffset:arenaOutput(x) toBuffer:F().stage destinationOffset:t->offset size:x.size];}
 [capture updateFence:F().stageDone];[capture endEncoding];
 id<MTLBlitCommandEncoder>feedback=[b blitCommandEncoder];[feedback waitForFence:F().stageDone];
 for(auto&x:F().links){if(pingOutput(x.output,x.size))continue;auto*i=input(x.input,x.size);auto*t=stage(x.output,x.size);[feedback copyFromBuffer:F().stage sourceOffset:t->offset toBuffer:S().arena destinationOffset:arenaInput(*i) size:x.size];}
 [feedback updateFence:F().feedbackDone];[feedback endEncoding];uint64_t tEncode=nowNS();[b commit];uint64_t tCommit=nowNS();[b waitUntilCompleted];uint64_t tWait=nowNS();double gs=b.GPUStartTime,ge=b.GPUEndTime;
 F().timing.push_back({F().resets,F().steps+1,tGraph-t0,tEncode-tGraph,tEncode-t0,tCommit-tEncode,tWait-tCommit,tWait-t0,gs,ge,gs>0&&ge>=gs});
 bool ok=b.status==MTLCommandBufferStatusCompleted&&!b.error;
 if(!ok){[chain release];[counter release];fail("profile GPU CB failed: "+text(b.error.localizedDescription));return 1;}
 NSMutableArray*rows=[NSMutableArray array];NSData*data=sampled?[counter resolveCounterRange:NSMakeRange(0,cmds.size()*2)]:nil;
 if(sampled&&(!data||data.length!=cmds.size()*2*sizeof(MTLCounterResultTimestamp))){[chain release];[counter release];fail("counter resolve length mismatch after completed CB");return 1;}
 const MTLCounterResultTimestamp*ts=sampled?(const MTLCounterResultTimestamp*)data.bytes:nullptr;
 for(size_t j=0;j<cmds.size();++j){auto&c=*cmds[j];auto name=S().psos.at(key(c.p)).name;NSMutableArray*bindings=[NSMutableArray array];
  for(auto&entry:c.bindings){auto&v=entry.second;[bindings addObject:@{@"index":@(entry.first),@"bytes_payload_base64":v.bytes?[v.bytes base64EncodedStringWithOptions:0]:@"",@"original_buffer":@(key(v.b)),@"offset":@(v.offset)}];}
  uint64_t start=sampled?ts[j*2].timestamp:0,end=sampled?ts[j*2+1].timestamp:0;bool valid=sampled&&start!=MTLCounterErrorValue&&end!=MTLCounterErrorValue&&start&&end>=start;
  [rows addObject:@{@"ICB_ordinal":@(j),@"PSO":[NSString stringWithUTF8String:name.c_str()],@"category":[NSString stringWithUTF8String:category63(name).c_str()],@"sampled":@(sampled),@"raw_start":@(start),@"raw_end":@(end),@"valid":@(valid),@"raw_delta":valid?@(end-start):(id)[NSNull null],@"bindings":bindings}];
 }
 NSDictionary*r=@{@"sampled":@(sampled),@"segments":@(cmds.size()),@"request":@(F().resets),@"step":@(F().steps+1),@"CB_GPUStartTime_seconds":@(gs),@"CB_GPUEndTime_seconds":@(ge),@"rows":rows,@"scope":@"perturbed serialized one-ICB-command-per-encoder GPU stage envelopes; not production exclusive kernel time/shares or performance adoption"};
 NSData*json=[NSJSONSerialization dataWithJSONObject:r options:0 error:nil];profile63.assign((const char*)json.bytes,json.length);
 ++I().executions;++I().bankExecutions[PP().bank];PP().bank^=1;++F().steps;++F().totalSteps;++S().replays;
 [chain release];[counter release];return 0;
}}
extern "C" const char*v63_profile_json(){return profile63.c_str();}
