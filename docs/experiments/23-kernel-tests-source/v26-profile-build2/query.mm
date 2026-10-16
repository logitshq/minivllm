#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
int main() { @autoreleasepool {
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();
 if(!d){puts("{\"device_available\":false,\"gpu_dispatches\":0}");return 2;}
 NSMutableArray *sets=[NSMutableArray array];
 for(id<MTLCounterSet> s in d.counterSets) {NSMutableArray *c=[NSMutableArray array]; for(id<MTLCounter> x in s.counters)[c addObject:x.name]; [sets addObject:@{@"name":s.name,@"counters":c}];}
 id<MTLCommandQueue> q=[d newCommandQueue]; id<MTLCommandBuffer> b=[q commandBuffer];
 id<MTLComputeCommandEncoder> e=[b computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
 NSDictionary *r=@{@"device":d.name,@"dispatch_boundary":@([d supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),@"stage_boundary":@([d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]),@"counter_sets":sets,@"device_class":NSStringFromClass([d class]),@"command_buffer_class":NSStringFromClass([b class]),@"compute_encoder_class":NSStringFromClass([e class]),@"gpu_dispatches":@0};
 NSData *data=[NSJSONSerialization dataWithJSONObject:r options:NSJSONWritingPrettyPrinted error:nil];puts([[NSString alloc]initWithData:data encoding:NSUTF8StringEncoding].UTF8String);[e endEncoding];
} return 0; }
