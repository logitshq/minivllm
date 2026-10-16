#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <atomic>
#include <chrono>
#include <mutex>

// Diagnostic-only Objective-C hooks. No GPU work is submitted by initialization.
static const void *CBKey=&CBKey,*EncoderCB=&EncoderCB,*KernelKey=&KernelKey,*ArgsKey=&ArgsKey,*SpanKey=&SpanKey;
static uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct State {
 std::mutex lock; std::atomic<bool> armed{false}; int step=-1; uint64_t next=0;
 NSMutableDictionary *records=[[NSMutableDictionary alloc]init];
 NSMutableDictionary *names=[[NSMutableDictionary alloc]init];
 NSMutableArray *hooks=[[NSMutableArray alloc]init]; NSString *error=nil;
};
static State& S(){static State s;return s;}
static NSNumber *ptr(id p){return @((uintptr_t)p);}
static NSNumber *cbid(id b){NSNumber *n=objc_getAssociatedObject(b,CBKey);if(!n){std::lock_guard<std::mutex> l(S().lock);n=@(++S().next);objc_setAssociatedObject(b,CBKey,n,OBJC_ASSOCIATION_RETAIN_NONATOMIC);}return n;}
static void attach(id e,id b){if(e){cbid(b);objc_setAssociatedObject(e,EncoderCB,b,OBJC_ASSOCIATION_ASSIGN);objc_setAssociatedObject(e,SpanKey,@(now_ns()),OBJC_ASSOCIATION_RETAIN_NONATOMIC);}}
static void remember(id p,id f){if(p){NSString *n=[f name]?:[p label]?:@"unknown";std::lock_guard<std::mutex> l(S().lock);S().names[ptr(p)]=n;}}
static id(*origPso)(id,SEL,id,NSError**);
static id pso(id s,SEL c,id f,NSError **e){id p=origPso(s,c,f,e);remember(p,f);return p;}
static id(*origPsoOptions)(id,SEL,id,NSUInteger,void*,NSError**);
static id psoOptions(id s,SEL c,id f,NSUInteger o,void *r,NSError **e){id p=origPsoOptions(s,c,f,o,r,e);remember(p,f);return p;}
static id(*origPsoDesc)(id,SEL,id,NSUInteger,void*,NSError**);
static id psoDesc(id s,SEL c,id d,NSUInteger o,void *r,NSError **e){id p=origPsoDesc(s,c,d,o,r,e);remember(p,[d computeFunction]);return p;}
static id(*origEncoder)(id,SEL);
static id encoder(id b,SEL c){id e=origEncoder(b,c);attach(e,b);return e;}
static id(*origEncoderType)(id,SEL,NSUInteger);
static id encoderType(id b,SEL c,NSUInteger t){id e=origEncoderType(b,c,t);attach(e,b);return e;}
static id(*origEncoderDesc)(id,SEL,id);
static id encoderDesc(id b,SEL c,id d){id e=origEncoderDesc(b,c,d);attach(e,b);return e;}
static void(*origSetPso)(id,SEL,id);
static void setPso(id e,SEL c,id p){origSetPso(e,c,p);NSString *n=nil;{std::lock_guard<std::mutex> l(S().lock);n=S().names[ptr(p)];}objc_setAssociatedObject(e,KernelKey,n?:[p label]?:@"unknown",OBJC_ASSOCIATION_RETAIN_NONATOMIC);objc_setAssociatedObject(e,ArgsKey,[NSMutableDictionary dictionary],OBJC_ASSOCIATION_RETAIN_NONATOMIC);}
static void(*origBytes)(id,SEL,const void*,NSUInteger,NSUInteger);
static void setBytes(id e,SEL c,const void *p,NSUInteger n,NSUInteger i){origBytes(e,c,p,n,i);if(!S().armed)return;NSMutableDictionary *a=objc_getAssociatedObject(e,ArgsKey);if(!a)return;NSMutableArray *v=[NSMutableArray array];if(n<=32 && n%4==0){for(NSUInteger j=0;j<n/4;++j){int32_t x;memcpy(&x,(const char*)p+4*j,4);[v addObject:@(x)];}}a[[NSString stringWithFormat:@"bytes_%lu",(unsigned long)i]]=@{@"length":@(n),@"int32_values":v};}
static void(*origBuffer)(id,SEL,id,NSUInteger,NSUInteger);
static void setBuffer(id e,SEL c,id b,NSUInteger o,NSUInteger i){origBuffer(e,c,b,o,i);if(!S().armed)return;NSMutableDictionary *a=objc_getAssociatedObject(e,ArgsKey);if(a)a[[NSString stringWithFormat:@"buffer_%lu",(unsigned long)i]]=@{@"length":@([b length]),@"offset":@(o)};}
static NSMutableDictionary *record_for(id b){NSNumber *key=cbid(b);NSMutableDictionary *r=S().records[key];if(!r){r=[NSMutableDictionary dictionaryWithDictionary:@{@"id":key,@"step":@(S().step),@"dispatches":[NSMutableArray array],@"completed":@NO,@"host_encoder_span_ns":@0}];S().records[key]=r;}return r;}
static void dispatch_record(id e,NSString *kind,MTLSize grid,MTLSize group,uint64_t host){if(!S().armed)return;id b=objc_getAssociatedObject(e,EncoderCB);if(!b)return;NSDictionary *d=@{@"kernel":objc_getAssociatedObject(e,KernelKey)?:@"unknown",@"kind":kind,@"grid":@[@(grid.width),@(grid.height),@(grid.depth)],@"threadgroup":@[@(group.width),@(group.height),@(group.depth)],@"host_dispatch_api_ns":@(host),@"bindings":[[objc_getAssociatedObject(e,ArgsKey) copy]autorelease]?:@{}};std::lock_guard<std::mutex> l(S().lock);[[record_for(b) objectForKey:@"dispatches"]addObject:d];}
static void(*origThreads)(id,SEL,MTLSize,MTLSize);
static void threads(id e,SEL c,MTLSize g,MTLSize t){uint64_t start=now_ns();origThreads(e,c,g,t);dispatch_record(e,@"threads",g,t,now_ns()-start);}
static void(*origGroups)(id,SEL,MTLSize,MTLSize);
static void groups(id e,SEL c,MTLSize g,MTLSize t){uint64_t start=now_ns();origGroups(e,c,g,t);dispatch_record(e,@"threadgroups",g,t,now_ns()-start);}
static void(*origEnd)(id,SEL);
static void endEncoding(id e,SEL c){uint64_t end=now_ns();origEnd(e,c);if(!S().armed)return;id b=objc_getAssociatedObject(e,EncoderCB);NSNumber *start=objc_getAssociatedObject(e,SpanKey);if(b && start){std::lock_guard<std::mutex> l(S().lock);NSMutableDictionary *r=record_for(b);r[@"host_encoder_span_ns"]=@([r[@"host_encoder_span_ns"]unsignedLongLongValue]+end-[start unsignedLongLongValue]);}}
static void(*origCommit)(id,SEL);
static void commit(id b,SEL c){NSNumber *key=cbid(b);BOOL track=NO;{std::lock_guard<std::mutex> l(S().lock);track=S().records[key]!=nil;}if(track){[b addCompletedHandler:^(id<MTLCommandBuffer> done){std::lock_guard<std::mutex> l(S().lock);NSMutableDictionary *r=S().records[key];r[@"completed"]=@YES;r[@"status"]=@((NSInteger)done.status);r[@"gpu_start_s"]=@(done.GPUStartTime);r[@"gpu_end_s"]=@(done.GPUEndTime);r[@"error"]=done.error.localizedDescription?:[NSNull null];}];}uint64_t start=now_ns();origCommit(b,c);if(track){std::lock_guard<std::mutex> l(S().lock);S().records[key][@"host_commit_api_ns"]=@(now_ns()-start);}}
static void hook(Class cls,SEL sel,IMP replacement,IMP *old){Method m=class_getInstanceMethod(cls,sel);if(!m)return;*old=method_getImplementation(m);class_replaceMethod(cls,sel,replacement,method_getTypeEncoding(m));[S().hooks addObject:[NSString stringWithFormat:@"%@: %@",NSStringFromClass(cls),NSStringFromSelector(sel)]];}
__attribute__((constructor))static void initialize(){@autoreleasepool{id<MTLDevice>d=MTLCreateSystemDefaultDevice();if(!d){S().error=@"Metal device unavailable";return;}id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>b=[q commandBuffer];id<MTLComputeCommandEncoder>e=[b computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];Class dc=[d class],bc=[b class],ec=[e class];[e endEncoding];
 hook(dc,@selector(newComputePipelineStateWithFunction:error:),(IMP)pso,(IMP*)&origPso);
 hook(dc,@selector(newComputePipelineStateWithFunction:options:reflection:error:),(IMP)psoOptions,(IMP*)&origPsoOptions);
 hook(dc,@selector(newComputePipelineStateWithDescriptor:options:reflection:error:),(IMP)psoDesc,(IMP*)&origPsoDesc);
 hook(bc,@selector(computeCommandEncoder),(IMP)encoder,(IMP*)&origEncoder);
 hook(bc,@selector(computeCommandEncoderWithDispatchType:),(IMP)encoderType,(IMP*)&origEncoderType);
 hook(bc,@selector(computeCommandEncoderWithDescriptor:),(IMP)encoderDesc,(IMP*)&origEncoderDesc);
 hook(bc,@selector(commit),(IMP)commit,(IMP*)&origCommit);
 hook(ec,@selector(setComputePipelineState:),(IMP)setPso,(IMP*)&origSetPso);
 hook(ec,@selector(setBytes:length:atIndex:),(IMP)setBytes,(IMP*)&origBytes);
 hook(ec,@selector(setBuffer:offset:atIndex:),(IMP)setBuffer,(IMP*)&origBuffer);
 hook(ec,@selector(dispatchThreads:threadsPerThreadgroup:),(IMP)threads,(IMP*)&origThreads);
 hook(ec,@selector(dispatchThreadgroups:threadsPerThreadgroup:),(IMP)groups,(IMP*)&origGroups);
 hook(ec,@selector(endEncoding),(IMP)endEncoding,(IMP*)&origEnd);
 [q release];
}}
extern "C" int v26_begin(){std::lock_guard<std::mutex> l(S().lock);if(S().error)return 1;[S().records removeAllObjects];S().step=-1;S().armed=true;return 0;}
extern "C" void v26_step(int step){std::lock_guard<std::mutex> l(S().lock);S().step=step;}
extern "C" int v26_end(const char *path){@autoreleasepool{std::lock_guard<std::mutex> l(S().lock);S().armed=false;NSArray *rows=[[S().records allValues]sortedArrayUsingComparator:^NSComparisonResult(id a,id b){return [a[@"id"]compare:b[@"id"]];}];NSDictionary *out=@{@"scope":@"instrumented command-buffer GPU envelope; no per-dispatch GPU timestamps",@"dispatch_boundary_supported":@NO,@"rows":rows,@"hooks":S().hooks,@"error":S().error?:[NSNull null]};NSError *e=nil;NSData *data=[NSJSONSerialization dataWithJSONObject:out options:NSJSONWritingPrettyPrinted error:&e];if(!data || ![data writeToFile:[NSString stringWithUTF8String:path]options:NSDataWritingAtomic error:&e])return 2;return 0;}}
