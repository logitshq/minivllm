#include "range_contract.h"
#include <cassert>
#include <iostream>
int main(){std::vector<V72Access>x;for(int i=0;i<16;++i)x.push_back({{uintptr_t(1+i%8),uintptr_t(20+i%8),uintptr_t(40+i%8)},uintptr_t(100+i)});assert(v72_independent(x).empty());auto b=x;b.pop_back();assert(!v72_independent(b).empty());b=x;b[0].write=b[1].write;assert(!v72_independent(b).empty());b=x;b[0].reads.insert(b[15].write);assert(!v72_independent(b).empty());b=x;b[0].write=0;assert(!v72_independent(b).empty());b=x;b[0].reads.clear();assert(!v72_independent(b).empty());for(unsigned groups:{1,2,3,4,5,7,8,16,128,1024}){uint64_t observed=0,barriers=0;for(unsigned i=0;i<groups;++i){observed+=8;++barriers;}assert(observed==8ull*groups&&barriers==groups);}const std::string A="nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0",B="custom_kernel_v69_nvfp4_r8_sg2_n17408_k5120__bfloat16_t_8_uint32_t_uint8_t_bfloat16_t_bfloat16_t";
assert(v72_source_writeonly(0,A,4,"y",1));assert(v72_source_writeonly(1,B,3,"output",1));
assert(!v72_source_writeonly(0,A,3,"x",1));assert(!v72_source_writeonly(0,A,0,"w",1));assert(!v72_source_writeonly(0,A,4,"wrong",1));assert(!v72_source_writeonly(0,A,3,"y",1));
assert(!v72_source_writeonly(1,B,2,"x",1));assert(!v72_source_writeonly(1,B,4,"output",1));assert(!v72_source_writeonly(1,B,3,"wrong",1));
assert(!v72_source_writeonly(0,A+"other",4,"y",1));assert(!v72_source_writeonly(1,B+"other",3,"output",1));assert(!v72_source_writeonly(0,A,4,"y",0));assert(!v72_source_writeonly(0,A,4,"y",2));assert(!v72_source_writeonly(0,A,4,"y",3));assert(!v72_source_writeonly(2,B,3,"output",1));
std::cout<<"PASS_CPU_CONTRACT_6_ACCESS_10_LOOP_15_EXACT_RW_CASES\n";}
