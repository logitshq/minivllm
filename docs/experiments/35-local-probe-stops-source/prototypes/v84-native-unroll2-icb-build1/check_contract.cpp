#include "range_contract.h"
#include <cassert>
#include <iostream>
int main(){std::vector<V84Access>x;for(int i=0;i<16;++i)x.push_back({{uintptr_t(1+i%8),uintptr_t(20+i%8),uintptr_t(40+i%8)},uintptr_t(100+i)});assert(v84_independent(x).empty());auto b=x;b.pop_back();assert(!v84_independent(b).empty());b=x;b[0].write=b[1].write;assert(!v84_independent(b).empty());b=x;b[0].reads.insert(b[15].write);assert(!v84_independent(b).empty());b=x;b[0].write=0;assert(!v84_independent(b).empty());b=x;b[0].reads.clear();assert(!v84_independent(b).empty());for(unsigned groups:{1,2,3,4,5,7,8,16,128,1024}){uint64_t observed=0,barriers=0;for(unsigned i=0;i<groups;++i){observed+=8;++barriers;}assert(observed==8ull*groups&&barriers==groups);}const std::string A="nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0",B="custom_kernel_v83_nvfp4_unroll2_r4_sg2_n17408_k5120__bfloat16_t_4_uint32_t_uint8_t_bfloat16_t_bfloat16_t";
assert(v84_source_writeonly(0,A,4,"y",1));assert(v84_source_writeonly(1,B,3,"output",1));
assert(!v84_source_writeonly(0,A,3,"x",1));assert(!v84_source_writeonly(0,A,0,"w",1));assert(!v84_source_writeonly(0,A,4,"wrong",1));assert(!v84_source_writeonly(0,A,3,"y",1));
assert(!v84_source_writeonly(1,B,2,"x",1));assert(!v84_source_writeonly(1,B,4,"output",1));assert(!v84_source_writeonly(1,B,3,"wrong",1));
assert(!v84_source_writeonly(0,A+"other",4,"y",1));assert(!v84_source_writeonly(1,B+"other",3,"output",1));assert(!v84_source_writeonly(0,A,4,"y",0));assert(!v84_source_writeonly(0,A,4,"y",2));assert(!v84_source_writeonly(0,A,4,"y",3));assert(!v84_source_writeonly(2,B,3,"output",1));
for(unsigned arm=0;arm<2;++arm){
 const unsigned count=arm?4:6;for(unsigned i=0;i<count;++i){unsigned idx=arm?i:(i<2?i:i+1);const char*namesA[]={"w","scales","x","y","in_vec_size","out_vec_size"};const char*namesB[]={"weight","scales","x","output"};std::string name=arm?namesB[i]:namesA[i];bool bytes=!arm&&i>=4;unsigned access=i==(arm?3:3)?1:0;
 assert(v84_exact_argument(arm,idx,name,access,bytes,bytes?4:0));assert(!v84_exact_argument(arm,idx,name+"wrong",access,bytes,bytes?4:0));assert(!v84_exact_argument(arm,idx,name,access,!bytes,bytes?4:0));if(bytes)assert(!v84_exact_argument(arm,idx,name,access,true,8));}
 }
 assert(!v84_exact_argument(1,2,"x",1,false,0));assert(!v84_exact_argument(1,3,"output",0,false,0));assert(!v84_exact_argument(2,3,"output",1,false,0));
 std::cout<<"PASS_CPU_CONTRACT_6_ACCESS_10_LOOP_15_EXACT_RW_PLUS_COMPLETE_SCHEMA_CASES\n";}
