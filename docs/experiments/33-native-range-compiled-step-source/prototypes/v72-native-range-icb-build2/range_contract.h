#pragma once
#include <vector>
#include <set>
#include <cstdint>
#include <string>
struct V72Access { std::set<uintptr_t> reads; uintptr_t write=0; };
inline std::string v72_independent(const std::vector<V72Access>& ops) {
 if(ops.size()!=16)return "requires exactly16 dispatches";
 std::set<uintptr_t> writes,reads;
 for(auto& op:ops){if(!op.write||!writes.insert(op.write).second)return "output allocations not distinct";if(op.reads.empty())return "missing readonly operands";reads.insert(op.reads.begin(),op.reads.end());}
 for(auto w:writes)if(reads.count(w))return "output allocation overlaps read operand";
 return {};
}

inline bool v72_exact_function(unsigned arm,const std::string&fn){
 return arm==0 ? fn=="nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0" : arm==1 && fn=="custom_kernel_v69_nvfp4_r8_sg2_n17408_k5120__bfloat16_t_8_uint32_t_uint8_t_bfloat16_t_bfloat16_t";
}
// Reflection remains ReadWrite(1). This ONLY licenses source-store-only output.
inline bool v72_source_writeonly(unsigned arm,const std::string&fn,unsigned index,const std::string&name,unsigned access){
 return access==1 && v72_exact_function(arm,fn) && (arm==0 ? index==4 && name=="y" : index==3 && name=="output");
}
