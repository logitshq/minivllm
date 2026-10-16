#pragma once
#include <vector>
#include <set>
#include <cstdint>
#include <string>
struct V80Access { std::set<uintptr_t> reads; uintptr_t write=0; };
inline std::string v80_independent(const std::vector<V80Access>& ops) {
 if(ops.size()!=16)return "requires exactly16 dispatches";
 std::set<uintptr_t> writes,reads;
 for(auto& op:ops){if(!op.write||!writes.insert(op.write).second)return "output allocations not distinct";if(op.reads.empty())return "missing readonly operands";reads.insert(op.reads.begin(),op.reads.end());}
 for(auto w:writes)if(reads.count(w))return "output allocation overlaps read operand";
 return {};
}

inline bool v80_exact_function(unsigned arm,const std::string&fn){
 return arm==0 ? fn=="nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0" : arm==1 && fn=="custom_kernel_v79_nvfp4_half2_r4_sg2_n17408_k5120__bfloat16_t_4_uint32_t_uint8_t_bfloat16_t_bfloat16_t";
}
// Reflection remains ReadWrite(1). This ONLY licenses source-store-only output.
inline bool v80_source_writeonly(unsigned arm,const std::string&fn,unsigned index,const std::string&name,unsigned access){
 return access==1 && v80_exact_function(arm,fn) && (arm==0 ? index==4 && name=="y" : index==3 && name=="output");
}

// Exact source-derived schema. B symbol is EXPECTED, not a claim of prior observation.
inline bool v80_exact_argument(unsigned arm,unsigned index,const std::string&name,unsigned access,bool bytes,size_t n){
 if(arm==0){
  if(index==0)return name=="w"&&access==0&&!bytes;
  if(index==1)return name=="scales"&&access==0&&!bytes;
  if(index==3)return name=="x"&&access==0&&!bytes;
  if(index==4)return name=="y"&&(access==1||access==2)&&!bytes;
  if(index==5)return name=="in_vec_size"&&access==0&&bytes&&n==4;
  if(index==6)return name=="out_vec_size"&&access==0&&bytes&&n==4;
 }else if(arm==1){
  if(index==0)return name=="weight"&&access==0&&!bytes;
  if(index==1)return name=="scales"&&access==0&&!bytes;
  if(index==2)return name=="x"&&access==0&&!bytes;
  if(index==3)return name=="output"&&(access==1||access==2)&&!bytes;
 }
 return false;
}
