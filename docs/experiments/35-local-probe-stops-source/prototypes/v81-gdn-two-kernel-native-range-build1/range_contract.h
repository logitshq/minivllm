#pragma once
#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <vector>
namespace v81 {
constexpr unsigned banks=15, perArm=30, commands=60, arrays=10;
inline const char* function(unsigned arm,unsigned stage){
 static const char* names[2][2]={
  {"custom_kernel_gdn_activated_norm_reuse_prework_v44_build1_bfloat16_t_float_float_float_float","custom_kernel_gdn_norm_reuse_cached_state_v40_build1_float_float_float_float_float_float_float_float_float"},
  {"custom_kernel_gdn_activated_scaled_ready_prework_v78_build1_bfloat16_t_float_float_float_float","custom_kernel_gdn_scaled_ready_cached_state_v78_build1_float_float_float_float_float_float_float_float_float"}};
 return arm<2&&stage<2?names[arm][stage]:"";
}
inline int arrayIndex(unsigned stage,unsigned index,const std::string&name){
 if(stage==0){static const char* names[]={"activated","q_ready","k_ready","v_ready","scales"};if(index>4||name!=names[index])return -1;return index==0?0:int(index+3);}
 if(stage!=1)return -1;
 static const char* names[]={"q_ready","k_ready","v","scales","g","beta","S","S_shape","o","S_new"};
 static const int map[]={4,5,6,7,1,2,3,-2,8,9};
 return index<10&&name==names[index]?map[index]:-1;
}
inline bool output(unsigned stage,unsigned index,const std::string&name){int a=arrayIndex(stage,index,name);return stage==0?a>=4&&a<=7:a>=8;}
inline bool storeOnly(unsigned arm,unsigned stage,const std::string&fn,unsigned index,const std::string&name,unsigned access){return arm<2&&stage<2&&fn==function(arm,stage)&&access==1&&output(stage,index,name);}
inline bool geometry(unsigned stage,unsigned kind,std::array<uint64_t,3>grid,std::array<uint64_t,3>tg){
 if(kind!=1&&kind!=2)return false;
 const std::array<uint64_t,3> expected=stage==0?std::array<uint64_t,3>{8192,1,1}:std::array<uint64_t,3>{128,192,1};
 const std::array<uint64_t,3> group=stage==0?std::array<uint64_t,3>{128,1,1}:std::array<uint64_t,3>{32,4,1};
 if(tg!=group)return false;
 auto total=grid;if(kind==2)for(unsigned i=0;i<3;++i)total[i]*=tg[i];return total==expected;
}
struct Access {std::set<uintptr_t> reads,writes;};
using Bank=std::array<uintptr_t,arrays>;
inline std::string dependencies(const std::vector<Access>&ops,const std::vector<Bank>&slots){
 if(ops.size()!=commands||slots.size()!=30)return "requires60 commands/30 arm-bank slots";
 std::set<uintptr_t>inputs,mutables;
 for(unsigned i=0;i<30;++i){
  for(unsigned a=0;a<4;++a){if(!slots[i][a])return "missing immutable input";if(i<15&&!inputs.insert(slots[i][a]).second)return "immutable bank allocations alias";if(i>=15&&slots[i][a]!=slots[i-15][a])return "A/B immutable inputs differ";}
  for(unsigned a=4;a<10;++a)if(!slots[i][a]||!mutables.insert(slots[i][a]).second)return "mutable allocations not exclusive";
 }
 for(auto id:mutables)if(inputs.count(id))return "mutable allocation aliases immutable input";
 for(unsigned i=0;i<30;++i){auto&b=slots[i];auto&pre=ops[i*2];auto&state=ops[i*2+1];
  if(pre.writes!=std::set<uintptr_t>{b[4],b[5],b[6],b[7]}||state.writes!=std::set<uintptr_t>{b[8],b[9]})return "output writer contract mismatch";
  if(pre.reads!=std::set<uintptr_t>{b[0]})return "prework input contract mismatch";
  for(unsigned a:{1u,2u,3u,4u,5u,6u})if(!state.reads.count(b[a]))return "missing state input or producer edge";
  if(i<15&&!state.reads.count(b[7]))return "A44 scales input missing";
  std::set<uintptr_t>expected={b[1],b[2],b[3],b[4],b[5],b[6]};if(i<15||state.reads.count(b[7]))expected.insert(b[7]);
  if(state.reads!=expected)return "unknown or extra state read allocation";
  for(auto id:state.reads)if(mutables.count(id)&&id!=b[4]&&id!=b[5]&&id!=b[6]&&id!=b[7])return "unexpected mutable consumer edge";
  for(auto id:state.reads)if(inputs.count(id)&&id!=b[1]&&id!=b[2]&&id!=b[3])return "state immutable bank mismatch";
 }
 return {};
}
inline bool groups(unsigned arm,unsigned count){return arm<2&&count>=1&&count<=8192;}
}
