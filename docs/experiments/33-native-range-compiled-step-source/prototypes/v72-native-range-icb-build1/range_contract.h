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
