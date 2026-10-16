#pragma once
// Exact explicit feedback ranges only. This does not infer shader byte extents.
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
namespace v57 {
struct Range {size_t offset,size;};
struct Proof {size_t inputOutputPairs=0,inputPairs=0;};
inline bool overlaps(Range a,Range b){return a.offset<b.offset+b.size&&b.offset<a.offset+a.size;}
inline void bounds(Range a,size_t capacity){if(!a.size||a.offset>capacity||a.size>capacity-a.offset)throw std::runtime_error("unknown/overflowing/out-of-arena registered range");}
inline Proof validate(const std::vector<Range>&inputs,const std::vector<Range>&outputs,size_t capacity){
 if(inputs.empty()||outputs.empty())throw std::runtime_error("empty feedback range set");
 for(auto a:inputs)bounds(a,capacity);for(auto a:outputs)bounds(a,capacity);
 Proof p;
 for(size_t i=0;i<inputs.size();++i)for(size_t j=i+1;j<inputs.size();++j){++p.inputPairs;if(overlaps(inputs[i],inputs[j]))throw std::runtime_error("registered input destinations overlap");}
 for(auto a:inputs)for(auto b:outputs){++p.inputOutputPairs;if(overlaps(a,b))throw std::runtime_error("registered output source overlaps an input destination; fast mode rejected");}
 return p;
}
}
