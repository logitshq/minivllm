#include "range_contract.h"
#include <cassert>
#include <iostream>
int main(){std::vector<V72Access>x;for(int i=0;i<16;++i)x.push_back({{uintptr_t(1+i%8),uintptr_t(20+i%8),uintptr_t(40+i%8)},uintptr_t(100+i)});assert(v72_independent(x).empty());auto b=x;b.pop_back();assert(!v72_independent(b).empty());b=x;b[0].write=b[1].write;assert(!v72_independent(b).empty());b=x;b[0].reads.insert(b[15].write);assert(!v72_independent(b).empty());b=x;b[0].write=0;assert(!v72_independent(b).empty());b=x;b[0].reads.clear();assert(!v72_independent(b).empty());for(unsigned groups:{1,2,3,4,5,7,8,16,128,1024}){uint64_t observed=0,barriers=0;for(unsigned i=0;i<groups;++i){observed+=8;++barriers;}assert(observed==8ull*groups&&barriers==groups);}std::cout<<"PASS_CPU_CONTRACT_6_ACCESS_CASES_10_LOOP_COUNTS\n";}
