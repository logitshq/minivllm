#include "feedback_plan.h"
#include <cstring>
#include <iostream>
#include <random>
using namespace v57;
static void need(bool b){if(!b)throw std::runtime_error("CPU feedback contract assertion");}
int main(){size_t cases=0;auto good=[&](std::vector<Range>i,std::vector<Range>o,size_t n){++cases;return validate(i,o,n);};auto bad=[&](std::vector<Range>i,std::vector<Range>o,size_t n){++cases;bool stopped=false;try{validate(i,o,n);}catch(...){stopped=true;}need(stopped);};
 auto p=good({{0,4},{8,4}},{{16,4},{24,4}},32);need(p.inputOutputPairs==4&&p.inputPairs==1);
 bad({{0,4}},{{0,4}},32); // identity also rejects: exact no-alias eligibility
 bad({{0,8}},{{4,8}},32);bad({{0,8},{4,8}},{{16,8}},32);
 bad({{0,4},{8,4}},{{8,4},{0,4}},32); // cycle
 bad({{SIZE_MAX-1,4}},{{0,4}},SIZE_MAX);bad({{0,0}},{{4,4}},32);bad({},{{4,4}},32);
 // Direct ordered copies must equal simultaneous assignment and preserve ALL
 // output sources after copying, including an unlinked logits-like range.
 std::mt19937 rng(57);for(unsigned trial=0;trial<1000;++trial){std::vector<unsigned char>a(1024);for(auto&x:a)x=(unsigned char)rng();auto original=a;std::vector<Range>inputs,outputs;for(size_t i=0;i<8;++i){inputs.push_back({i*32,16});outputs.push_back({512+i*32,16});}outputs.push_back({1000,16});good(inputs,outputs,a.size());for(size_t i=0;i<inputs.size();++i)memcpy(a.data()+inputs[i].offset,a.data()+outputs[i].offset,inputs[i].size);for(size_t i=0;i<inputs.size();++i)need(!memcmp(a.data()+inputs[i].offset,original.data()+outputs[i].offset,inputs[i].size));for(auto o:outputs)need(!memcmp(a.data()+o.offset,original.data()+o.offset,o.size));}
 std::cout<<"{\"status\":\"PASS_CPU_DIRECT_FEEDBACK_CONTRACT\",\"cases\":"<<cases<<",\"random_simultaneous_assignments\":1000,\"all_output_sources_preserved\":true,\"alias_identity_cycle_overflow_rejected\":true,\"GPU_used\":false}\n";
}
