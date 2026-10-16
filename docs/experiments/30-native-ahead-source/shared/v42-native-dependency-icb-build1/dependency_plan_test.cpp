#include "dependency_plan.h"
#include <iostream>
#include <random>
using namespace v42;
static Entry d(IDs r={},IDs w={}){return {Type::Dispatch,Origin::Dispatch,0,r,w};}
static Entry scope(){return {Type::CandidateBuffersScope,Origin::Scope,0,{},{}};}
static Entry force(Origin o){return {Type::Forced,o,0,{},{}};}
static void need(bool b){if(!b)throw std::runtime_error("CPU test assertion");}
int main(){unsigned cases=0;auto test=[&](std::vector<Entry>t){++cases;return plan(t);};
 for(auto e:std::vector<Entry>{d({1},{}),d({},{1}),d({1},{1})}){auto p=test({d({1},{1}),e});need(p.rows[1].before);}
 auto p=test({d({9},{1}),scope(),d({9},{2}),d({1,2},{3})});need(!p.rows[1].before&&p.rows[2].before&&p.candidatesDeferred==1&&p.removedLocations==1&&p.insertedLocations==2);
 for(auto o:{Origin::EncoderEnd,Origin::CBCommit,Origin::Resources,Origin::FenceUpdate,Origin::FenceWait,Origin::EventSignal,Origin::EventWait,Origin::Scope}){auto q=test({d({}, {1}),force(o),d({}, {2})});need(q.rows[1].before&&q.forcedEntries==1);}
 auto tail=test({d({}, {1}),scope()});need(tail.trailing&&tail.candidatesCompletionCovered==1);
 auto missing=test({d({}, {1}),d({1},{2})});need(missing.insertedLocations==2&&missing.proofRAW==1);
 auto ro=test({d({1},{}),scope(),d({1},{})});need(ro.barriers==1&&!ro.rows[1].before&&ro.candidatesCompletionCovered==1);
 bool rejected=false;try{plan({force(Origin::Unknown)});}catch(...){rejected=true;}need(rejected);
 rejected=false;try{plan({{Type::CandidateBuffersScope,Origin::EncoderEnd,0,{},{}}});}catch(...){rejected=true;}need(rejected);
 auto corrupt=missing;corrupt.rows[1].before=false;rejected=false;try{verify(corrupt);}catch(...){rejected=true;}need(rejected);
 std::mt19937 rng(42);for(unsigned trial=0;trial<2000;++trial){std::vector<Entry>t;for(unsigned i=0;i<1+rng()%35;++i){if(rng()%4==0)t.push_back(scope());if(rng()%13==0)t.push_back(force(Origin::CBCommit));IDs r,w;for(uintptr_t b=1;b<=7;++b){unsigned a=rng()%8;if(a&1)r.insert(b);if(a&2)w.insert(b);}t.push_back(d(r,w));}auto a=test(t);auto swapped=t;for(auto&e:swapped){IDs r,w;for(auto x:e.reads)r.insert(x==1?2:x==2?1:x);for(auto x:e.writes)w.insert(x==1?2:x==2?1:x);e.reads=r;e.writes=w;}auto b=plan(swapped);need(a.verified&&b.verified&&sameBarrierPlan(a,b));}
 std::cout<<"{\"status\":\"PASS_CPU_DEPENDENCY_PLAN\",\"cases\":"<<cases<<",\"seed\":42,\"random_traces\":2000,\"independent_all_conflict_edges\":true,\"GPU_used\":false}\n";
}
