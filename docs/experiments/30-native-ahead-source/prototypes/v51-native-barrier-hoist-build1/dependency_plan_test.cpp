#include "dependency_plan.h"
#include <iostream>
#include <random>
using namespace v51;
static Entry d(IDs r={},IDs w={}){return {Type::Dispatch,Origin::Dispatch,0,r,w};}
static Entry scope(){return {Type::CandidateBuffersScope,Origin::Scope,0,{},{}};}
static Entry force(Origin o){return {Type::Forced,o,0,{},{}};}
static void need(bool b){if(!b)throw std::runtime_error("CPU test assertion");}
int main(){unsigned cases=0,randomMoves=0;auto test=[&](std::vector<Entry>t){++cases;auto p=plan(t);need(p.candidates==p.candidatesRetained+p.candidatesDeferred+p.candidatesHoisted+p.candidatesCompletionCovered);need(p.hoistAttempts==p.hoists.size()+p.hoistRejected);return p;};
 auto move=test({d({}, {1}),d({}, {2}),scope(),d({1},{3})});need(move.rows[0].before&&move.rows[1].before&&!move.rows[2].before&&move.hoists.size()==1&&move.candidatesHoisted==1&&move.preHoistBarriers==move.barriers&&move.hoistEdgeChecks==move.proofConflictPairs);
 for(auto e:std::vector<Entry>{d({1},{}),d({},{1}),d({1},{1})}){auto p=test({d({1},{1}),scope(),e});need(p.rows[1].before&&p.hoists.empty());}
 for(auto o:{Origin::EncoderEnd,Origin::CBCommit,Origin::Resources,Origin::FenceUpdate,Origin::FenceWait,Origin::EventSignal,Origin::EventWait,Origin::Scope}){auto q=test({d({}, {1}),d({}, {2}),scope(),force(o),d({1},{3})});need(q.rows[2].before&&q.hoists.empty()&&q.forcedEntries==1);}
 auto chain=test({d({}, {1}),d({}, {2}),scope(),d({1},{3}),scope(),d({2},{4})});need(chain.hoists.empty()&&chain.hoistRejected==1&&chain.rejectedHoists[0].uncoveredFrom==1&&chain.rejectedHoists[0].uncoveredTo==3);
 auto two=test({d({}, {1}),d({}, {2}),scope(),d({1},{3}),d({}, {4}),scope(),d({3},{5})});need(two.hoists.size()==2&&two.candidatesHoisted==2);
 auto tail=test({d({}, {1}),scope()});need(tail.trailing&&tail.candidatesCompletionCovered==1);
 auto missing=test({d({}, {1}),d({1},{2})});need(missing.insertedLocations==2&&missing.proofRAW==1&&missing.hoists.empty());
 bool rejected=false;try{plan({force(Origin::Unknown)});}catch(...){rejected=true;}need(rejected);
 rejected=false;try{plan({{Type::CandidateBuffersScope,Origin::EncoderEnd,0,{},{}}});}catch(...){rejected=true;}need(rejected);
 auto corrupt=missing;auto edges=allConflictEdges(corrupt);corrupt.rows[1].before=false;need(!coversEveryEdge(corrupt,edges));rejected=false;try{verify(corrupt);}catch(...){rejected=true;}need(rejected);
 auto forced=move;forced.rows[1].forced=true;forced.rows[1].before=false;need(!coversEveryEdge(forced,allConflictEdges(forced)));
 std::mt19937 rng(51);for(unsigned trial=0;trial<2000;++trial){std::vector<Entry>t;for(unsigned i=0;i<1+rng()%35;++i){if(rng()%4==0)t.push_back(scope());if(rng()%13==0)t.push_back(force(Origin::CBCommit));IDs r,w;for(uintptr_t b=1;b<=7;++b){unsigned a=rng()%24;if(a==0||a==2)r.insert(b);if(a==1||a==2)w.insert(b);}t.push_back(d(r,w));}auto a=test(t);randomMoves+=a.hoists.size();auto swapped=t;for(auto&e:swapped){IDs r,w;for(auto x:e.reads)r.insert(x==1?2:x==2?1:x);for(auto x:e.writes)w.insert(x==1?2:x==2?1:x);e.reads=r;e.writes=w;}auto b=plan(swapped);need(a.verified&&b.verified&&sameBarrierPlan(a,b));}
 std::cout<<"{\"status\":\"PASS_CPU_PREVIOUS_OP_HOIST\",\"cases\":"<<cases<<",\"seed\":51,\"random_traces\":2000,\"random_moves\":"<<randomMoves<<",\"every_trial_all_edges\":true,\"original_final_verifier\":true,\"GPU_used\":false}\n";
}
