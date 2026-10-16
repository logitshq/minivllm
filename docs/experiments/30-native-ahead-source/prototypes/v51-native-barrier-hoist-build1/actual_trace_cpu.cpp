#include "dependency_plan.h"
#include <fstream>
#include <iostream>
using namespace v51;
int main(int argc,char**argv){if(argc!=2&&argc!=3)return 2;std::ifstream f(argv[1]);std::vector<Entry>trace;char type;size_t command,nr,nw;unsigned origin;uintptr_t id;
 while(f>>type>>command){if(type=='D'){f>>nr>>nw;Entry e{Type::Dispatch,Origin::Dispatch,command,{},{}};for(size_t i=0;i<nr;++i){f>>id;e.reads.insert(id);}for(size_t i=0;i<nw;++i){f>>id;e.writes.insert(id);}trace.push_back(e);}else{f>>origin;trace.push_back({type=='C'?Type::CandidateBuffersScope:Type::Forced,(Origin)origin,command,{},{}});}if(!f)throw std::runtime_error("incomplete CPU trace");}
 auto p=plan(trace);if(!p.verified||p.rows.size()!=1769||p.candidates!=1059||p.forcedLocations!=83||p.proofConflictPairs!=27363)throw std::runtime_error("actual V42 trace identity mismatch");
 std::cout<<"{\"status\":\"PASS_CPU_EXISTING_V42_TRACE_HOIST\",\"dispatches\":"<<p.rows.size()<<",\"pre_hoist_barriers\":"<<p.preHoistBarriers<<",\"final_barriers\":"<<p.barriers<<",\"hoist_attempts\":"<<p.hoistAttempts<<",\"hoist_accepted\":"<<p.hoists.size()<<",\"hoist_rejected\":"<<p.hoistRejected<<",\"merged_existing\":"<<p.hoistMerged<<",\"candidate_hoisted\":"<<p.candidatesHoisted<<",\"candidate_retained\":"<<p.candidatesRetained<<",\"all_trial_edge_checks\":"<<p.hoistEdgeChecks<<",\"all_pair_conflicting_edges\":"<<p.proofConflictPairs<<",\"forced_locations\":"<<p.forcedLocations<<",\"both_bank_GPU_engagement\":\"UNTESTED_CPU_ONLY\",\"GPU_used\":false}\n";
 if(argc==3){std::ofstream o(argv[2]);o<<"{\"scope\":\"CPU-only replay of existing V42 actual metadata, not fresh GPU engagement\",\"accepted\":[";bool first=true;for(auto&m:p.hoists){if(!first)o<<",";first=false;o<<"{\"from\":"<<m.from<<",\"to\":"<<m.to<<",\"all_edges_checked\":"<<m.edgeChecks<<"}";}o<<"],\"rejected\":[";first=true;for(auto&m:p.rejectedHoists){if(!first)o<<",";first=false;o<<"{\"from\":"<<m.from<<",\"to\":"<<m.to<<",\"uncovered_edge_from\":"<<m.uncoveredFrom<<",\"uncovered_edge_to\":"<<m.uncoveredTo<<",\"all_edges_checked\":"<<m.edgeChecks<<",\"reason\":\"would remove ordering of previous dispatch versus a later consumer\"}";}o<<"]}\n";}

}
