#pragma once
// CPU-only planner. IDs mean complete original MTLBuffer allocations, never views.
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace v51 {
using IDs=std::set<uintptr_t>;
enum class Type {Dispatch, CandidateBuffersScope, Forced};
enum class Origin {Unknown, Dispatch, Scope, EncoderEnd, CBCommit, Resources, FenceUpdate, FenceWait, EventSignal, EventWait};
inline const char*name(Origin o){switch(o){case Origin::Dispatch:return "dispatch";case Origin::Scope:return "memoryBarrierWithScope";case Origin::EncoderEnd:return "encoder_end";case Origin::CBCommit:return "CB_commit";case Origin::Resources:return "memoryBarrierWithResources";case Origin::FenceUpdate:return "fence_update";case Origin::FenceWait:return "fence_wait";case Origin::EventSignal:return "event_signal";case Origin::EventWait:return "event_wait";default:return "unknown";}}
struct Entry {Type type;Origin origin;size_t command;IDs reads,writes;};
struct Row {size_t command=0;IDs reads,writes;bool before=false,original=false,forced=false;size_t raw=0,war=0,waw=0;};
struct Hoist {size_t from=0,to=0,edgeChecks=0;bool merged=false;};
struct RejectedHoist {size_t from=0,to=0,edgeChecks=0,uncoveredFrom=0,uncoveredTo=0;};
struct Plan {std::vector<RejectedHoist>rejectedHoists;std::vector<Row> rows;std::map<std::string,size_t> origins;std::vector<Hoist>hoists;size_t preHoistBarriers=0,hoistAttempts=0,hoistRejected=0,hoistEdgeChecks=0,hoistMerged=0,candidatesHoisted=0;size_t originalEntries=0,candidates=0,forcedEntries=0,originalLocations=0,forcedLocations=0,barriers=0,insertedLocations=0,removedLocations=0,candidatesRetained=0,candidatesDeferred=0,candidatesCompletionCovered=0,raw=0,war=0,waw=0,proofRAW=0,proofWAR=0,proofWAW=0,proofConflictPairs=0;bool trailing=false,verified=false;};
inline size_t intersection(const IDs&a,const IDs&b){size_t n=0;auto i=a.begin(),j=b.begin();while(i!=a.end()&&j!=b.end()){if(*i<*j)++i;else if(*j<*i)++j;else{++n;++i;++j;}}return n;}
// Independent proof checks ALL earlier/later allocation conflict edges, not just
// the frontier used to generate the plan. A barrier must separate every edge.
inline void verify(Plan&p){p.proofRAW=p.proofWAR=p.proofWAW=p.proofConflictPairs=0;std::vector<size_t>prefix(1,0);for(auto&r:p.rows)prefix.push_back(prefix.back()+r.before);for(size_t j=0;j<p.rows.size();++j){if((j==0||p.rows[j].forced)&&!p.rows[j].before)throw std::runtime_error("missing first/forced barrier");for(size_t i=0;i<j;++i){auto&a=p.rows[i];auto&b=p.rows[j];size_t raw=intersection(a.writes,b.reads),war=intersection(a.reads,b.writes),waw=intersection(a.writes,b.writes);p.proofRAW+=raw;p.proofWAR+=war;p.proofWAW+=waw;if(raw||war||waw){++p.proofConflictPairs;if(prefix[j+1]==prefix[i+1])throw std::runtime_error("uncovered RAW/WAR/WAW conflict edge");}}}p.verified=true;}
struct ConflictEdge {size_t from,to;};
inline std::vector<ConflictEdge> allConflictEdges(const Plan&p){
 std::vector<ConflictEdge>edges;
 for(size_t j=0;j<p.rows.size();++j)for(size_t i=0;i<j;++i){auto&a=p.rows[i];auto&b=p.rows[j];if(intersection(a.writes,b.reads)||intersection(a.reads,b.writes)||intersection(a.writes,b.writes))edges.push_back({i,j});}
 return edges;
}
inline bool coversEveryEdge(const Plan&p,const std::vector<ConflictEdge>&edges,ConflictEdge*firstFailure=nullptr){
 std::vector<size_t>prefix(1,0);bool valid=true,recorded=false;
 for(size_t i=0;i<p.rows.size();++i){auto&r=p.rows[i];if((i==0||r.forced)&&!r.before){valid=false;if(firstFailure&&!recorded){*firstFailure={i,i};recorded=true;}}prefix.push_back(prefix.back()+r.before);}
 // All graph edges are checked for EVERY attempted move, including edges
 // unrelated to this local pair. No shader byte extent or local-only proof.
 for(auto&e:edges)if(prefix[e.to+1]==prefix[e.from+1]){valid=false;if(firstFailure&&!recorded){*firstFailure=e;recorded=true;}}
 return valid;
}
inline void previousOpHoist(Plan&p,std::vector<size_t>&candidateLocations,const std::vector<size_t>&candidateDestinations){
 p.preHoistBarriers=p.barriers;
 const auto edges=allConflictEdges(p);
 if(edges.size()!=p.proofConflictPairs||!coversEveryEdge(p,edges))throw std::runtime_error("baseline edge graph/proof mismatch");
 std::set<size_t>candidateDispatches(candidateDestinations.begin(),candidateDestinations.end());
 for(size_t i=1;i<p.rows.size();++i){auto&current=p.rows[i];auto&previous=p.rows[i-1];
  if(!candidateDispatches.count(i)||!current.before||current.forced)continue;
  if(intersection(previous.writes,current.reads)||intersection(previous.reads,current.writes)||intersection(previous.writes,current.writes))continue;
  ++p.hoistAttempts;p.hoistEdgeChecks+=edges.size();bool oldPrevious=previous.before;
  previous.before=true;current.before=false;
  ConflictEdge failure{};
  if(!coversEveryEdge(p,edges,&failure)){p.rejectedHoists.push_back({i,i-1,edges.size(),failure.from,failure.to});previous.before=oldPrevious;current.before=true;++p.hoistRejected;continue;}
  p.hoists.push_back({i,i-1,edges.size(),oldPrevious});p.hoistMerged+=oldPrevious;
  for(auto&at:candidateLocations)if(at==i)at=i-1;
 }
 // The original independent all-pairs verifier is re-run on the final plan,
 // recomputing every conflict directly from R/W, without the trial cache.
 verify(p);
}
inline Plan plan(const std::vector<Entry>&trace){Plan p;IDs reads,writes;bool original=false,forced=false;std::vector<size_t>candidateDestinations;for(auto&e:trace){if(e.origin==Origin::Unknown)throw std::runtime_error("unknown command origin");if(e.type!=Type::Dispatch){if(e.type!=Type::CandidateBuffersScope&&e.type!=Type::Forced)throw std::runtime_error("unknown boundary type");if(e.origin==Origin::Dispatch)throw std::runtime_error("boundary labeled dispatch");++p.originalEntries;++p.origins[name(e.origin)];original=true;if(e.type==Type::CandidateBuffersScope){if(e.origin!=Origin::Scope)throw std::runtime_error("only original Buffers scope is movable");++p.candidates;candidateDestinations.push_back(p.rows.size());}else{++p.forcedEntries;forced=true;}continue;}if(e.origin!=Origin::Dispatch)throw std::runtime_error("dispatch origin mismatch");Row r;r.command=e.command;r.reads=e.reads;r.writes=e.writes;r.original=original;r.forced=forced;r.raw=intersection(writes,e.reads);r.war=intersection(reads,e.writes);r.waw=intersection(writes,e.writes);r.before=p.rows.empty()||forced||r.raw||r.war||r.waw;p.raw+=r.raw;p.war+=r.war;p.waw+=r.waw;if(r.before){reads.clear();writes.clear();++p.barriers;}p.originalLocations+=original;p.forcedLocations+=forced;p.insertedLocations+=r.before&&!original;p.removedLocations+=original&&!r.before;reads.insert(e.reads.begin(),e.reads.end());writes.insert(e.writes.begin(),e.writes.end());p.rows.push_back(r);original=forced=false;}p.trailing=original;
 verify(p);std::vector<size_t>candidateLocations;
 for(size_t at:candidateDestinations){size_t j=at;while(j<p.rows.size()&&!p.rows[j].before)++j;candidateLocations.push_back(j);}
 previousOpHoist(p,candidateLocations,candidateDestinations);
 p.barriers=p.insertedLocations=p.removedLocations=p.raw=p.war=p.waw=0;reads.clear();writes.clear();
 for(auto&r:p.rows){r.raw=intersection(writes,r.reads);r.war=intersection(reads,r.writes);r.waw=intersection(writes,r.writes);p.raw+=r.raw;p.war+=r.war;p.waw+=r.waw;if(r.before){++p.barriers;reads.clear();writes.clear();}else if(r.raw||r.war||r.waw)throw std::runtime_error("final open group contains conflict");p.insertedLocations+=r.before&&!r.original;p.removedLocations+=r.original&&!r.before;reads.insert(r.reads.begin(),r.reads.end());writes.insert(r.writes.begin(),r.writes.end());}
 for(size_t k=0;k<candidateDestinations.size();++k){size_t at=candidateDestinations[k],location=candidateLocations[k];if(location==p.rows.size())++p.candidatesCompletionCovered;else if(location<at)++p.candidatesHoisted;else if(location==at)++p.candidatesRetained;else ++p.candidatesDeferred;}
 return p;}

inline bool sameBarrierPlan(const Plan&a,const Plan&b){if(a.rows.size()!=b.rows.size())return false;for(size_t i=0;i<a.rows.size();++i)if(a.rows[i].before!=b.rows[i].before||a.rows[i].raw!=b.rows[i].raw||a.rows[i].war!=b.rows[i].war||a.rows[i].waw!=b.rows[i].waw)return false;if(a.rejectedHoists.size()!=b.rejectedHoists.size())return false;for(size_t i=0;i<a.rejectedHoists.size();++i)if(a.rejectedHoists[i].from!=b.rejectedHoists[i].from||a.rejectedHoists[i].uncoveredFrom!=b.rejectedHoists[i].uncoveredFrom||a.rejectedHoists[i].uncoveredTo!=b.rejectedHoists[i].uncoveredTo)return false;if(a.hoists.size()!=b.hoists.size()||a.hoistAttempts!=b.hoistAttempts||a.hoistRejected!=b.hoistRejected||a.candidatesHoisted!=b.candidatesHoisted)return false;for(size_t i=0;i<a.hoists.size();++i)if(a.hoists[i].from!=b.hoists[i].from||a.hoists[i].to!=b.hoists[i].to||a.hoists[i].merged!=b.hoists[i].merged)return false;return a.proofRAW==b.proofRAW&&a.proofWAR==b.proofWAR&&a.proofWAW==b.proofWAW;}
}
