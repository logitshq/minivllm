#pragma once
#include <cstdint>
#include <deque>
#include <string>
struct V55Ticket {uint64_t id=0;unsigned slot=0,inputBank=0,outputBank=0;};
struct V55Core {
 bool ready=false,lastValid=false,lastConsumed=false;unsigned nextBank=0;bool busy[2]={false,false};
 uint64_t requestSubmitted=0,requestCompleted=0,requestDelivered=0,requestDiscarded=0,requestFailed=0;
 uint64_t totalSubmitted=0,totalCompleted=0,totalDelivered=0,totalDiscarded=0,totalFailed=0,maxOutstanding=0;
 std::deque<V55Ticket> pending;V55Ticket last;
 bool reset(std::string&why){if(!pending.empty()){why="reset requires drain/no inflight tickets";return false;}ready=true;lastValid=lastConsumed=false;nextBank=0;busy[0]=busy[1]=false;requestSubmitted=requestCompleted=requestDelivered=requestDiscarded=requestFailed=0;last={};return true;}
 bool canWriteSeed()const{return ready&&pending.empty()&&requestSubmitted==0;}
 bool plan(V55Ticket&j,std::string&why)const{
  if(!ready){why="submit requires pipeline reset";return false;}
  if(pending.size()>=2){why="at most two outstanding tickets";return false;}
  if(lastValid&&!lastConsumed){why="consume last delivered result before next submit";return false;}
  j={requestSubmitted+1,unsigned(requestSubmitted%2),nextBank,nextBank^1};
  if(busy[j.slot]){why="stage slot still owned by unwaited ticket";return false;}return true;
 }
 void submitted(const V55Ticket&j){pending.push_back(j);busy[j.slot]=true;nextBank=j.outputBank;++requestSubmitted;++totalSubmitted;if(pending.size()>maxOutstanding)maxOutstanding=pending.size();lastValid=lastConsumed=false;}
 bool canWait(uint64_t ticket,bool discard,std::string&why)const{
  if(pending.empty()||pending.front().id!=ticket){why="wait requires oldest FIFO ticket";return false;}
  if(!discard&&lastValid&&!lastConsumed){why="consume delivered result before waiting another ticket";return false;}return true;
 }
 void finished(const V55Ticket&j,bool discard,bool success){pending.pop_front();busy[j.slot]=false;
  if(success){++requestCompleted;++totalCompleted;if(discard){++requestDiscarded;++totalDiscarded;lastValid=lastConsumed=false;}else{++requestDelivered;++totalDelivered;last=j;lastValid=true;lastConsumed=false;}}
  else{++requestFailed;++totalFailed;lastValid=lastConsumed=false;}
 }
 bool read(){if(!lastValid)return false;lastConsumed=true;return true;}
};
