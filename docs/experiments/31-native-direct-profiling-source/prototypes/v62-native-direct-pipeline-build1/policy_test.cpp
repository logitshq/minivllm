#include "direct_policy.h"
#include "feedback_plan.h"
#include <iostream>
#include <stdexcept>
static void need(bool b){if(!b)throw std::runtime_error("V62 CPU policy assertion");}
int main(){std::string why;V55Core c;unsigned tests=0;auto check=[&](bool x){need(x);++tests;};
 check(!v62::quiescent(c,0,false));check(c.reset(why));V55Ticket a,b,d;
 check(c.plan(a,why));c.submitted(a);check(c.plan(b,why));c.submitted(b);
 check(!v62::quiescent(c,2,false));check(c.canWait(a.id,false,why));c.finished(a,false,true);
 check(!v62::quiescent(c,1,false));check(v62::ordinaryAllowed(false,100,4,100));check(!v62::ordinaryAllowed(false,101,4,100));check(!v62::ordinaryAllowed(false,100,8,100));check(v62::ordinaryAllowed(true,101,8,100));
 check(c.read());check(c.plan(d,why));c.submitted(d);check(!c.lastValid);check(c.canWait(b.id,false,why));c.finished(b,false,true);check(c.read());check(c.canWait(d.id,false,why));c.finished(d,false,true);
 check(v62::quiescent(c,0,false));check(!v62::quiescent(c,1,false));check(!v62::quiescent(c,0,true));check(c.last.outputBank==1);check(c.read());check(v62::quiescent(c,0,false));
 check(c.reset(why));check(!v62::quiescent(c,0,false));check(c.plan(a,why));c.submitted(a);c.finished(a,true,true);check(!v62::quiescent(c,0,false));
 check(c.reset(why));check(c.plan(a,why));c.submitted(a);c.finished(a,false,false);check(!v62::quiescent(c,0,false));
 auto proof=v57::validate({{0,4},{8,4}},{{16,4},{24,4}},32);check(proof.inputOutputPairs==4&&proof.inputPairs==1);
 bool rejected=false;try{v57::validate({{0,4}},{{0,4}},32);}catch(...){rejected=true;}check(rejected);
 std::cout<<"{\"status\":\"PASS_CPU_V62_FIFO_MODE_QUIESCENCE\",\"assertions\":"<<tests<<",\"reset_pending_jobs_discard_failure_guard\":true,\"production_only_prediction_ordinary\":true,\"original_FIFO_core_used\":true,\"GPU_used\":false}\n";
}
