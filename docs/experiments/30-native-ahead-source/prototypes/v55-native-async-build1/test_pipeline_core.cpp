#include "pipeline_core.h"
#include <cassert>
#include <iostream>
static V55Ticket submit(V55Core&c){std::string why;V55Ticket j;assert(c.plan(j,why));c.submitted(j);return j;}
int main(){std::string why;V55Core c;V55Ticket j;assert(!c.plan(j,why));assert(!c.canWriteSeed());assert(c.reset(why));assert(c.canWriteSeed());
 auto a=submit(c),b=submit(c);assert(a.id==1&&a.slot==0&&a.inputBank==0&&a.outputBank==1);assert(b.id==2&&b.slot==1&&b.inputBank==1&&b.outputBank==0);
 assert(!c.canWriteSeed());assert(!c.plan(j,why));assert(!c.canWait(2,false,why));assert(!c.reset(why));assert(c.pending.size()==2);
 assert(c.canWait(1,false,why));c.finished(a,false,true);assert(c.lastValid&&c.last.outputBank==1&&!c.lastConsumed);assert(!c.plan(j,why));assert(!c.canWait(2,false,why));
 assert(c.read());auto d=submit(c);assert(d.id==3&&d.slot==0&&d.inputBank==0);assert(!c.lastValid);assert(!c.read());
 assert(c.canWait(2,false,why));c.finished(b,false,true);assert(c.last.outputBank==0);assert(c.read());auto e=submit(c);assert(e.id==4&&e.slot==1&&e.inputBank==1);
 assert(c.canWait(3,false,why));c.finished(d,false,true);assert(c.read());assert(c.canWait(4,false,why));c.finished(e,false,true);assert(c.read());
 assert(c.pending.empty()&&c.totalCompleted==4&&c.totalDelivered==4&&c.maxOutstanding==2);assert(c.reset(why));assert(c.requestCompleted==0&&c.requestSubmitted==0&&c.nextBank==0);
 auto x=submit(c),y=submit(c);assert(x.id==1&&y.id==2);assert(c.canWait(x.id,true,why));c.finished(x,true,true);assert(c.canWait(y.id,true,why));c.finished(y,true,true);assert(!c.lastValid&&c.requestDiscarded==2&&c.totalDiscarded==2&&c.totalCompleted==6);
 assert(c.reset(why));auto z=submit(c);c.finished(z,true,false);assert(c.requestFailed==1&&c.requestCompleted==0&&c.totalFailed==1&&c.pending.empty());
 assert(c.reset(why));assert(c.canWriteSeed());std::cout<<"PASS_FIFO_DEPTH2_BANKS_SLOTS_CONSUME_RESET_WRITE_FAILED_DISCARD\n";
}
