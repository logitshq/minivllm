#pragma once
#include "pipeline_core.h"
#include <cstddef>
#include <cstdint>
namespace v62 {
inline bool ordinaryAllowed(bool diagnostic,uintptr_t addr,size_t n,uintptr_t prediction){return diagnostic||(addr==prediction&&n==4);}
inline bool quiescent(const V55Core&c,size_t retainedJobs,bool nativeError){return !nativeError&&c.ready&&c.lastValid&&c.pending.empty()&&!retainedJobs&&c.requestSubmitted>0&&c.last.id==c.requestSubmitted&&c.requestCompleted==c.requestSubmitted&&c.requestDelivered==c.requestSubmitted&&!c.requestDiscarded&&!c.requestFailed;}
}
