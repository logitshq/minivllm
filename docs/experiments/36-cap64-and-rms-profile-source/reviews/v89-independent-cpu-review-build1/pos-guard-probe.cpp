#include <cstdint>
#include <climits>
// Exact original guard subexpression; CPU-only, no Metal/Foundation.
__attribute__((noinline)) bool rejected(int32_t token,int32_t pos,int32_t len,uint32_t capacity){return token<0||pos<0||len!=pos+1||len>(int)capacity;}
int main(){volatile int32_t pos=INT32_MAX,len=INT32_MIN;return rejected(1,pos,len,64)?0:2;}
