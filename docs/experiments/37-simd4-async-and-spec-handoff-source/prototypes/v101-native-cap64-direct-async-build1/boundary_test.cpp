// CPU-only actual two guard predicates, extracted identically from replay.mm.
#include <cassert>
#include <cstdint>
#include <climits>
#include <iostream>
bool allows(uint32_t capacity) {return capacity==64||capacity==256;}
bool valid(int32_t token,int32_t pos,int32_t len,uint32_t capacity) {
 if(token<0||pos<0||pos>=(int)capacity||len!=pos+1||len>(int)capacity)return false;
 return true;
}
int main(){
 uint64_t checks=0;
 for(uint32_t cap: {64u,256u}){
  assert(allows(cap));++checks;
  for(int32_t p: {INT32_MIN,-1,0,1,62,63,64,254,255,256,INT32_MAX})
   for(int32_t l: {INT32_MIN,-1,0,1,2,63,64,65,255,256,257,INT32_MAX})
    for(int32_t t: {-1,0,INT32_MAX}){
      bool expect=t>=0&&p>=0&&(int64_t)p<(int64_t)cap&&(int64_t)l==(int64_t)p+1&&(int64_t)l<=cap;
      assert(valid(t,p,l,cap)==expect);++checks;
    }
 }
 for(uint32_t cap: {0u,1u,32u,63u,65u,128u,257u,UINT32_MAX}){assert(!allows(cap));++checks;}
 std::cout<<"CPU_BOUNDARY_UBSAN_PASS checks="<<checks<<"\n";
}
