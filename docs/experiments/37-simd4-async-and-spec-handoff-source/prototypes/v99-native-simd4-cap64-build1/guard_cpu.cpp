#include <climits>
#include <cassert>
#include <cstdio>
#include <vector>
static bool guard(int token,int pos,int len,unsigned capacity){
 if(capacity!=64&&capacity!=256)return false;
 if(token<0||pos<0||pos>=(int)capacity||len!=pos+1||len>(int)capacity)return false;
 return true;
}
int main(){unsigned checks=0;for(unsigned cap:{64u,256u}){
 for(int pos:{INT_MIN,-1,0,1,63,64,65,255,256,INT_MAX}){
  int len=(pos>=0&&pos<(int)cap)?pos+1:0;
  bool expected=pos>=0&&pos<(int)cap;assert(guard(0,pos,len,cap)==expected);++checks;
  for(int bad:{INT_MIN,-1,0,INT_MAX}){assert(!guard(0,pos,bad,cap));++checks;}
 }
 for(int pos=0;pos<(int)cap;++pos){assert(guard(0,pos,pos+1,cap));assert(!guard(-1,pos,pos+1,cap));checks+=2;}
}for(unsigned cap:{0u,1u,32u,63u,65u,128u,257u,UINT_MAX}){assert(!guard(0,0,1,cap));++checks;}
 std::printf("PASS_CPU_UBSAN_SCALAR_GUARD checks=%u\n",checks);}
