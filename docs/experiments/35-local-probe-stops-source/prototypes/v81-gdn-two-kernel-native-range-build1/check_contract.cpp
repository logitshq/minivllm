#include "range_contract.h"
#include <cassert>
#include <iostream>
int main(){
 using namespace v81;std::vector<Bank>b(30);std::vector<Access>ops(60);
 for(unsigned i=0;i<30;++i){for(unsigned a=0;a<4;++a)b[i][a]=1+(i%15)*4+a;for(unsigned a=4;a<10;++a)b[i][a]=1000+i*6+a;
  ops[i*2].reads={b[i][0]};ops[i*2].writes={b[i][4],b[i][5],b[i][6],b[i][7]};
  ops[i*2+1].reads={b[i][1],b[i][2],b[i][3],b[i][4],b[i][5],b[i][6]};if(i<15)ops[i*2+1].reads.insert(b[i][7]);ops[i*2+1].writes={b[i][8],b[i][9]};}
 assert(dependencies(ops,b).empty());auto base=ops;auto banks0=b;
 ops[1].reads.erase(b[0][4]);assert(!dependencies(ops,b).empty());ops=base;
 ops[1].reads.insert(b[1][4]);assert(!dependencies(ops,b).empty());ops=base;
 ops[1].writes.insert(b[0][3]);assert(!dependencies(ops,b).empty());ops=base;
 b[1][8]=b[0][8];assert(!dependencies(ops,b).empty());b=banks0;
 b[15][0]=b[0][1];assert(!dependencies(ops,b).empty());b=banks0;
 b[0][4]=b[0][3];assert(!dependencies(ops,b).empty());b=banks0;
 ops[1].reads.insert(b[0][8]);assert(!dependencies(ops,b).empty());ops=base;
 ops[1].reads.insert(b[1][3]);assert(!dependencies(ops,b).empty());ops=base;
 ops[0].reads.insert(b[0][1]);assert(!dependencies(ops,b).empty());ops=base;
 ops[1].reads.insert(999999);assert(!dependencies(ops,b).empty());ops=base;
 for(unsigned arm=0;arm<2;++arm)for(unsigned stage=0;stage<2;++stage){std::string fn=function(arm,stage);assert(!fn.empty());assert(!storeOnly(arm,stage,fn+"bad",stage?9:1,stage?"S_new":"q_ready",1));assert(!storeOnly(arm,stage,fn,6,"S",1));assert(storeOnly(arm,stage,fn,stage?9:1,stage?"S_new":"q_ready",1));assert(!storeOnly(arm,stage,fn,stage?9:1,stage?"S_new":"q_ready",0));}
 assert(geometry(0,1,{8192,1,1},{128,1,1}));assert(geometry(0,2,{64,1,1},{128,1,1}));assert(geometry(1,1,{128,192,1},{32,4,1}));assert(geometry(1,2,{4,48,1},{32,4,1}));assert(!geometry(1,1,{128,192,1},{64,4,1}));
 for(unsigned arm=0;arm<3;++arm)for(unsigned count:{0u,1u,8192u,8193u})assert(groups(arm,count)==(arm<2&&count>=1&&count<=8192));
 std::cout<<"PASS_CPU_DEPENDENCY_ALIAS_EXACT_ACCESS_GEOMETRY_LOOP_CONTRACT\n";
}
