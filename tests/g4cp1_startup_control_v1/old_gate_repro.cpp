#include "/home/lizheng/query_minco_mainline/query_minco/integration/yopo_startup_v2/plant_gate.hpp"
#include <cassert>
#include <iostream>
int main(){g4c::PlantGate g;g.nonce="x";g.hover=.98*9.81;for(int i=0;i<5;++i)g.accept(10,10,0,0,g.hover,1);bool permitted=g.permit("x",10);std::cout<<"FROZEN_V2_DUPLICATE_COUNT="<<g.count<<" PERMIT_WITHOUT_PREPARE="<<permitted<<"\n";assert(g.count==5&&permitted);}
