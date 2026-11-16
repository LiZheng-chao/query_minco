#include "plant_gate.hpp"
#include <cassert>
#include <limits>
int main(){
 g4cp1::PlantGate g;g.hover=.98*9.81;g.nonce="manual";
 assert(!g.step_allowed(10));assert(!g.permit("manual",10));
 assert(!g.prepare("wrong",10));assert(g.prepare("manual",10));
 assert(g.phase==g4cp1::PlantGate::PREPARE);
 assert(g.accept(10,10,0,0,g.hover,1));
 for(int i=0;i<20;++i)assert(!g.accept(10,10.01,0,0,g.hover,1));
 assert(g.count==1);assert(!g.permit("manual",10.01));
 for(int i=1;i<5;++i)assert(g.accept(10+i*.01,10+i*.01,0,0,g.hover,1));
 assert(g.phase==g4cp1::PlantGate::ACK);assert(!g.step_allowed(10.04));
 assert(g.permit("manual",10.04));assert(g.step_allowed(10.04));
 assert(!g.accept(10.03,10.05,0,0,g.hover,1));assert(g.aborted);assert(!g.step_allowed(10.05));
 assert(!g.prepare("manual",10.05));assert(!g.permit("manual",10.05));
 g4cp1::PlantGate z;z.hover=g.hover;z.nonce="x";assert(z.prepare("x",10));assert(!z.accept(0,10,0,0,z.hover,1));assert(z.aborted);
 g4cp1::PlantGate h;h.hover=g.hover;h.nonce="x";assert(!h.accept(10,10,0,0,h.hover,1));assert(h.aborted);
 g4cp1::PlantGate stale;stale.hover=g.hover;stale.nonce="x";assert(stale.prepare("x",10));
 for(int i=0;i<5;++i)assert(stale.accept(10+i*.01,10+i*.01,0,0,g.hover,1));
 assert(stale.permit("x",10.04));assert(!stale.step_allowed(10.25));assert(stale.aborted);
}
