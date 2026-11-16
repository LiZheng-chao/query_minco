#include "plant_gate.hpp"
#include <cassert>
#include <limits>
int main(){
 g4c::PlantGate g;g.hover=.98*9.81;g.nonce="manual";
 assert(!g.step_allowed(10));assert(g.steps==0);
 assert(!g.permit("manual",std::numeric_limits<double>::quiet_NaN()));
 for(int i=0;i<5;++i)g.accept(10,10,0,0,g.hover,1);
 assert(!g.step_allowed(10));assert(!g.permit("wrong",10));
 assert(g.permit("manual",10));assert(g.step_allowed(10));
 assert(!g.step_allowed(10.21));assert(g.aborted);
 g.accept(10.22,10.22,0,0,g.hover,1);assert(!g.permit("manual",10.22));
 g4c::PlantGate z;z.hover=g.hover;z.accept(0,10,0,0,g.hover,1);assert(z.aborted);
 g4c::PlantGate f;f.hover=g.hover;f.nonce="manual";for(int i=0;i<5;++i)f.accept(10,10,0,0,0,1);
 assert(f.aborted);assert(!f.permit("manual",10));assert(!f.step_allowed(10));assert(f.steps==0);
}
