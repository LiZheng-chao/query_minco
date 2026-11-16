"""Bounded exact-time match;no receipt-time freshness substitution."""
from collections import OrderedDict
class Pairing:
 def __init__(self):
  self.poses=OrderedDict();self.depths=OrderedDict();self.last=0.;self.drops=[]
 def pose(self,stamp,value): 
  self.poses[stamp]=value
  while len(self.poses)>512:self.poses.popitem(last=False)
 def depth(self,stamp,value):
  self.depths[stamp]=value
  while len(self.depths)>4:
   s,_=self.depths.popitem(last=False);self.drops.append((s,'QUEUE_BOUND'))
 def take(self,now):
  ready=[]
  for s in sorted(self.depths):
   if not 0<=now-s<=.2:self.depths.pop(s);self.drops.append((s,'SOURCE_EXPIRED'));continue
   if s<=self.last:self.depths.pop(s);self.drops.append((s,'SUPERSEDED'));continue
   if s in self.poses:
    ready.append((s,self.depths.pop(s),self.poses[s]));self.last=s
  return ready
