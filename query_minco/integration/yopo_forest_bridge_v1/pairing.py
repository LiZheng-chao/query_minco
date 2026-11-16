"""Exact source-time pairing independent of ROS callback arrival order."""
from collections import OrderedDict,deque
class Pairing:
 def __init__(self,max_poses=512,max_depth=4,max_age=.2):
  self.poses=OrderedDict();self.depth=OrderedDict();self.ready=deque(maxlen=4);self.max_poses=max_poses;self.max_depth=max_depth;self.max_age=max_age;self.excluded=[]
 def add(self,kind,stamp,value,now):
  if stamp<=0 or now-stamp<0 or now-stamp>self.max_age:return False
  target=self.poses if kind=='pose' else self.depth
  target[stamp]=value
  while len(self.poses)>self.max_poses:self.poses.popitem(last=False)
  while len(self.depth)>self.max_depth:self.depth.popitem(last=False)
  for s in list(self.depth):
   if now-s>self.max_age:self.depth.pop(s);continue
   if s in self.poses:self.ready.append((s,self.depth.pop(s),self.poses[s]))
  return True
 def latest(self,now):
  values=[item for item in self.ready if 0<=now-item[0]<=self.max_age]
  self.ready.clear()
  return max(values,key=lambda v:v[0]) if values else None
