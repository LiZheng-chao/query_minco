"""Native worker binding. No ROS/control. All56 phases and full swept bodies.
Only numerical basis/code cached; evidence expiry always inherited unchanged.
"""
import ctypes,dataclasses,hashlib,json,struct,time
from pathlib import Path
import numpy as np
from query_minco.inference.observed_space import Scope,ErrorBudget
from query_minco.inference.trajectory_execution import Program
@dataclasses.dataclass(frozen=True)
class NativeBank:
 body:bytes
 digest:str
 prepared_ros:float
 prepared_mono:float
 valid_until_ros:float
 valid_until_mono:float
 horizon:float
 minimum_slack:float
 phase_count:int
 timing:dict
def curve_bytes(p):
 data=struct.pack('<i',len(p.durations))
 for c,t in zip(p.coefficients,p.durations):data+=struct.pack('<d',t)+np.asarray(c,dtype='<f8').tobytes()
 return data
def decode_body(body):
 offset=0
 def curve():
  nonlocal offset
  n=struct.unpack_from('<i',body,offset)[0];offset+=4;c=[];d=[]
  for _ in range(n):
   d.append(struct.unpack_from('<d',body,offset)[0]);offset+=8
   c.append(np.frombuffer(body,dtype='<f8',count=18,offset=offset).reshape(3,6).copy());offset+=144
  return Program(c,d)
 prefix=curve();n=struct.unpack_from('<i',body,offset)[0];offset+=4;stops=[]
 for _ in range(n):
  phase=struct.unpack_from('<d',body,offset)[0];offset+=8;stops.append((phase,curve()))
 if offset!=len(body):raise ValueError('TRAILING_BODY')
 return prefix,stops
class NativeWorker:
 stages=('prefix_continuous_kinematics','56_phase_heads','soft_stop_solve_and_continuous_extrema','56_full_reserve_PVA_lifetime','Bernstein_whole_body_sweep','certificate_body_encode','native_output_copy')
 def __init__(self,library):
  self.lib=ctypes.CDLL(str(library))
  ptr=ctypes.POINTER
  self.lib.qm_prepare.argtypes=[ptr(ctypes.c_double),ctypes.c_int,ctypes.c_double,ctypes.c_double,ptr(ctypes.c_ubyte),ctypes.c_int,ptr(ctypes.c_double),ctypes.c_char_p,ctypes.c_int]
  self.lib.qm_prepare.restype=ctypes.c_int
  self.output=(ctypes.c_ubyte*(1024*1024))()
 def prepare(self,prefix,volume,now_ros,now_mono,inherited_mono_expiry=None):
  if volume.scope!=Scope.EXTERNAL_SUPERVISOR or len(volume.cells)!=1:raise ValueError('UNSUPPORTED_TRUSTED_CELL')
  cell=volume.cells[0]
  if not np.array_equal(cell.A,np.r_[np.eye(3),-np.eye(3)]) or not np.array_equal(cell.b,[6.8,2.8,3.8,2.8,2.8,-.2]) or not np.array_equal(cell.anchor,[0,0,0]) or volume.errors!=ErrorBudget():raise ValueError('CELL_OR_ERROR_BUDGET_NOT_BOUND')
  valid_for=min(120.,cell.valid_until-now_ros-.001)
  if not 1<=valid_for<=120:raise ValueError('INHERITED_CELL_EXPIRED')
  start=time.perf_counter_ns();cpu=time.thread_time_ns()
  data=np.ascontiguousarray(np.c_[prefix.durations,prefix.coefficients.reshape(-1,18)],dtype=np.float64);metrics=(ctypes.c_double*40)();error=ctypes.create_string_buffer(256)
  n=self.lib.qm_prepare(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(data),now_ros,cell.valid_until,self.output,len(self.output),metrics,error,len(error))
  if n<0:raise ValueError(error.value.decode())
  beforecopy=time.perf_counter_ns();body=bytes(self.output[:n]);digest=hashlib.sha256(body).hexdigest();finish=time.perf_counter_ns()
  timing={name:dict(wall_ms=metrics[i*3],thread_cpu_ms=metrics[i*3+1],runqueue_wait_ms=metrics[i*3+2],wall_minus_thread_cpu_ms=max(0,metrics[i*3]-metrics[i*3+1])) for i,name in enumerate(self.stages)}
  timing.update(python_immutable_copy_and_sha_ms=(finish-beforecopy)/1e6,total_wall_ms=(finish-start)/1e6,total_thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,root_solves=0)
  valid_mono=now_mono+valid_for if inherited_mono_expiry is None else min(now_mono+valid_for,inherited_mono_expiry-.001)
  if now_mono+metrics[30]+.06>valid_mono:raise ValueError('MONOTONIC_INHERITED_LIFETIME')
  return NativeBank(body,digest,now_ros,now_mono,now_ros+valid_for,valid_mono,metrics[30],metrics[31],int(metrics[33]),timing)
