"""Bounded immutable packet for the C++ READONLY inbox. No control permission."""
import hashlib,json,socket,struct,time
import numpy as np
from query_minco.inference.proposal_worker import curve_bytes
from query_minco.inference.observed_space import ErrorBudget
METHOD='query_minco_b2c1_c1_e1_v1'
CHECKPOINT='2352de6dd1e4a839b78ec12c115b5319b34da3c61cada105dbdeb24a4faec6d1'
CALIBRATION=hashlib.sha256(np.array([160.,128.,100,0,79.5,0,100,63.5,0,0,1],dtype='<f8').tobytes()).hexdigest()
ERRORS=list(ErrorBudget().__dict__.values())
def packet(meta,body):
 header=json.dumps(meta,separators=(',',':'),allow_nan=False).encode()
 if len(header)>16384 or len(body)+len(header)>262140:raise ValueError('PACKET_BOUND')
 return struct.pack('<i',len(header))+header+body
def metadata(bank,snapshot,sequence,source_stamp,position,rotation,activation_ros,activation_mono,config):
 return dict(method=METHOD,checkpoint=CHECKPOINT,frame='world:B0:optical_depth_128x160',calibration=CALIBRATION,
 owner_epoch=config['owner_epoch'],parent_version=1,parent_digest=config['parent_digest'],snapshot_token=int(snapshot['token']),
 sequence=sequence,depth_stamp=source_stamp,source_odom_stamp=source_stamp,source_position=np.asarray(position).tolist(),
 source_rotation=np.asarray(rotation).reshape(-1).tolist(),state_stamp=float(snapshot['stamp']),state_recv_mono=float(snapshot['recv_mono']),
 planning_pva=list(map(float,snapshot['pva'])),prepared_ros=bank.prepared_ros,prepared_mono=bank.prepared_mono,
 activation_ros=activation_ros,activation_mono=activation_mono,valid_until_ros=bank.valid_until_ros,valid_until_mono=bank.valid_until_mono,
 geometry=config['geometry'],error_budget=ERRORS,body_sha256=bank.digest,stop_bank_version=1)
class Inbox:
 def __init__(self,address):
  self.address='\0'+address[1:] if address.startswith('@') else address
 def request(self,body):
  t=time.perf_counter_ns();cpu=time.thread_time_ns()
  with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
   sock.settimeout(.1);sock.connect(self.address);sock.sendall(struct.pack('<i',len(body))+body)
   def read(n):
    data=b''
    while len(data)<n:
     b=sock.recv(n-len(data))
     if not b:raise ValueError('INBOX_EOF')
     data+=b
    return data
   size=struct.unpack('<i',read(4))[0]
   if not 0<size<=16384:raise ValueError('RESPONSE_BOUND')
   response=json.loads(read(size))
  response['client_wall_ms']=(time.perf_counter_ns()-t)/1e6;response['client_thread_cpu_ms']=(time.thread_time_ns()-cpu)/1e6
  if response.get('passed') in (True,'true'):return response
  raise ValueError('CPP_GATE_'+response.get('reason','REJECTED'))
 def snapshot(self):return self.request(b'S')
 def submit(self,meta,body):return self.request(b'P'+packet(meta,body))
