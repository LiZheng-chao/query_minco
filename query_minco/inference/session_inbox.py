"""Fail-closed readonly RPC. Trusted fault/EOF latches, prevents new inference.
No physical publishing, old-proposal replay, or hidden reconnect after fault.
"""
import json,socket,struct,time
from query_minco.inference.dynamic_proposal import Inbox,packet
from query_minco.inference.causal_source_state import SourceBoundInbox
from query_minco.inference.input_lifecycle import SessionStopped
class GuardedSourceInbox(SourceBoundInbox):
 FATAL=frozenset(('SESSION_FAULT_LATCHED','SESSION_FINISHED','PUBLISHER_CONFLICT','CLOCK_JUMP'))
 def __init__(self,address,session,expected_peer_pid=None,transport=None):
  super().__init__(address);self.session=session;self.expected_peer_pid=expected_peer_pid;self.transport=transport;self.rpc_calls=0
 def _raw(self,body):
  if self.transport:return self.transport(body)
  with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
   sock.settimeout(.1);sock.connect(self.address)
   if self.expected_peer_pid is not None:
    peer=struct.unpack('3i',sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
    if peer[0]!=self.expected_peer_pid:raise ValueError('CPP_PEER_IDENTITY')
   sock.sendall(struct.pack('<i',len(body))+body)
   def receive(n):
    data=b''
    while len(data)<n:
     chunk=sock.recv(n-len(data))
     if not chunk:raise EOFError('INBOX_EOF')
     data+=chunk
    return data
   length=struct.unpack('<i',receive(4))[0]
   if not 0<length<=16384:raise ValueError('RESPONSE_BOUND')
   return json.loads(receive(length))
 def request(self,body):
  self.session.require_live();wall=time.perf_counter_ns();cpu=time.thread_time_ns();self.rpc_calls+=1
  try:response=self._raw(body)
  except (OSError,EOFError,ValueError) as exc:
   self.session.trip('CPP_INBOX_CLOSED_OR_INVALID:'+type(exc).__name__);raise SessionStopped(self.session.reason) from exc
  response['client_wall_ms']=(time.perf_counter_ns()-wall)/1e6;response['client_thread_cpu_ms']=(time.thread_time_ns()-cpu)/1e6
  if response.get('fault_latched') in (True,'true') or response.get('reason') in self.FATAL:
   self.session.trip(response.get('reason','SESSION_FAULT_LATCHED'));raise SessionStopped(self.session.reason)
  if response.get('passed') in (True,'true'):return response
  raise ValueError('CPP_GATE_'+response.get('reason','REJECTED'))
