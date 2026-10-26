"""Exact existing rays with explicit query cadence and real timestamp diagnostics."""
import inspect,sys
from query_minco.inference import shadow_camera_low_speed as original
from query_minco.inference.shadow_camera_cadence import period_ns
def camera_source():
 s=inspect.getsource(original.main).replace('next_query+=200_000_000','next_query+=_query_period_ns')
 s=s.replace('nonlocal next_query,count,proofs','nonlocal next_query,count,proofs\n        received_mono_ns=time.monotonic_ns()')
 s=s.replace('rec=dict(stamp_ns=stamp,query_index=count,','rec=dict(received_mono_ns=received_mono_ns,emit_mono_ns=time.monotonic_ns(),query_period_ns=_query_period_ns,stamp_ns=stamp,query_index=count,')
 s=s.replace('log.write(json.dumps(rec)+"\\n");count+=1','log.write(json.dumps(rec)+"\\n");count+=1')
 needle='        except Exception as e:'
 assert s.count(needle)==1
 s=s.replace(needle,'            log.write(json.dumps(dict(type="NATIVE_INPUT",stamp_ns=image.header.stamp.to_nsec(),received_mono_ns=received_mono_ns,processed_mono_ns=time.monotonic_ns()))+"\\n")\n'+needle)
 return s
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--query-hz',required=True,type=float);args,rest=p.parse_known_args()
 env=dict(vars(original));env['_query_period_ns']=period_ns(args.query_hz);sys.argv=[sys.argv[0]]+rest
 exec(compile(camera_source(),__file__+'[exact-ray-cadence]','exec'),env);env['main']()
if __name__=='__main__':main()
