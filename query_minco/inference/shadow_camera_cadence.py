"""Explicit cadence-only overlay of existing exact-ray camera. NOT run in007.
No resize, new renderer, calibration or mask change. Must preregister query Hz.
"""
import inspect,sys
from query_minco.inference import shadow_camera_low_speed as original
def period_ns(hz):
 if not 0<hz<=33:raise ValueError('NATIVE_RATE_LIMIT')
 return int(round(1e9/hz))
def output_stamps(stamps,hz):
 period=period_ns(hz);next_stamp=None;out=[]
 for stamp in stamps:
  if next_stamp is None:next_stamp=stamp
  if stamp>=next_stamp:
   while next_stamp<=stamp:next_stamp+=period
   out.append(stamp)
 return out
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--query-hz',type=float,required=True)
 args,rest=p.parse_known_args();period=period_ns(args.query_hz)
 source=inspect.getsource(original.main)
 if source.count('next_query+=200_000_000')!=1:raise ValueError('CAMERA_BASE_BINDING_CHANGED')
 source=source.replace('next_query+=200_000_000','next_query+=_query_period_ns')
 env=dict(vars(original));env['_query_period_ns']=period;sys.argv=[sys.argv[0]]+rest
 exec(compile(source,__file__+'[cadence-only-overlay]','exec'),env);env['main']()
if __name__=='__main__':main()
