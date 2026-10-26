"""Export existing fully certified immutable references to bounded C++ transport."""
import argparse,json,pickle,struct,time,hashlib
from pathlib import Path
import numpy as np
from query_minco.inference.committed_reference import prepare_reference,FrozenPolynomial
from query_minco.inference.trajectory_execution import Program,quintic,trim
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
def thaw(p):return Program(np.frombuffer(p.coefficients,dtype='<f8').reshape(-1,3,6),p.durations)
def export(bundles,path):
    with path.open('xb') as f:
        f.write(b'QMRF005\0');f.write(struct.pack('<i',len(bundles)))
        def poly(p):
            f.write(struct.pack('<i',len(p.durations)));c=np.frombuffer(p.coefficients,dtype='<f8').reshape(-1,3,6)
            for a,T in zip(c,p.durations):f.write(struct.pack('<d',T));f.write(a.astype('<f8').tobytes())
        for bid,b in bundles.items():
            f.write(struct.pack('<idddd',bid,b.prepared_ros,b.prepared_mono,b.valid_until_ros,b.valid_until_mono));poly(b.prefix)
            f.write(struct.pack('<i',len(b.stops)))
            for row in b.stops:f.write(struct.pack('<d',row.phase));poly(row.polynomial)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    with (out/'PREPARED_REFERENCES.pkl').open('rb') as f:base=pickle.load(f)
    result={0:base['STATIC'],1:base['NOMINAL'],2:base['WATCHDOG']}
    now=time.time();mm=time.monotonic();volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],now+120,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
    for bid,phase in zip(range(3,7),[2.56,2.58,2.60,2.62]):
        stop=next(row for row in base['WATCHDOG'].stops if abs(row.phase-phase)<1e-9)
        head=stop.polynomial.pva(stop.polynomial.duration)
        tail=head.copy();tail[0]+=np.array([2,0,0])
        prefix=trim(quintic(head,tail,8.),0,2.8)
        result[bid]=prepare_reference(prefix,volume,now,mm)
    # Independently validate every actual transported prefix+stop digest and extrema.
    checked=0
    from query_minco.inference.trajectory_execution import concatenate,MotionLimits
    for b in result.values():
        pre=thaw(b.prefix)
        for row in b.stops:
            sp=thaw(row.polynomial);full=sp if row.phase==0 else concatenate(trim(pre,0,row.phase),sp)
            full.validate(MotionLimits());proof=json.loads(row.proof)
            assert full.digest==proof['trajectory_sha256'] and proof['status']=='CERTIFIED_FREE' and proof['scope']=='EXTERNAL_SUPERVISOR'
            checked+=1
    export(result,out/'NATIVE_REFERENCES.bin')
    with (out/'TRANSPORT_BUNDLES.pkl').open('xb') as f:pickle.dump(result,f,protocol=4)
    with (out/'EXPORT_AUDIT.json').open('x') as f:json.dump(dict(bank_count=checked,bundle_ids=list(result),native_file_sha256=hashlib.sha256((out/'NATIVE_REFERENCES.bin').read_bytes()).hexdigest(),expiry_ros_min=min(b.valid_until_ros for b in result.values()),scope='EXTERNAL_SUPERVISOR',Query=False),f,indent=2)
    print('fully checked rows',checked)
if __name__=='__main__':main()
