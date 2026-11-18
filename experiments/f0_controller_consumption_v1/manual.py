"""Explicit researcher entry; no automatic physical episode execution."""
from pathlib import Path
import sys,json,hashlib,xmlrpc.client
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_outputs/f0_controller_consumption_20261118_v1'
sys.path.insert(0,str(ROOT))
def verify_controller(uri,nonce):
    master=xmlrpc.client.ServerProxy(uri)
    state=master.getSystemState('/f0_audit_verifier')
    if state[0]!=1:raise ValueError('MASTER_STATE')
    pubs=dict(state[2][0])
    expected={'/p2/so3_raw':['/f0_audit_position_controller'],
              '/p2/so3_cmd':['/p2_execution_supervisor'],
              '/p2/position_cmd':['/p2_execution_supervisor']}
    for topic,nodes in expected.items():
        if sorted(pubs.get(topic,[]))!=nodes:raise ValueError('CONTROLLER_PUBLISHER_IDENTITY '+topic)
    endpoint=master.lookupNode('/f0_audit_verifier','/f0_audit_position_controller')
    if endpoint[0]!=1:raise ValueError('CONTROLLER_NODE_MISSING')
    pid=xmlrpc.client.ServerProxy(endpoint[2]).getPid('/f0_audit_verifier')[2]
    maps=Path('/proc/%d/maps'%pid).read_text()
    library=OUT/'devel_v2/lib/libf0_audit_nodelet.so'
    build=json.loads((OUT/'BUILD_IDENTITY_v2.json').read_text())
    actual=hashlib.sha256(library.read_bytes()).hexdigest()
    if str(library) not in maps or '/libso3_control_nodelet.so' in maps or actual!=build['library_sha256']:
        raise ValueError('LOADED_CONTROLLER_LIBRARY_IDENTITY')
    return dict(verified=True,pid=pid,nonce=nonce,library=str(library),library_sha256=actual,
                plugin=build['plugin'],publishers={k:pubs[k] for k in expected})
def main():
    if Path.cwd().resolve()!=ROOT:raise ValueError('EXPECTED_REPOSITORY_CWD')
    from experiments.stop_latency_fix_v1.manual import main as shared
    shared(config_root=OUT/'runtime',binding_name='SOURCE_BINDING.json',audit_build=OUT)
if __name__=='__main__':main()
