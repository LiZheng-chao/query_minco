"""Exclusive run provenance and flock-protected attempt ledger; no training logic."""
import hashlib,json,os,signal,subprocess,time,fcntl,uuid
from pathlib import Path
COMPONENT_ID="exclusive_process_provenance_and_atomic_ledger_v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def exclusive_json(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def proc_tree(root):
    records={}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():continue
        try:
            fields=(d/"stat").read_text().rsplit(")",1)[1].split()
            records[int(d.name)]=(int(fields[1]),int(fields[2]),int(fields[3]),int(fields[21])*os.sysconf("SC_PAGE_SIZE"),float(fields[11])+float(fields[12]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    ids={i for i,v in records.items() if i==root or v[1]==root or v[2]==root}
    while True:
        more={i for i,v in records.items() if v[0] in ids}
        if more<=ids:break
        ids|=more
    return dict(processes=len(ids),rss_bytes=sum(records[i][3] for i in ids),cpu_ticks=sum(records[i][4] for i in ids))
def run_logged(argv,output,*,cwd,env=None,runspec=None,approval=None,timeout_s=300):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.mkdir()
    actual_env=dict(os.environ) if env is None else dict(env)
    recorded={};redacted=[]
    for key,value in actual_env.items():
        if any(t in key.upper() for t in ["TOKEN","PASSWORD","SECRET","CREDENTIAL","API_KEY"]):
            recorded[key]=dict(redacted=True,value_sha256=hashlib.sha256(value.encode()).hexdigest());redacted.append(key)
        else:recorded[key]=value
    copied={}
    for name,source in [("RUNSPEC",runspec),("APPROVAL",approval)]:
        if source is not None:
            raw=Path(source).read_bytes();target=output/(name+".json")
            with target.open("xb") as f:f.write(raw)
            copied[name]=dict(original_path=str(Path(source).resolve()),copy_path=str(target),sha256=hashlib.sha256(raw).hexdigest())
    before=dict(component_id=COMPONENT_ID,argv=[str(x) for x in argv],cwd=str(Path(cwd).resolve()),environment=recorded,redacted_keys=redacted,inputs=copied,started_unix=time.time())
    exclusive_json(output/"PROCESS_START.json",before)
    started=time.monotonic();peak=0;max_processes=0;samples=[];timed_out=False
    with (output/"stdout.log").open("xb") as stdout,(output/"stderr.log").open("xb") as stderr:
        try:
            child=subprocess.Popen(before["argv"],cwd=cwd,env=actual_env,stdout=stdout,stderr=stderr,start_new_session=True)
        except BaseException as error:
            exclusive_json(output/"PROCESS_LAUNCH_ERROR.json",dict(error=repr(error)));raise
        while child.poll() is None:
            q=proc_tree(child.pid);q["elapsed_s"]=time.monotonic()-started
            samples.append(q);peak=max(peak,q["rss_bytes"]);max_processes=max(max_processes,q["processes"])
            if q["elapsed_s"]>timeout_s:
                timed_out=True
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                break
            time.sleep(.02)
        code=child.wait()
    result=dict(**before,pid=child.pid,exit_code=code,timed_out=timed_out,elapsed_s=time.monotonic()-started,
                sampled_process_tree_peak_rss_bytes=peak,max_sampled_processes=max_processes,
                sample_interval_s=.02,samples=samples,stdout_sha256=sha(output/"stdout.log"),stderr_sha256=sha(output/"stderr.log"),
                caveat="Resource samples are not continuous peaks; secrets redacted with SHA. No historical run reconstruction.")
    exclusive_json(output/"PROCESS_RESULT.json",result)
    return result

class AtomicAttemptLedger:
    def __init__(self,path,budget):
        self.path=Path(path);self.budget=budget
        if type(budget) is not int or budget<1:raise ValueError("BUDGET")
        self.path.parent.mkdir(parents=True,exist_ok=True)
    def _change(self,operation,token=None):
        lock=self.path.with_suffix(self.path.suffix+".lock")
        with lock.open("a") as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            state=json.loads(self.path.read_text()) if self.path.exists() else dict(budget=self.budget,attempted=0,completed=0,pending=[],done=[])
            if state["budget"]!=self.budget:raise ValueError("LEDGER_BUDGET_MISMATCH")
            if operation=="reserve":
                if state["attempted"]>=self.budget:raise PermissionError("BUDGET_EXHAUSTED")
                token=uuid.uuid4().hex;state["attempted"]+=1;state["pending"].append(token)
            else:
                if token not in state["pending"]:raise ValueError("UNRESERVED_OR_DUPLICATE_COMPLETE")
                state["pending"].remove(token);state["done"].append(token);state["completed"]+=1
            temporary=self.path.with_name(self.path.name+"."+uuid.uuid4().hex+".tmp")
            with temporary.open("x") as out:json.dump(state,out);out.flush();os.fsync(out.fileno())
            os.replace(temporary,self.path)
            fd=os.open(str(self.path.parent),os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            return token,state
    def reserve(self):return self._change("reserve")[0]
    def complete(self,token):return self._change("complete",token)[1]

def main(argv=None):
    import argparse,sys
    parser=argparse.ArgumentParser(description="Exclusive child-process provenance; no training is started by default")
    parser.add_argument("--output",required=True)
    parser.add_argument("--cwd",default=".")
    parser.add_argument("--runspec")
    parser.add_argument("--approval")
    parser.add_argument("--timeout-s",type=float,default=300.)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    if not command:parser.error("explicit child argv required after --")
    result=run_logged(command,args.output,cwd=args.cwd,runspec=args.runspec,approval=args.approval,timeout_s=args.timeout_s)
    print(json.dumps(dict(exit_code=result["exit_code"],output=args.output)))
    return result["exit_code"] if result["exit_code"]>=0 else 128-result["exit_code"]
if __name__=="__main__":
    raise SystemExit(main())
