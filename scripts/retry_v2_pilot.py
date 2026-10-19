"""One-time explicit recovery for failed private Kaggle v2 pilot.

Guarded resubmission to SAME private notebook slug; refuses any run unless
the previous execution is ERROR and the predecessor artifacts pass checks.
No repeat is allowed if QUEUED/RUNNING/COMPLETE. No raw tweet is printed.
"""
import json,os,pathlib,subprocess,sys,tempfile,time
from advance_v2 import STAGES,kaggle,state,check_predecessors

def main():
    if not (os.environ.get("KAGGLE_API_TOKEN") or
            (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        raise RuntimeError("Missing Kaggle API secret")
    step=STAGES[0]
    current=state(step["ref"])
    print("V2_RETRY_PRECONDITION",current,flush=True)
    if current!="ERROR":
        print("NO_RETRY: only the documented ERROR pilot may be repaired")
        return
    check_predecessors()
    content=pathlib.Path(step["file"]).read_text(encoding="utf-8")
    needle='MODE="MODE_PLACEHOLDER"'
    if content.count(needle)!=1:
        raise RuntimeError("V2 pilot mode contract changed")
    if 'sys.path.insert(0,str(src_dir))' not in content:
        raise RuntimeError("Missing reviewed in-process module path repair")
    content=content.replace(needle,'MODE="pilot"')
    with tempfile.TemporaryDirectory() as tmp:
        d=pathlib.Path(tmp)
        filename=pathlib.Path(step["file"]).name
        (d/filename).write_text(content,encoding="utf-8")
        metadata={
           "id":step["ref"],"title":step["title"],"code_file":filename,
           "language":"python","kernel_type":"script","is_private":True,
           "enable_gpu":True,"enable_internet":True,"dataset_sources":[],
           "kernel_sources":step["inputs"],"competition_sources":[],
           "model_sources":[]}
        (d/"kernel-metadata.json").write_text(json.dumps(metadata),encoding="utf-8")
        p=kaggle("kernels","push","-p",str(d),"--timeout",
                 str(step["timeout"]),timeout=240)
        if p.returncode:
            raw=(p.stderr+" "+p.stdout).lower()
            reason=("409_conflict" if "409" in raw else
                    "quota" if "quota" in raw else
                    "auth" if "403" in raw or "401" in raw else "other")
            raise RuntimeError("Kaggle v2 pilot recovery submission failed: "+reason)
    print("V2_PILOT_REPAIRED_VERSION_SUBMITTED",step["ref"],flush=True)
    time.sleep(12)
    print("V2_PILOT_NEW_STATE",state(step["ref"]),flush=True)

if __name__=="__main__":
    try:main()
    except Exception as err:
        print("V2_RECOVERY_BLOCKED:"+type(err).__name__+":"+str(err),
              file=sys.stderr)
        sys.exit(1)
