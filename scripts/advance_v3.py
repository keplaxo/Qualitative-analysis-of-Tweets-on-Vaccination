"""Fail-closed, hourly private Kaggle v2 diagnosis and v3 intention development.

Stages: prior v2 failure audit (CPU), dual-head 16-case preflight (GPU),
dual-head 61-case DEVELOPMENT comparison (GPU), stop for independent data.
Never auto retries ERROR/CANCELLED/UNKNOWN. One submission per tick.
"""
import json,os,pathlib,re,subprocess,sys,tempfile
STAGES=[
 {"name":"v2_failure_audit","ref":"keplaxo/vax-v2-intent-failure-audit-20261018",
  "title":"Vax V2 Intent Failure Audit 20261018",
  "file":"kaggle/v2_intent_failure_audit.py","inputs":["keplaxo/vax-v2-intent-proxy-20261018"],
  "gpu":False,"mode":None,"timeout":1800},
 {"name":"v3_pilot","ref":"keplaxo/vax-intention-v3-pilot-20261018",
  "title":"Vax Intention V3 Pilot 20261018",
  "file":"kaggle/v3_intention_development.py","inputs":["keplaxo/vax-v2-intent-proxy-20261018"],
  "gpu":True,"mode":"pilot","timeout":5400},
 {"name":"v3_full","ref":"keplaxo/vax-intention-v3-full-20261018",
  "title":"Vax Intention V3 Full 20261018",
  "file":"kaggle/v3_intention_development.py",
  "inputs":["keplaxo/vax-v2-intent-proxy-20261018",
            "keplaxo/vax-intention-v3-pilot-20261018"],
  "gpu":True,"mode":"full","timeout":10800},
]
def call(*args,timeout=120):
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)
def state(ref):
    r=call("kernels","status",ref)
    if r.returncode:return "MISSING_OR_INACCESSIBLE"
    m=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",r.stdout)
    return m[-1] if m else "UNKNOWN"
def report(ref,name):
    with tempfile.TemporaryDirectory() as tmp:
        r=call("kernels","output",ref,"-p",tmp,
               "--file-pattern",name,timeout=180)
        files=list(pathlib.Path(tmp).rglob(name))
        if r.returncode or len(files)!=1:raise RuntimeError("Private checkpoint unavailable "+name)
        return json.loads(files[0].read_text(encoding="utf-8"))
def verify_input():
    r="keplaxo/vax-v2-intent-proxy-20261018"
    if state(r)!="COMPLETE":raise RuntimeError("Frozen v2 proxy is not complete")
    p=report(r,"vacma_intent_evaluation.json")
    if (p.get("eligible_unique_tweets")!=61 or p.get("schema_valid")!=56 or
        p.get("reference_commit")!="2e30fac04ad3caece6ca361fde53ca828ef68543" or
        p.get("six_way_vaccine_stance_accuracy_measured") is not False):
        raise RuntimeError("Frozen v2 reference provenance changed")
def approve(index):
    stage=STAGES[index]
    if index==0:
        d=report(stage["ref"],"v2_intent_failure_audit.json")
        if d.get("n")!=61 or d.get("model_accuracy_newly_estimated") is not False:
            raise RuntimeError("CPU failure audit incomplete")
    else:
        d=report(stage["ref"],"v3_development_report.json")
        expected,minimum,mode=(16,12,"pilot") if index==1 else (61,46,"full")
        if (d.get("mode")!=mode or d.get("total_n")!=expected or
            d.get("valid_schema_n",0)<minimum or
            d.get("development_not_independent") is not True or
            d.get("six_way_historical_stance_accuracy_measured") is not False or
            d.get("original_v2_outputs_unmodified") is not True):
            raise RuntimeError("V3 output gate failed: "+mode)
    return d
def submit(stage):
    owned=call("kernels","list","--mine","--csv")
    if owned.returncode:raise RuntimeError("Kaggle owned notebook list unavailable")
    if stage["ref"].lower() in owned.stdout.lower():
        raise RuntimeError("Existing/private notebook registered, refusing duplicate")
    source=pathlib.Path(stage["file"]).read_text(encoding="utf-8")
    if stage["mode"]:
        needle='MODE="MODE_PLACEHOLDER"'
        if source.count(needle)!=1:raise RuntimeError("V3 mode contract broken")
        source=source.replace(needle,'MODE="'+stage["mode"]+'"')
    with tempfile.TemporaryDirectory() as t:
        p=pathlib.Path(t)
        name=pathlib.Path(stage["file"]).name
        (p/name).write_text(source,encoding="utf-8")
        metadata={"id":stage["ref"],"title":stage["title"],
                  "code_file":name,"language":"python","kernel_type":"script",
                  "is_private":True,"enable_gpu":stage["gpu"],"enable_internet":True,
                  "dataset_sources":[],"kernel_sources":stage["inputs"],
                  "competition_sources":[],"model_sources":[]}
        (p/"kernel-metadata.json").write_text(json.dumps(metadata),encoding="utf-8")
        r=call("kernels","push","-p",str(p),"--timeout",str(stage["timeout"]),timeout=240)
        if r.returncode:
            response=(r.stderr+" "+r.stdout).lower()
            reason="conflict" if "409" in response else "quota" if "quota" in response else "other"
            raise RuntimeError("Kaggle submission refused category="+reason)
    print("V3_STAGE_SUBMITTED",stage["name"],stage["ref"],
          "gpu="+str(stage["gpu"]),flush=True)
def main():
    if not(os.getenv("KAGGLE_API_TOKEN") or
           (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("Missing Kaggle credentials")
    if pathlib.Path("kaggle/v3-enable.marker").read_text().strip()!="enabled":
        print("V3_DISABLED");return
    verify_input()
    states={s["name"]:state(s["ref"]) for s in STAGES}
    print("V3_PIPELINE_STATUS",json.dumps(states,sort_keys=True),flush=True)
    for index,stage in enumerate(STAGES):
        st=states[stage["name"]]
        if st=="COMPLETE":
            final=approve(index)
            if index==2:
                print("V3_DEVELOPMENT_COMPLETE",json.dumps({
                    "n":final["total_n"],
                    "valid":final["valid_schema_n"],
                    "strict_development_score":final["strict_intent_proxy_score_including_abstentions"],
                    "independent_accuracy_available":False,
                    "full_corpus_authorized":False
                }),flush=True)
            continue
        if st=="MISSING_OR_INACCESSIBLE":
            submit(stage)
        elif st in ("ERROR","FAILED","CANCELLED","CANCELED","CANCEL_ACKNOWLEDGED"):
            print("V3_BLOCKED",stage["name"],st,"manual investigation required; no retry")
        else:
            print("V3_WAITING",stage["name"],st)
        break
if __name__=="__main__":
    try:main()
    except Exception as e:
        print("V3_CONTROLLER_BLOCKED",type(e).__name__,str(e),file=sys.stderr)
        sys.exit(1)
