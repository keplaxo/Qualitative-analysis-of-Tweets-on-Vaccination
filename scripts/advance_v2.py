"""Gated private v2 research pipeline: 12-case GPU -> 60-case GPU -> CPU schema.

On each invocation submit at most one missing notebook, never retry queued,
canceled, failed, or unrecognized stages. This script does not modify old
datasets or Kaggle kernels and does not authorize full-corpus inference.
"""
import json,os,pathlib,re,subprocess,sys,tempfile
from datetime import datetime,timezone

STAGES=[
 {"name":"v2_pilot","ref":"keplaxo/vax-stance-v2-pilot-20261018",
  "title":"Vax Stance V2 Pilot 20261018","file":"kaggle/v2_experiment.py",
  "inputs":["keplaxo/vax-sentiment-benchmark-full","keplaxo/vax-qwen-300"],
  "gpu":True,"mode":"pilot","timeout":3600},
 {"name":"v2_full","ref":"keplaxo/vax-stance-v2-full-20261018",
  "title":"Vax Stance V2 Full 20261018","file":"kaggle/v2_experiment.py",
  "inputs":["keplaxo/vax-sentiment-benchmark-full","keplaxo/vax-qwen-300",
            "keplaxo/vax-stance-v2-pilot-20261018"],
  "gpu":True,"mode":"full","timeout":9000},
 {"name":"external_schema","ref":"keplaxo/vax-external-reference-probe-20261018",
  "title":"Vax External Reference Probe 20261018",
  "file":"kaggle/external_benchmark_probe.py",
  "inputs":[],"gpu":False,"mode":None,"timeout":1200},
 {"name":"external_intent_proxy","ref":"keplaxo/vax-v2-intent-proxy-20261018",
  "title":"Vax V2 Intent Proxy 20261018",
  "file":"kaggle/vacma_intent_evaluation.py",
  "inputs":[],"gpu":True,"mode":None,"timeout":7200},
]
def kaggle(*args,timeout=120):
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)

def state(ref):
    p=kaggle("kernels","status",ref)
    if p.returncode:return "MISSING_OR_INACCESSIBLE"
    m=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout)
    return m[-1] if m else "UNKNOWN"

def approved_file(ref,filename):
    with tempfile.TemporaryDirectory() as tmp:
        p=kaggle("kernels","output",ref,"-p",tmp,
                 "--file-pattern",filename,timeout=150)
        found=list(pathlib.Path(tmp).rglob(filename))
        if p.returncode or len(found)!=1:
            raise RuntimeError("Prior private stage output unavailable: "+filename)
        return json.loads(found[0].read_text(encoding="utf-8"))

def check_predecessors():
    if state("keplaxo/vax-sentiment-benchmark-full")!="COMPLETE":
        raise RuntimeError("Frozen 2k benchmark not complete")
    base=approved_file("keplaxo/vax-sentiment-benchmark-full","summary.json")
    if base.get("n_sampled")!=2000 or base.get("stage_coverage",{}).get("zero_shot_stance")!=1666:
        raise RuntimeError("Frozen 2k prior report invalid")
    if state("keplaxo/vax-qwen-300")!="COMPLETE":
        raise RuntimeError("Frozen 300 Qwen comparison not complete")
    q=approved_file("keplaxo/vax-qwen-300","qwen_validation.json")
    if q.get("processed")!=300 or q.get("schema_valid_count")!=290:
        raise RuntimeError("Frozen Qwen prior report invalid")

def check_pilot():
    r=approved_file(STAGES[0]["ref"],"v2_aggregate.json")
    if (r.get("mode")!="pilot" or r.get("evaluated")!=12 or
        r.get("schema_valid_count",0)<9 or
        r.get("key_integrity") is not True or
        r.get("reference_accuracy_measured") is not False):
        raise RuntimeError("V2 pilot failed preregistered execution gate")
    return r

def check_full():
    r=approved_file(STAGES[1]["ref"],"v2_aggregate.json")
    if (r.get("mode")!="full" or r.get("evaluated")!=60 or
        r.get("schema_valid_count",0)<45 or
        r.get("key_integrity") is not True or
        r.get("reference_accuracy_measured") is not False):
        raise RuntimeError("V2 60-case comparison failed execution gate")
    return r

def submit(stage):
    # Avoid restarting an existing notebook, including inaccessible/transient
    # kernels which may still be registered (Kaggle API eventual consistency).
    l=kaggle("kernels","list","--mine","--csv")
    if l.returncode:
        raise RuntimeError("Cannot safely list owned kernels")
    if stage["ref"].lower() in l.stdout.lower():
        raise RuntimeError("Notebook registered; cannot auto resubmit")
    source=pathlib.Path(stage["file"]).read_text(encoding="utf-8")
    if stage["mode"] is not None:
        key='MODE="MODE_PLACEHOLDER"'
        if source.count(key)!=1:raise RuntimeError("V2 mode injection missing")
        source=source.replace(key,'MODE="'+stage["mode"]+'"')
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp)
        file=pathlib.Path(stage["file"]).name
        (root/file).write_text(source,encoding="utf-8")
        metadata={
           "id":stage["ref"],"title":stage["title"],
           "code_file":file,"language":"python","kernel_type":"script",
           "is_private":True,"enable_gpu":bool(stage["gpu"]),
           "enable_internet":True,
           "dataset_sources":[],"kernel_sources":stage["inputs"],
           "competition_sources":[],"model_sources":[]
        }
        (root/"kernel-metadata.json").write_text(json.dumps(metadata),encoding="utf-8")
        p=kaggle("kernels","push","-p",str(root),
                 "--timeout",str(stage["timeout"]),timeout=240)
        if p.returncode:
            err=p.stderr.lower()+" "+p.stdout.lower()
            category=("409_conflict" if "409" in err else
                      "quota" if "quota" in err else
                      "authentication" if "401" in err or "403" in err else
                      "other")
            raise RuntimeError("Kaggle stage submission rejected: "+category)
    print("V2_STAGE_SUBMITTED",stage["name"],stage["ref"],
          "gpu="+str(stage["gpu"]),flush=True)

def main():
    if not (os.getenv("KAGGLE_API_TOKEN") or
            (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("GitHub Actions Kaggle secret missing")
    if pathlib.Path("kaggle/v2-enable.marker").read_text().strip()!="enabled":
        print("V2 controlled launch disabled")
        return
    check_predecessors()
    states={s["name"]:state(s["ref"]) for s in STAGES}
    print("V2_PIPELINE_STATUSES",json.dumps(states,sort_keys=True),flush=True)
    for i,step in enumerate(STAGES):
        current=states[step["name"]]
        if current=="COMPLETE":
            if i==0:check_pilot()
            elif i==1:check_full()
            elif i==2:
                result=approved_file(step["ref"],"external_benchmark_schema.json")
                if (result.get("doi")!="10.5281/zenodo.7601328" or
                    result.get("accuracy_evaluated") is not False or
                    "text" in result.get("files",{}).get("test",{}).get("columns",[])):
                    raise RuntimeError("Published VaxxHesitancy ID-only schema differs")
                print("ID_ONLY_REFERENCE_VERIFIED: no text rehydration attempted",flush=True)
            else:
                evaluation=approved_file(step["ref"],"vacma_intent_evaluation.json")
                if (evaluation.get("eligible_unique_tweets")!=61 or
                    evaluation.get("six_way_vaccine_stance_accuracy_measured") is not False or
                    evaluation.get("no_tweet_text_or_row_data_published") is not True):
                    raise RuntimeError("External intention proxy evaluation missing or unsafe")
                print("V2_DEVELOPMENT_PIPELINE_COMPLETE",json.dumps({
                    "versioned_60_case_analysis":True,
                    "public_id_only_reference_verified":True,
                    "independent_out_of_domain_intention_proxy_evaluated":True,
                    "exact_six_way_stance_accuracy_still_pending":True,
                    "full_corpus_not_authorized":True
                }),flush=True)
            continue
        if current=="MISSING_OR_INACCESSIBLE":
            submit(step)
        elif current in ("ERROR","CANCELLED","CANCELED","CANCEL_ACKNOWLEDGED","FAILED"):
            print("V2_BLOCKED",step["name"],current,
                  "Manual diagnosis required; no retry.",flush=True)
        else:
            print("V2_WAITING",step["name"],current,flush=True)
        break

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("V2_CONTROLLER_BLOCKED: "+type(e).__name__+": "+str(e),file=sys.stderr)
        sys.exit(1)
