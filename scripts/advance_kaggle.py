"""Idempotent hourly Kaggle research DAG scheduler. No automatic full-corpus launch.

Only starts the next private kernel after strictly verifying its upstream
completed artifacts. A QUEUED/RUNNING/ERROR/CANCELLED kernel is never retried
automatically. Exceptions fail closed, preventing unplanned GPU usage.
"""
import json, os, pathlib, re, subprocess, sys, tempfile, time
from datetime import datetime, timezone

OWNER="keplaxo"
SRC="keplaxo/vax-sentiment-benchmark-full"
STAGES=[
 {"name":"pilot","kernel":"keplaxo/vax-qwen-pilot",
  "title":"Vax Qwen Pilot","script":"kaggle/qwen_stage.py",
  "sources":[SRC],"gpu":True,"timeout":3600},
 {"name":"full","kernel":"keplaxo/vax-qwen-300",
  "title":"Vax Qwen 300","script":"kaggle/qwen_stage.py",
  "sources":[SRC,"keplaxo/vax-qwen-pilot"],"gpu":True,"timeout":18000},
 {"name":"report","kernel":"keplaxo/vax-semantic-research-report",
  "title":"Vax Semantic Research Report","script":"kaggle/report_stage.py",
  "sources":[SRC,"keplaxo/vax-qwen-300"],"gpu":False,"timeout":1800},
]
STATES_ACTIVE={"QUEUED","RUNNING","STARTING","CANCEL_ACKNOWLEDGED",
               "CANCEL_PENDING","PENDING"}
STATES_FINISHED={"COMPLETE","ERROR","FAILED","CANCELLED","CANCELED"}
def call(*args,timeout=100):
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)

def state(ref):
    p=call("kernels","status",ref)
    if p.returncode:
        return "NOT_AVAILABLE"
    m=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout)
    return m[-1] if m else "UNRECOGNIZED"

def outputs(ref,filename):
    with tempfile.TemporaryDirectory() as folder:
        p=call("kernels","output",ref,"-p",folder,
               "--file-pattern",filename,timeout=150)
        located=list(pathlib.Path(folder).rglob(filename))
        if p.returncode or len(located)!=1:
            raise RuntimeError("Expected exactly one approved summary output for "+ref)
        return json.loads(located[0].read_text(encoding="utf-8"))

def validated_baseline():
    if state(SRC)!="COMPLETE":
        raise RuntimeError("Prior 2,000-record benchmark is not COMPLETE")
    d=outputs(SRC,"summary.json")
    cover=d.get("stage_coverage",{})
    if (d.get("n_sampled")!=2000 or
        cover.get("multilingual_sentiment")!=2000 or
        cover.get("english_sentiment")!=1666 or
        cover.get("zero_shot_stance")!=1666):
        raise RuntimeError("2,000-record baseline output validation failed")
    return True

def approve_pilot():
    d=outputs("keplaxo/vax-qwen-pilot","qwen_validation.json")
    if (d.get("mode")!="pilot" or
        d.get("triage_pool")!=300 or
        d.get("processed")!=20 or
        d.get("schema_valid_count",0)<14):
        raise RuntimeError("Qwen pilot failed 70% structural-parse gate")
    return d

def approve_full():
    d=outputs("keplaxo/vax-qwen-300","qwen_validation.json")
    if (d.get("mode")!="full" or
        d.get("triage_pool")!=300 or
        d.get("processed")!=300 or
        d.get("schema_valid_count",0)<210):
        raise RuntimeError("300-case Qwen review failed output and schema gate")
    return d

def completed_report():
    d=outputs("keplaxo/vax-semantic-research-report","research_report.json")
    if (d.get("base_n")!=2000 or d.get("qwen_selected_n")!=300 or
        d.get("qwen_schema_valid_n",0)<210 or
        d.get("accuracy_measured") is not False):
        raise RuntimeError("Final aggregate report failed validity checks")
    return d

def existing():
    p=call("kernels","list","--mine","--csv")
    if p.returncode: raise RuntimeError("Cannot list owned Kaggle kernels; refusing to submit")
    return p.stdout.lower()

def submit(stage):
    # Never update or restart an extant notebook; only create a missing stage.
    if stage["kernel"].lower() in existing():
        raise RuntimeError("Kernel already listed; status uncertain, refusing duplicate submission")
    src=pathlib.Path(stage["script"]).read_text(encoding="utf-8")
    if stage["name"] in ("pilot","full"):
        placeholder='MODE = "MODE_PLACEHOLDER"'
        if src.count(placeholder)!=1: raise RuntimeError("Mode injection contract broken")
        src=src.replace(placeholder,'MODE = "'+stage["name"]+'"')
    kernel_type="script"
    with tempfile.TemporaryDirectory() as tmp:
        d=pathlib.Path(tmp)
        code_file=pathlib.Path(stage["script"]).name
        (d/code_file).write_text(src,encoding="utf-8")
        metadata={
         "id":stage["kernel"],"title":stage["title"],
         "code_file":code_file,"language":"python","kernel_type":kernel_type,
         "is_private":True,"enable_gpu":stage["gpu"],
         "enable_internet":bool(stage["gpu"]),
         "dataset_sources":[],"kernel_sources":stage["sources"],
         "competition_sources":[],"model_sources":[]}
        (d/"kernel-metadata.json").write_text(json.dumps(metadata),encoding="utf-8")
        p=call("kernels","push","-p",str(d),"--timeout",
               str(stage["timeout"]),timeout=300)
        if p.returncode:
            raise RuntimeError("Kaggle rejected "+stage["name"]+" submission (exit "+
                               str(p.returncode)+"). Check private Kaggle account.")
    print("SUBMITTED",stage["name"],stage["kernel"],
          "gpu="+str(stage["gpu"]),"timeout_seconds="+str(stage["timeout"]),flush=True)

def main():
    if not (os.getenv("KAGGLE_API_TOKEN") or
            (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("Kaggle GitHub secret missing")
    if pathlib.Path("kaggle/pipeline-enable.marker").read_text().strip() != "enabled":
        print("Disabled by feature-branch enable marker. No jobs submitted.")
        return
    validated_baseline()
    status={x["name"]:state(x["kernel"]) for x in STAGES}
    print("PIPELINE_STATUS",json.dumps(status,sort_keys=True),flush=True)
    next_stage=None
    if status["pilot"]!="COMPLETE":
        if status["pilot"]=="NOT_AVAILABLE":
            next_stage=STAGES[0]
        elif status["pilot"] not in STATES_ACTIVE:
            print("BLOCKED: pilot state "+status["pilot"]+"; manual inspection required")
    elif status["full"]!="COMPLETE":
        approve_pilot()
        if status["full"]=="NOT_AVAILABLE":
            next_stage=STAGES[1]
        elif status["full"] not in STATES_ACTIVE:
            print("BLOCKED: Qwen 300 state "+status["full"]+"; manual inspection required")
    elif status["report"]!="COMPLETE":
        approve_pilot()
        approve_full()
        if status["report"]=="NOT_AVAILABLE":
            next_stage=STAGES[2]
        elif status["report"] not in STATES_ACTIVE:
            print("BLOCKED: report state "+status["report"]+"; manual inspection required")
    else:
        report=completed_report()
        print("PIPELINE_COMPLETE",json.dumps({
          "baseline_n":report["base_n"],
          "qwen_n":report["qwen_selected_n"],
          "qwen_valid_n":report["qwen_schema_valid_n"],
          "paired_stance_agreement":report["qwen_deberta_stance_agreement_rate"],
          "no_accuracy_measured":True,
          "full_corpus_inference_not_automatically_authorized":True
        }))
    if next_stage:
        submit(next_stage)
    summ=os.getenv("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ,"a",encoding="utf-8") as f:
            f.write("## Vaccination research pipeline (UTC "+
                    datetime.now(timezone.utc).isoformat()+")\n\n")
            for stage in STAGES:
                f.write("- "+stage["name"]+": **"+status[stage["name"]]+"**\n")
            if next_stage:
                f.write("\nSubmitted **"+next_stage["name"]+"**; "+
                        "GPU job completion must be verified on a later poll.\n")
            f.write("\nOriginal tweet rows and confidential per-record results remain in private Kaggle notebooks.\n")

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("Pipeline blocked: "+type(e).__name__+": "+str(e),file=sys.stderr)
        sys.exit(1)
