"""Kaggle full-corpus GPU orchestrator: CPU index -> pilot -> 64+64 shards.

All model predictions remain in PRIVATE Kaggle notebooks. The GitHub log
receives aggregate status only. This intentionally does not deploy unvalidated
Qwen vaccine stance or publish full-corpus stance prevalence as truth.

A per-shard capacity estimate from the 1,000-record three-model pilot
prevents submitting jobs that exceed a conservative Kaggle execution window.
One new GPU job maximum per scheduler iteration; failure/cancellation stalls.
"""
import json,math,os,pathlib,re,subprocess,sys,tempfile
INDEX="keplaxo/vax-full-corpus-index-20261019"
PILOT="keplaxo/vax-corpus-capacity-20261019"
COUNT=64
MAX_SECONDS=14500
SLUG="20261019"
def command(*args,timeout=110):
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)
def state(ref):
    p=command("kernels","status",ref)
    if p.returncode:return "NOT_AVAILABLE"
    a=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout)
    return a[-1] if a else "UNKNOWN"
def read_summary(ref,filename):
    with tempfile.TemporaryDirectory() as tmp:
        p=command("kernels","output",ref,"-p",tmp,
                  "--file-pattern",filename,timeout=180)
        matches=list(pathlib.Path(tmp).rglob(filename))
        if p.returncode or len(matches)!=1:
            raise RuntimeError("Completed private Kaggle summary unavailable: "+filename)
        return json.loads(matches[0].read_text(encoding="utf-8"))
def make_ref(which,index):
    return "keplaxo/vax-corpus-"+which+"-%02d-"%index+SLUG
def manifest():
    d=read_summary(INDEX,"full_corpus_manifest.json")
    counts=d.get("shard_unique_counts",{})
    if (d.get("shard_count")!=COUNT or
        d.get("sha256_exact_text_dedup") is not True or
        d.get("input_dataset")!="keplaxo/twitter-vaccination-dataset" or
        len(counts)!=COUNT or
        sum(counts.values())!=d.get("unique_texts") or
        d.get("raw_rows_total",0)<d.get("unique_texts",0) or
        d.get("unique_texts",0)<2000):
        raise RuntimeError("Index shard manifest failed counts/privacy integrity")
    return d
def pilot_ok():
    d=read_summary(PILOT,"corpus_capacity_probe.json")
    if (d.get("mode")!="pilot" or d.get("shard")!=0 or
        d.get("input_unique_n")!=1000 or
        d.get("coverage_complete") is not True or
        d.get("accuracy_measured") is not False or
        d.get("stance_validated") is not False):
        raise RuntimeError("Unverified CPU/index/GPU model capacity probe")
    for key,n in [("multilingual_sentiment",1000),("english_sentiment",d["english_screened_n"]),("deberta_stance",d["english_screened_n"])]:
        x=d["n_model_preds"].get(key,{})
        if x.get("predicted")!=n or x.get("output_hash_integrity_passed") is not True:
            raise RuntimeError("Pilot "+key+" output coverage failed")
    if any(float(d["runtime_seconds"].get(v,0))<=0 for v in
           ("xlmt_seconds","roberta_seconds","deberta_seconds")):
        raise RuntimeError("Missing complete model runtime signals")
    return d
def estimate(stage,n,d):
    p=d["input_unique_n"]
    ratio=d["english_screened_n"]/p
    time=d["runtime_seconds"]
    if stage=="sentiment":
        sec=(float(time["xlmt_seconds"])+float(time["roberta_seconds"]))*n/p
    else:
        sec=float(time["deberta_seconds"])*n/p
    # Model warmup, Kaggle variability, and skew in text lengths.
    return round(600+2*sec,1)
def validate_stage(stage,index,expected):
    ref=make_ref(stage,index)
    d=read_summary(ref,"corpus_shard_report.json")
    if (d.get("mode")!=stage or d.get("shard")!=index or
        d.get("input_unique_n")!=expected or
        d.get("coverage_complete") is not True or
        d.get("accuracy_measured") is not False or
        d.get("stance_validated") is not False or
        d.get("source_text_private") is not True):
        raise RuntimeError("Full-corpus "+stage+" "+str(index)+" output gate failed")
    keys=("multilingual_sentiment","english_sentiment") if stage=="sentiment" else ("deberta_stance",)
    if not all(d.get("n_model_preds",{}).get(k,{}).get("output_hash_integrity_passed") is True for k in keys):
        raise RuntimeError("Shard model key lineage mismatch")
def submit(mode,index):
    if mode not in ("pilot","sentiment","stance") or index not in range(COUNT):
        raise RuntimeError("Invalid controlled corpus stage")
    ref=PILOT if mode=="pilot" else make_ref(mode,index)
    owned=command("kernels","list","--mine","--csv")
    if owned.returncode:raise RuntimeError("Private notebook registry unavailable")
    if ref.lower() in owned.stdout.lower():
        raise RuntimeError("Kernel already registered; refusing duplicate")
    source=pathlib.Path("kaggle/full_corpus_gpu_stage.py").read_text(encoding="utf-8")
    if source.count('MODE="MODE_PLACEHOLDER"')!=1 or source.count('SHARD_INDEX=SHARD_PLACEHOLDER')!=1:
        raise RuntimeError("Corpus mode/shard injection contract broken")
    source=source.replace('MODE="MODE_PLACEHOLDER"','MODE="'+mode+'"')
    source=source.replace("SHARD_INDEX=SHARD_PLACEHOLDER","SHARD_INDEX="+str(index))
    inputs=[INDEX]
    if mode=="stance":inputs.append(make_ref("sentiment",index))
    if mode!="pilot" and index==0:inputs.append(PILOT)
    title=("Vax Corpus Capacity 20261019" if mode=="pilot" else
           "Vax Corpus "+mode.title()+" %02d 20261019"%index)
    with tempfile.TemporaryDirectory() as td:
        root=pathlib.Path(td)
        filename="full_corpus_gpu_stage.py"
        (root/filename).write_text(source,encoding="utf-8")
        meta={"id":ref,"title":title,"code_file":filename,
              "language":"python","kernel_type":"script","is_private":True,
              "enable_gpu":True,"enable_internet":True,
              "dataset_sources":[],"kernel_sources":inputs,
              "competition_sources":[],"model_sources":[]}
        (root/"kernel-metadata.json").write_text(json.dumps(meta),encoding="utf-8")
        p=command("kernels","push","-p",str(root),
                  "--timeout","18000",timeout=240)
        if p.returncode:
            detail=(p.stdout+" "+p.stderr).lower()
            category="quota" if "quota" in detail else "conflict" if "409" in detail else "other"
            raise RuntimeError("Kaggle corpus GPU stage rejected: "+category)
    print("FULL_CORPUS_GPU_SUBMITTED",mode,index,ref,flush=True)
def main():
    if not (os.getenv("KAGGLE_API_TOKEN") or
            (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("Missing Kaggle auth")
    if pathlib.Path("kaggle/full-corpus-enable.marker").read_text().strip()!="enabled":
        print("FULL_CORPUS_DISABLED");return
    status=state(INDEX)
    print("FULL_CORPUS_INDEX_STATE",status,flush=True)
    if status!="COMPLETE":
        print("FULL_CORPUS_WAIT_INDEX" if status in ("QUEUED","RUNNING","PENDING","STARTING") else
              "FULL_CORPUS_BLOCKED_INDEX",status,flush=True)
        return
    m=manifest()
    print("FULL_CORPUS_MANIFEST",json.dumps({
        "raw_rows":m["raw_rows_total"],
        "unique_texts":m["unique_texts"],"shards":m["shard_count"],
        "duplicates":m["duplicate_text_rows"]}),flush=True)
    ps=state(PILOT)
    if ps!="COMPLETE":
        if ps=="NOT_AVAILABLE":submit("pilot",0)
        elif ps in ("QUEUED","RUNNING","STARTING","PENDING"):
            print("FULL_CORPUS_WAIT_CAPACITY_PROBE",ps,flush=True)
        else:print("FULL_CORPUS_BLOCKED_CAPACITY_PROBE",ps,flush=True)
        return
    p=pilot_ok()
    count=m["shard_unique_counts"]
    maxes={stage:max(estimate(stage,count["%02d"%i],p) for i in range(COUNT))
           for stage in ("sentiment","stance")}
    print("FULL_CORPUS_ESTIMATES_SECONDS",json.dumps(maxes,sort_keys=True),flush=True)
    for stage in ("sentiment","stance"):
        if maxes[stage]>MAX_SECONDS:
            print("FULL_CORPUS_CAPACITY_BLOCKED",stage,
                  "estimated_shard_exceeds_14500_seconds",
                  "requires smaller shards / alternative compute",flush=True)
            return
    complete={"sentiment":0,"stance":0}
    for stage in ("sentiment","stance"):
        for i in range(COUNT):
            ref=make_ref(stage,i)
            s=state(ref)
            if s=="COMPLETE":
                validate_stage(stage,i,count["%02d"%i])
                complete[stage]+=1
                continue
            print("FULL_CORPUS_PROGRESS",json.dumps(complete),flush=True)
            if s=="NOT_AVAILABLE":submit(stage,i)
            elif s in ("QUEUED","RUNNING","STARTING","PENDING"):
                print("FULL_CORPUS_WAIT_SHARD",stage,i,s,flush=True)
            else:
                print("FULL_CORPUS_BLOCKED_SHARD",stage,i,s,
                      "no automatic retry",flush=True)
            return
    print("FULL_CORPUS_INFERENCE_COMPLETE",json.dumps({
        "unique_texts":m["unique_texts"],
        "sentiment_shards":64,"stance_shards":64,
        "stance_accuracy_not_established":True,
        "raw_event_prevalence_not_established":True
    }),flush=True)
if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("FULL_CORPUS_CONTROLLER_BLOCKED",type(exc).__name__,str(exc),
              file=sys.stderr)
        sys.exit(1)
