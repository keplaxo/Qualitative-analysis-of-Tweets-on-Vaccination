"""Kaggle full-corpus GPU orchestrator: CPU index -> pilot -> 64+64 shards.

All model predictions remain in PRIVATE Kaggle notebooks. The GitHub log
receives aggregate status only. This intentionally does not deploy unvalidated
Qwen vaccine stance or publish full-corpus stance prevalence as truth.

A per-shard capacity estimate from the 1,000-record three-model pilot
prevents submitting jobs that exceed a conservative Kaggle execution window.
One new GPU job maximum per scheduler iteration; failure/cancellation stalls.
"""
import json,math,os,pathlib,re,subprocess,sys,tempfile,time
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
# Kaggle CLI --file-pattern is a regular expression applied to its output
# path (often vax_outputs/<filename>), not an exact basename.
REPORT_FILES=frozenset((
    "full_corpus_manifest.json","corpus_capacity_probe.json",
    "corpus_shard_report.json","full_corpus_occurrence_manifest.json",
    "corpus_report_wave.json","full_corpus_research_report.json"
))
def read_summary(ref,filename):
    if filename not in REPORT_FILES:
        raise RuntimeError("Unapproved Kaggle summary filename")
    pattern=r".*"+re.escape(filename)+r"$"
    with tempfile.TemporaryDirectory() as tmp:
        status=None
        matches=[]
        for attempt in range(3):
            status=command("kernels","output",ref,"-p",tmp,
                           "--file-pattern",pattern,timeout=180)
            matches=list(pathlib.Path(tmp).rglob(filename))
            if status.returncode==0 and len(matches)==1:
                return json.loads(matches[0].read_text(encoding="utf-8"))
            if attempt<2:time.sleep(2)
        # Diagnostics disclose only filenames/availability, never Kaggle stdout,
        # private tweet text, prediction rows, or authentication material.
        names=command("kernels","files",ref,"--csv","--page-size","200")
        listed=(names.returncode==0 and filename in names.stdout)
        raise RuntimeError(
            "Completed private Kaggle summary unavailable: "+ref+"/"+filename+
            " report_listed="+str(listed)+
            " matched_files="+str(len(matches))+
            " retrieval_returncode="+str(status.returncode))
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
EVENTS="keplaxo/vax-corpus-event-join-20261019-v2"
FINAL_REF="keplaxo/vax-full-corpus-research-report-20261019"
WAVES=16

def wave_ref(index):
    return "keplaxo/vax-corpus-report-wave-%02d-"%index+SLUG

def event_weights_ok(m):
    r=read_summary(EVENTS,"full_corpus_occurrence_manifest.json")
    if (r.get("unique_texts")!=m["unique_texts"] or
        r.get("total_nonempty_events")!=m["raw_rows_total"]-m["blank_rows"] or
        r.get("exact_sha256_text_join") is not True or
        r.get("future_aggregation_can_report_event_weighted_counts") is not True):
        raise RuntimeError("Raw event backfill coverage incomplete")
    return r

def wave_ok(index):
    r=read_summary(wave_ref(index),"corpus_report_wave.json")
    if (r.get("wave")!=index or
        r.get("shards")!=list(range(index*4,(index+1)*4)) or
        r.get("entire_wave_hash_coverage_verified") is not True or
        r.get("tweet_content_or_hashes_exported") is not False or
        r.get("stance_unvalidated") is not True):
        raise RuntimeError("Corrupt or missing private corpus aggregation wave")
    return r

def submit_aggregation(which,index=0):
    if which=="wave":
        name=wave_ref(index)
        title="Vax Corpus Report Wave %02d 20261019"%index
        path="kaggle/full_corpus_aggregate_wave.py"
        script=pathlib.Path(path).read_text(encoding="utf-8")
        needle="WAVE_INDEX=WAVE_PLACEHOLDER"
        if script.count(needle)!=1:raise RuntimeError("Wave parameter not injectible")
        script=script.replace(needle,"WAVE_INDEX="+str(index))
        sources=[EVENTS]
        for i in range(index*4,(index+1)*4):
            sources += [make_ref("sentiment",i),make_ref("stance",i)]
    elif which=="final":
        name=FINAL_REF
        title="Vax Full Corpus Research Report 20261019"
        path="kaggle/full_corpus_aggregate_final.py"
        script=pathlib.Path(path).read_text(encoding="utf-8")
        sources=[INDEX,EVENTS]+[wave_ref(i) for i in range(WAVES)]
    else:raise RuntimeError("Unknown aggregation job")
    owned=command("kernels","list","--mine","--csv")
    if owned.returncode or name.lower() in owned.stdout.lower():
        raise RuntimeError("Ambiguous/registered aggregation kernel; no duplicate")
    with tempfile.TemporaryDirectory() as tmp:
        d=pathlib.Path(tmp);filename=pathlib.Path(path).name
        (d/filename).write_text(script,encoding="utf-8")
        meta={"id":name,"title":title,"code_file":filename,
              "language":"python","kernel_type":"script","is_private":True,
              "enable_gpu":False,"enable_internet":False,"dataset_sources":[],
              "kernel_sources":sources,"competition_sources":[],"model_sources":[]}
        (d/"kernel-metadata.json").write_text(json.dumps(meta),encoding="utf-8")
        result=command("kernels","push","-p",str(d),"--timeout","5400",timeout=240)
        if result.returncode:
            response=(result.stdout+" "+result.stderr).lower()
            reason="quota" if "quota" in response else "conflict" if "409" in response else "other"
            raise RuntimeError("Private corpus aggregate rejected "+reason)
    print("FULL_CORPUS_AGGREGATE_SUBMITTED",which,index,name,flush=True)

def finish_aggregation(m):
    st=state(EVENTS)
    if st!="COMPLETE":
        print("FULL_CORPUS_WAIT_EVENT_BACKFILL" if st in ("RUNNING","QUEUED","PENDING","STARTING")
              else "FULL_CORPUS_BLOCKED_EVENT_BACKFILL",st,flush=True)
        return
    event_weights_ok(m)
    for i in range(WAVES):
        st=state(wave_ref(i))
        if st=="COMPLETE":
            wave_ok(i)
            continue
        if st=="NOT_AVAILABLE":submit_aggregation("wave",i)
        elif st in ("RUNNING","QUEUED","PENDING","STARTING"):
            print("FULL_CORPUS_WAIT_AGGREGATE",i,st,flush=True)
        else:print("FULL_CORPUS_BLOCKED_AGGREGATE",i,st,flush=True)
        return
    state_final=state(FINAL_REF)
    if state_final=="COMPLETE":
        r=read_summary(FINAL_REF,"full_corpus_research_report.json")
        if (r.get("n_sha256_unique_texts")!=m["unique_texts"] or
            r.get("n_original_source_rows")!=m["raw_rows_total"] or
            r.get("shards_covered")!=64 or
            r.get("accuracy_measured") is not False or
            r.get("stance_prevalence_validated") is not False):
            raise RuntimeError("Final full corpus research report failed validity checks")
        print("FULL_CORPUS_RESEARCH_FINISHED",json.dumps({
            "raw_rows":r["n_original_source_rows"],
            "unique_texts":r["n_sha256_unique_texts"],
            "sentiment_shards":64,"stance_shards":64,
            "accuracy_established":False
        }),flush=True)
    elif state_final=="NOT_AVAILABLE":submit_aggregation("final")
    elif state_final in ("RUNNING","QUEUED","STARTING","PENDING"):
        print("FULL_CORPUS_WAIT_FINAL_REPORT",state_final,flush=True)
    else:print("FULL_CORPUS_BLOCKED_FINAL_REPORT",state_final,flush=True)

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
    finish_aggregation(m)
if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("FULL_CORPUS_CONTROLLER_BLOCKED",type(exc).__name__,str(exc),
              file=sys.stderr)
        sys.exit(1)
