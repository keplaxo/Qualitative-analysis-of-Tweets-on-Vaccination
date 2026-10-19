"""Private Kaggle full-corpus model stage: GPU capacity pilot or one hash shard.

The CPU index provides 64 disjoint SHA-256 text partitions. All source text,
record hashes and per-record inference outputs remain in private Kaggle kernels.
Stage outputs contain descriptive model predictions, NOT accuracy estimates.
"""
MODE="MODE_PLACEHOLDER"
SHARD_INDEX=SHARD_PLACEHOLDER
assert MODE in ("pilot","sentiment","stance")
assert 0<=SHARD_INDEX<64
import collections,json,hashlib,subprocess,sys,time,shutil
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

OUT=Path("/kaggle/working/vax_outputs")
OUT.mkdir(parents=True,exist_ok=True)
INDEX="vax-full-corpus-index-20261019"
def source(filename,slug):
    found=[p for p in Path("/kaggle/input").rglob(filename)
           if slug in str(p).lower()]
    if len(found)!=1:
        raise RuntimeError("Private dependency unavailable or ambiguous: "+
                           filename+" count="+str(len(found)))
    return found[0]
def run(*cmd):
    subprocess.run([sys.executable,*map(str,cmd)],check=True)
def readjsonl(p):
    out={}
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row=json.loads(line)
                k=row.get("record_key")
                if not k or k in out:raise RuntimeError("Duplicated/missing hash")
                out[k]=row
    return out

manifest=json.loads(source("full_corpus_manifest.json",INDEX).read_text(encoding="utf-8"))
if (manifest.get("shard_count")!=64 or
    manifest.get("input_dataset")!="keplaxo/twitter-vaccination-dataset" or
    manifest.get("sha256_exact_text_dedup") is not True or
    manifest.get("unique_texts",0)<2000):
    raise RuntimeError("Unverified full-corpus index manifest")
shard=source("shard_%02d.csv.gz"%SHARD_INDEX,INDEX)
if MODE=="stance":
    predecessor="vax-corpus-sentiment-%02d-20261019"%SHARD_INDEX
    screened=source("shard_screened.csv",predecessor)
    prior=json.loads(source("corpus_shard_report.json",predecessor).read_text())
    if prior.get("mode")!="sentiment" or prior.get("shard")!=SHARD_INDEX or not prior.get("coverage_complete"):
        raise RuntimeError("Sentiment prior model stage incomplete")
    # Exact key integrity checked against the index partition.
    original=pd.read_csv(shard,dtype=str,keep_default_na=False)
    frame=pd.read_csv(screened,dtype=str,keep_default_na=False)
    if set(original.record_key)!=set(frame.record_key):
        raise RuntimeError("Screened input hash set mismatch")
else:
    frame=pd.read_csv(shard,dtype=str,keep_default_na=False)
    if MODE=="pilot":
        frame=frame.sort_values("record_key").head(1000).copy()
    screened=OUT/"shard_screened.csv"
assert len(frame)>0 and frame.record_key.nunique()==len(frame)
assert all(hashlib.sha256(str(t).encode("utf-8")).hexdigest()==k
           for t,k in zip(frame.text,frame.record_key))
if MODE!="stance":
    input_csv=OUT/"shard_input.csv"
    frame.to_csv(input_csv,index=False)
repo=Path("/kaggle/working/vax-repo-corpus")
subprocess.run(["git","clone","--depth","1","--branch",
  "feature/kaggle-model-benchmark-v1",
  "https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination.git",
  str(repo)],check=True,capture_output=True,text=True)
run("-m","pip","install","-q","-e",str(repo))
run("-m","pip","install","-q","-r",str(repo/"requirements.txt"))
revision=subprocess.run(["git","rev-parse","HEAD"],cwd=repo,
                        capture_output=True,text=True,check=True).stdout.strip()
runtime={}
if MODE!="stance":
    t=time.perf_counter()
    run("-m","vaxbench.langscreen","--input",input_csv,
        "--output",screened)
    runtime["language_screen_seconds"]=round(time.perf_counter()-t,3)
eligible=pd.read_csv(screened,dtype=str,keep_default_na=False)
if len(eligible)!=len(frame):
    raise RuntimeError("Language screening coverage mismatch")
english=eligible["language_screen"].str.lower().isin(("en","eng","english"))
english_n=int(english.sum())
if MODE!="pilot" and SHARD_INDEX==0:
    # Recover capacity-probe predictions on a 1,000-record deterministic
    # subset. This avoids duplicating expensive predictions in shard zero.
    proof=source("corpus_capacity_probe.json","vax-corpus-capacity-20261019")
    pilot=json.loads(proof.read_text(encoding="utf-8"))
    if (pilot.get("mode")!="pilot" or pilot.get("input_unique_n")!=1000 or
        pilot.get("coverage_complete") is not True or
        pilot.get("repository_revision") is None):
        raise RuntimeError("Cannot reuse unverified GPU probe outputs")
    saved=("xlmt_sentiment.jsonl","roberta_english_sentiment.jsonl") if MODE=="sentiment" else ("deberta_stance.jsonl",)
    for file in saved:
        old=source(file,"vax-corpus-capacity-20261019")
        rows=readjsonl(old)
        if not set(rows).issubset(set(eligible.record_key)):
            raise RuntimeError("Capacity probe output keys are not in shard zero")
        shutil.copyfile(old,OUT/file)
    print("REUSED_CAPACITY_PROBE_PREDICTIONS",MODE,
          "shard=0",flush=True)

if MODE in ("pilot","sentiment"):
    t=time.perf_counter()
    run("-m","vaxbench.cli","sentiment","--model","multilingual",
        "--input",screened,"--output",OUT/"xlmt_sentiment.jsonl",
        "--batch-size","32")
    runtime["xlmt_seconds"]=round(time.perf_counter()-t,3)
    t=time.perf_counter()
    run("-m","vaxbench.cli","sentiment","--model","english",
        "--input",screened,"--output",OUT/"roberta_english_sentiment.jsonl",
        "--batch-size","32")
    runtime["roberta_seconds"]=round(time.perf_counter()-t,3)
if MODE in ("pilot","stance"):
    t=time.perf_counter()
    run("-m","vaxbench.cli","zeroshot","--input",screened,
        "--output",OUT/"deberta_stance.jsonl","--batch-size","4")
    runtime["deberta_seconds"]=round(time.perf_counter()-t,3)

models={}
for stage,filename,target in (
    ("multilingual_sentiment","xlmt_sentiment.jsonl",len(frame)),
    ("english_sentiment","roberta_english_sentiment.jsonl",english_n),
    ("deberta_stance","deberta_stance.jsonl",english_n)):
    if not (MODE=="pilot" or
           MODE=="sentiment" and stage!="deberta_stance" or
           MODE=="stance" and stage=="deberta_stance"):
        continue
    p=OUT/filename
    rows=readjsonl(p)
    expected=set(eligible.loc[english,"record_key"]) if stage!="multilingual_sentiment" else set(eligible.record_key)
    if len(rows)!=target or set(rows)!=expected:
        raise RuntimeError("Private "+stage+" completed with incomplete exact hash coverage")
    label_field="label" if stage=="deberta_stance" else "model_label"
    models[stage]={
        "predicted":len(rows),
        "expected":target,
        "labels":dict(sorted(collections.Counter(
            str(v.get(label_field,"missing")).lower() for v in rows.values()).items())),
        "output_hash_integrity_passed":True
    }
report={
    "generated_utc":datetime.now(timezone.utc).isoformat(),
    "mode":MODE,"shard":SHARD_INDEX,
    "repository_revision":revision,
    "input_unique_n":len(frame),
    "indexed_unique_n_for_shard":int(manifest["shard_unique_counts"]["%02d"%SHARD_INDEX]),
    "english_screened_n":english_n,
    "language_screen_method":"langid heuristic, not human-validated",
    "n_model_preds":models,
    "runtime_seconds":runtime,
    "coverage_complete":all(x["expected"]==x["predicted"] for x in models.values()),
    "accuracy_measured":False,
    "stance_validated":False,
    "full_corpus_population_prevalence_validated":False,
    "source_text_private":True,
    "individual_tweet_data_exported":False,
    "sample_n_for_capacity_only":len(frame) if MODE=="pilot" else None,
    "capacity_estimates_require_quota_review":True
}
if MODE!="pilot" and report["input_unique_n"]!=report["indexed_unique_n_for_shard"]:
    raise RuntimeError("Shard count differs from CPU index")
(OUT/("corpus_capacity_probe.json" if MODE=="pilot" else "corpus_shard_report.json")).write_text(
    json.dumps(report,indent=2),encoding="utf-8")
print("FULL_CORPUS_MODEL_STAGE_COMPLETE",json.dumps({
    "mode":MODE,"shard":SHARD_INDEX,"n":len(frame),"en":english_n,
    "models":{k:v["predicted"] for k,v in models.items()},
    "runtime_seconds":runtime,"no_accuracy_measured":True}),flush=True)
