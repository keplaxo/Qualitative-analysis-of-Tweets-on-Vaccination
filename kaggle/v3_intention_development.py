"""Private versioned Qwen v3 personal-intention development comparison.

16 balanced cases are used as a preflight (8 accept, 8 reject), then all
61 known VACMA references if operational quality passes. IMPORTANT: v3
prompt design was influenced by earlier results on this exact reference.
Metrics are exploratory DEVELOPMENT scores, NOT independent validation.
"""
MODE="MODE_PLACEHOLDER"
assert MODE in ("pilot","full")
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json,subprocess,sys,shutil
import pandas as pd
from hashlib import sha256

OUT=Path("/kaggle/working/vax_outputs");OUT.mkdir(parents=True,exist_ok=True)
SEED="v3-dual-axis-20261018"
PREV_SLUG="vax-v2-intent-proxy-20261018"
PILOT_SLUG="vax-intention-v3-pilot-20261018"
SIZE=16 if MODE=="pilot" else 61
REF_SHA="9f2991c7eba6e5dc2cd6028cce367139116d8f335c495821d8ef88075be73da7"
def one(filename,slug):
    matches=[p for p in Path("/kaggle/input").rglob(filename)
             if slug in str(p).lower()]
    if len(matches)!=1:raise RuntimeError("Required prior private file inaccessible: "+filename)
    return matches[0]
def run(*cmd):
    subprocess.run([sys.executable,*map(str,cmd)],check=True)
prev=one("vacma_intent_evaluation.json",PREV_SLUG)
old=json.loads(prev.read_text(encoding="utf-8"))
if old.get("eligible_unique_tweets")!=61 or old.get("file_sha256")!=REF_SHA:
    raise RuntimeError("Frozen intention proxy provenance differs")
df=pd.read_csv(prev.parent/"vacma_reference_61.csv",dtype=str,keep_default_na=False)
assert len(df)==61 and df.record_key.nunique()==61
assert df.reference.value_counts().to_dict()=={"support":43,"oppose":18}
if not all(sha256(str(t).encode("utf-8")).hexdigest()==k
           for t,k in zip(df.text,df.record_key)):
    raise RuntimeError("Prior published reference SHA-256 text identity mismatch")
# Predeclare sample using only published class labels and deterministic seed.
# No text-level cherry-picking, LLM score selection, or v2 output selection.
def rank(key):
    return sha256((SEED+"\0"+str(key)).encode("utf-8")).hexdigest()
positive=df[df.reference=="support"].copy()
negative=df[df.reference=="oppose"].copy()
positive["priority"]=positive.record_key.map(rank)
negative["priority"]=negative.record_key.map(rank)
positive=positive.sort_values("priority").drop(columns="priority")
negative=negative.sort_values("priority").drop(columns="priority")
pilot=pd.concat([positive.head(8),negative.head(8)],ignore_index=True)
if MODE=="pilot":selected=pilot
else:
    seen=set(pilot.record_key)
    remaining=df[~df.record_key.isin(seen)].copy()
    remaining["priority"]=remaining.record_key.map(rank)
    selected=pd.concat([pilot,
       remaining.sort_values("priority").drop(columns="priority")],
       ignore_index=True)
if len(selected)!=SIZE or selected.record_key.nunique()!=SIZE:
    raise RuntimeError("Invalid frozen v3 stratified pilot/full subset")
repo=Path("/kaggle/working/vax-intent-v3-repo")
subprocess.run(["git","clone","--depth","1","--branch",
 "feature/kaggle-model-benchmark-v1",
 "https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination.git",
 str(repo)],check=True,capture_output=True,text=True)
run("-m","pip","install","-q","-e",str(repo))
run("-m","pip","install","-q","-r",str(repo/"requirements.txt"))
source=repo/"src"
if not (source/"vaxbench"/"ontology_v3.py").is_file():
    raise RuntimeError("Versioned v3 classifier missing")
# Unlike v2's first run, invoke the CLI in a NEW process after pip install;
# no in-process imports of the editable package are needed.
source_csv=OUT/"v3_reference_sample.csv"
selected.to_csv(source_csv,index=False)
checkpoint=OUT/"v3_intention_predictions.jsonl"
if MODE=="full":
    pilot_report=one("v3_development_report.json",PILOT_SLUG)
    p=json.loads(pilot_report.read_text(encoding="utf-8"))
    if(p.get("mode")!="pilot" or p.get("total_n")!=16 or
       p.get("valid_schema_n",0)<12 or p.get("seed")!=SEED):
        raise RuntimeError("Pilot did not meet prespecified 12/16 validity gate")
    previous=pilot_report.parent/"v3_intention_predictions.jsonl"
    rows=[json.loads(l) for l in previous.read_text(encoding="utf-8").splitlines() if l.strip()]
    if (len(rows)!=16 or len(set(x["record_key"] for x in rows))!=16 or
        set(x["record_key"] for x in rows)!=set(selected.head(16).record_key) or
        any(x.get("label_ontology_version")!="v3" for x in rows)):
        raise RuntimeError("Pilot checkpoint hash/version mismatch")
    shutil.copyfile(previous,checkpoint)
    print("V3_PILOT_REUSED_16 no duplicate inference",flush=True)
print("V3_STAGE_STARTED",MODE,SIZE,flush=True)
run("-m","vaxbench.cli","qwen","--input",source_csv,"--output",checkpoint,
    "--max-records",str(SIZE),"--max-new-tokens","570","--four-bit",
    "--ontology-version","v3")
pred={}
for line in checkpoint.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    p=json.loads(line); k=p.get("record_key")
    if not k or k in pred:raise RuntimeError("Duplicate v3 prediction")
    pred[k]=p
if set(pred)!=set(selected.record_key):
    raise RuntimeError("Incomplete v3 checkpoint")
labels=("accept","reject","undecided","not_expressed","insufficient_context",
        "invalid_schema")
cross={"support":Counter(),"oppose":Counter()}
stance={"support":Counter(),"oppose":Counter()}
basis=Counter()
for item in selected.to_dict("records"):
    p=pred[item["record_key"]]
    if p.get("text_hash")!=item["record_key"] or p.get("label_ontology_version")!="v3":
        raise RuntimeError("v3 output key or ontology version mismatch")
    ok=p.get("status")=="validated_schema"
    val=p["annotation"].get("personal_intention") if ok else "invalid_schema"
    if val not in labels:raise RuntimeError("Unexpected v3 intention label")
    cross[item["reference"]][val]+=1
    if ok:
        ann=p["annotation"]
        stance[item["reference"]][ann["stance"]]+=1
        basis[ann["intention_basis"]]+=1
correct=cross["support"]["accept"]+cross["oppose"]["reject"]
valid=SIZE-sum(c["invalid_schema"] for c in cross.values())
covered=sum(c["accept"]+c["reject"] for c in cross.values())
report={
 "generated_utc":datetime.now(timezone.utc).isoformat(),
 "mode":MODE,"seed":SEED,
 "total_n":SIZE,
 "reference_support_n":int(sum(cross["support"].values())),
 "reference_oppose_n":int(sum(cross["oppose"].values())),
 "valid_schema_n":valid,"invalid_schema_n":SIZE-valid,
 "valid_schema_fraction":round(valid/SIZE,6),
 "intent_reference_vs_v3":{k:dict(v) for k,v in cross.items()},
 "stance_reference_vs_v3":{k:dict(v) for k,v in stance.items()},
 "intention_basis":dict(basis),
 "personal_intent_binary_coverage_n":covered,
 "personal_intent_correct_n":correct,
 "strict_intent_proxy_score_including_abstentions":round(correct/SIZE,6),
 "same_reference_previously_used_to_design_v3":True,
 "development_not_independent":True,
 "not_independent_accuracy":True,
 "six_way_historical_stance_accuracy_measured":False,
 "original_v2_outputs_unmodified":True,
 "raw_text_or_individual_predictions_in_public_repo":False,
}
(OUT/"v3_development_report.json").write_text(json.dumps(report,indent=2))
print("V3_DEVELOPMENT_AGGREGATE",json.dumps({
 "n":SIZE,"valid":valid,"binary_covered":covered,
 "correct_development":correct,"confirmatory_validation":False}),flush=True)
gate=12 if MODE=="pilot" else 46
if valid<gate:
    raise RuntimeError("V3_SCHEMA_GATE_FAILED "+str(valid)+"/"+str(SIZE))
print("V3_SCHEMA_GATE_PASSED",MODE,flush=True)
