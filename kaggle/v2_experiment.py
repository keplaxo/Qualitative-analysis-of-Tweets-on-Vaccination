"""Private, versioned Qwen v2 experiment on 12/60 prespecified past records.

The mode placeholder is replaced by a GitHub Actions orchestrator. This stage
cannot overwrite historical benchmark or Qwen v1 outputs. It never exports
tweet text or row-level predictions from private Kaggle to public GitHub.
"""
MODE="MODE_PLACEHOLDER"
assert MODE in ("pilot","full")
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json,shutil,subprocess,sys
import pandas as pd

OUT=Path("/kaggle/working/vax_outputs")
OUT.mkdir(parents=True,exist_ok=True)
SEED="vax-ontology-v2-20261018"
BASE="vax-sentiment-benchmark-full"
ORIGINAL="vax-qwen-300"
PILOT="vax-stance-v2-pilot-20261018"
EXPECTED=12 if MODE=="pilot" else 60

def one(name,slug):
    matches=[p for p in Path("/kaggle/input").rglob(name)
             if slug in str(p).lower()]
    if len(matches)!=1:
        raise RuntimeError("Missing/ambiguous prior PRIVATE Kaggle output: "+
                           slug+"/"+name+" count="+str(len(matches)))
    return matches[0]

def jlines(path):
    results={}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj=json.loads(line)
                k=obj.get("record_key")
                if not k or k in results:raise RuntimeError("Bad checkpoint identity")
                results[k]=obj
    return results

def pip(*cmd):
    subprocess.run([sys.executable,*map(str,cmd)],check=True)

b_summary=one("summary.json",BASE)
q_summary=one("qwen_validation.json",ORIGINAL)
b=json.loads(b_summary.read_text(encoding="utf-8"))
q=json.loads(q_summary.read_text(encoding="utf-8"))
if (b.get("n_sampled")!=2000 or
    b.get("stage_coverage",{}).get("zero_shot_stance")!=1666 or
    q.get("processed")!=300 or q.get("schema_valid_count")!=290):
    raise RuntimeError("Unexpected input model benchmark revisions/coverage")
pred_nli=jlines(b_summary.parent/"deberta_stance.jsonl")
pred_v1=jlines(q_summary.parent/"qwen_semantics.jsonl")
queue=pd.read_csv(q_summary.parent/"qwen_triage.csv",
                  dtype=str,keep_default_na=False)
if len(queue)!=300 or queue.record_key.nunique()!=300:
    raise RuntimeError("Triage input not unique 300")
repo=Path("/kaggle/working/vax-research-v2")
subprocess.run(["git","clone","--depth","1","--branch",
 "feature/kaggle-model-benchmark-v1",
 "https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination.git",
 str(repo)],check=True,capture_output=True,text=True)
pip("-m","pip","install","-q","-e",str(repo))
pip("-m","pip","install","-q","-r",str(repo/"requirements.txt"))
# The Kaggle script interpreter has already started before editable pip
# installation. Newly written .pth metadata is not processed until a new
# interpreter starts. Put the checked-out src tree on THIS process's path.
src_dir=repo/"src"
if not (src_dir/"vaxbench"/"v2_selection.py").is_file():
    raise RuntimeError("Checked-out v2 package source is missing")
sys.path.insert(0,str(src_dir))
from vaxbench.v2_selection import select_v2_cases
from vaxbench.core import text_digest
selected=select_v2_cases(pred_nli,pred_v1)
s=pd.DataFrame(selected)
data=s.merge(queue,on="record_key",how="left",validate="one_to_one")
if (len(data)!=60 or data["text"].isna().any() or
    data["record_key"].nunique()!=60 or
    not all(text_digest(str(t))==k for t,k in
            zip(data.text,data.record_key))):
    raise RuntimeError("v2 join and exact-text SHA-256 checks failed")
data=data.head(EXPECTED).copy()
dataset=OUT/"v2_sample.csv"
data.to_csv(dataset,index=False)
checkpoint=OUT/"v2_predictions.jsonl"
if MODE=="full":
    proof=one("v2_aggregate.json",PILOT)
    old=json.loads(proof.read_text(encoding="utf-8"))
    if (old.get("mode")!="pilot" or old.get("evaluated")!=12 or
        old.get("schema_valid_count",0)<9 or
        old.get("design_seed")!=SEED):
        raise RuntimeError("v2 pilot failed predeclared schema gate")
    previous=proof.parent/"v2_predictions.jsonl"
    rows=jlines(previous)
    if (len(rows)!=12 or
        set(rows)!=set(data.head(12).record_key) or
        any(v.get("label_ontology_version")!="v2" for v in rows.values())):
        raise RuntimeError("Pilot outputs not consistent with v2 selection")
    shutil.copyfile(previous,checkpoint)
    print("Reusing 12 frozen pilot inferences; only 48 new Qwen calls.",flush=True)
print("V2_STAGE",MODE,"sample",EXPECTED,flush=True)
pip("-m","vaxbench.cli","qwen",
    "--input",dataset,"--output",checkpoint,
    "--max-records",str(EXPECTED),
    "--max-new-tokens","420","--four-bit","--ontology-version","v2")

results=jlines(checkpoint)
if set(results)!=set(data.record_key) or len(results)!=EXPECTED:
    raise RuntimeError("Incomplete versioned Qwen v2 checkpoint")
for k,v in results.items():
    if v.get("label_ontology_version")!="v2" or v.get("text_hash")!=k:
        raise RuntimeError("Qwen v2 version/key mismatch")
old_stance=Counter()
new_stance=Counter()
transitions=Counter()
strata=defaultdict(lambda:Counter())
schema_valid=0
matching_new_old=0
matching_new_nli=0
matching_old_nli=0
valid_new_old=0
abstentions=Counter()
for row in data.to_dict("records"):
    key=row["record_key"]
    y=results[key]
    pred=pred_v1[key]
    nli=pred_nli[key]
    group=str(row["selection_reason"])
    strata[group]["evaluated"]+=1
    p=pred.get("annotation",{}).get("stance","invalid")
    dl=nli.get("label","missing")
    old_stance[p]+=1
    matching_old_nli+=int(p==dl)
    if y.get("status")=="validated_schema":
        label=y["annotation"]["stance"]
        schema_valid+=1
        valid_new_old+=1
        matching_new_old+=int(label==p)
        matching_new_nli+=int(label==dl)
        new_stance[label]+=1
        transitions[p+"__TO__"+label]+=1
        strata[group]["v2_schema_valid"]+=1
        strata[group]["v2_matches_original_qwen"]+=int(label==p)
        strata[group]["v2_matches_deberta"]+=int(label==dl)
        abstentions["insufficient_context"]+=int(label=="insufficient_context")
        abstentions["uncertain"]+=int(label=="uncertain")
    else:
        strata[group]["v2_invalid"]+=1
        abstentions["invalid_reason_"+str(y.get("error","unknown")).split(":")[0]]+=1

report={
 "generated_utc":datetime.now(timezone.utc).isoformat(),
 "mode":MODE,
 "design_seed":SEED,
 "design":"60 fixed independent of v2 predictions: 20 NLI abstention/Qwen uncertain, 15 NLI support/Qwen uncertain, 10 opposition conflicts, 15 exact-agreement controls; pilot 12 interleaved",
 "baseline_2000_confirmed":True,
 "original_qwen_300_confirmed":True,
 "evaluated":EXPECTED,
 "schema_valid_count":schema_valid,
 "schema_invalid_count":EXPECTED-schema_valid,
 "schema_valid_rate":round(schema_valid/EXPECTED,6),
 "v1_automatic_labels":dict(old_stance),
 "v2_automatic_labels":dict(new_stance),
 "v1_to_v2_transitions":dict(transitions),
 "v2_vs_v1_matches_among_valid":matching_new_old,
 "v2_vs_v1_paired_valid":valid_new_old,
 "v1_vs_nli_matches_all_selected":matching_old_nli,
 "v2_vs_nli_matches_among_valid":matching_new_nli,
 "by_stratum":{k:dict(v) for k,v in sorted(strata.items())},
 "key_integrity":True,
 "v2_is_better_not_established":True,
 "reference_accuracy_measured":False,
 "representative_prevalence_sample":False,
 "human_labels_newly_generated":False,
 "qwen_v2_model":"Qwen/Qwen3-4B-Instruct-2507",
 "not_full_corpus_inference":True,
}
(OUT/"v2_aggregate.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print("V2_AGGREGATE",json.dumps({"mode":MODE,"evaluated":EXPECTED,
  "valid":schema_valid,"invalid":EXPECTED-schema_valid,
  "v2_vs_v1_matching":matching_new_old,"v2_vs_nli_matching":matching_new_nli}),flush=True)
threshold=9 if MODE=="pilot" else 45
if schema_valid<threshold:
    raise RuntimeError("V2_SCHEMA_GATE_BLOCKED: below preregistered "+
                       str(threshold)+"/"+str(EXPECTED)+" valid outcomes")
print("V2_SCHEMA_GATE_PASSED",MODE,flush=True)
