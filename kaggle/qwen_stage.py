"""Private Kaggle GPU Qwen3 stage: 20-case preflight then 300-case semantic review.

MODE is substituted by the GitHub submission script; this source does not
contain tweet data, API credentials, or any personal identifiers.
"""
MODE = "MODE_PLACEHOLDER"
import collections
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

OUT=Path("/kaggle/working/vax_outputs")
OUT.mkdir(parents=True,exist_ok=True)
FULL_REF="keplaxo/vax-sentiment-benchmark-full"
PILOT_REF="keplaxo/vax-qwen-pilot"
SEED="20261017-qwen"
QWEN_EXPECTED=20 if MODE=="pilot" else 300
assert MODE in ("pilot","full"), "Invalid Qwen execution mode"

def safe_python(*parts):
    subprocess.run([sys.executable,*map(str,parts)],check=True)

def locate_source(filename, expected_ref):
    # Attached notebook inputs can use old or new Kaggle input mount layouts.
    matches=[p for p in Path("/kaggle/input").rglob(filename)
             if expected_ref.split("/")[-1] in str(p).lower()]
    if len(matches)!=1:
        raise RuntimeError("Expected exactly one private upstream notebook file "+
                           filename+"; found "+str(len(matches)))
    return matches[0]

def read_jsonl(path):
    records=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

summary_file=locate_source("summary.json",FULL_REF)
upstream=summary_file.parent
summary=json.loads(summary_file.read_text(encoding="utf-8"))
coverage=summary.get("stage_coverage",{})
if (summary.get("n_sampled")!=2000 or
    coverage.get("multilingual_sentiment")!=2000 or
    coverage.get("english_sentiment")!=1666 or
    coverage.get("zero_shot_stance")!=1666):
    raise RuntimeError("Private 2,000-tweet upstream benchmark coverage changed; stop")

sample=upstream/"benchmark_sample_screened.csv"
nli=upstream/"deberta_stance.jsonl"
xlmt=upstream/"xlmt_sentiment.jsonl"
roberta=upstream/"roberta_english_sentiment.jsonl"
for p in (sample,nli,xlmt,roberta):
    if not p.is_file(): raise RuntimeError("Missing private upstream artifact "+p.name)

# Never redo the 2,000-case sentiment and NLI inference.
repo=Path("/kaggle/working/vaccination-sentiment-research")
subprocess.run(["git","clone","--depth","1","--branch",
  "feature/kaggle-model-benchmark-v1",
  "https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination.git",
  str(repo)],check=True,capture_output=True,text=True)
safe_python("-m","pip","install","-q","-e",str(repo))
safe_python("-m","pip","install","-q","-r",str(repo/"requirements.txt"))
revision=subprocess.run(["git","rev-parse","HEAD"],cwd=repo,
  capture_output=True,text=True,check=True).stdout.strip()
print("Stage mode:",MODE,"Repository commit:",revision,flush=True)

triage=OUT/"qwen_triage.csv"
safe_python("-m","vaxbench.triage",
    "--input",sample,"--zeroshot",nli,"--output",triage,
    "--max-records","300","--hard-fraction","0.6","--seed",SEED)
import pandas as pd
queue=pd.read_csv(triage,dtype=str,keep_default_na=False)
if len(queue)!=300 or queue["record_key"].nunique()!=300:
    raise RuntimeError("Triage did not select 300 distinct records")

checkpoint=OUT/"qwen_semantics.jsonl"
if MODE=="full":
    pilot_report=locate_source("qwen_validation.json",PILOT_REF)
    pilot_state=json.loads(pilot_report.read_text(encoding="utf-8"))
    if (pilot_state.get("mode")!="pilot" or
        pilot_state.get("processed")!=20 or
        pilot_state.get("schema_valid_count",0)<14 or
        pilot_state.get("triage_seed")!=SEED):
        raise RuntimeError("Prior 20-case Qwen pilot did not meet operational QC gate")
    pilot_checkpoint=pilot_report.parent/"qwen_semantics.jsonl"
    if not pilot_checkpoint.is_file(): raise RuntimeError("Pilot Qwen checkpoint absent")
    pilot_rows=read_jsonl(pilot_checkpoint)
    first20=set(queue.head(20)["record_key"])
    if len(pilot_rows)!=20 or set(str(x["record_key"]) for x in pilot_rows)!=first20:
        raise RuntimeError("Qwen pilot keys do not match current deterministic triage")
    shutil.copyfile(pilot_checkpoint,checkpoint)
    print("Reusing 20 saved Qwen pilot outputs; no repeat model calls for those keys.",flush=True)

# A single stage can be checkpointed inside the Kaggle run. Separate private
# notebook outputs preserve the 20-case pilot for the next stage.
safe_python("-m","vaxbench.cli","qwen",
    "--input",triage,"--output",checkpoint,
    "--max-records",str(QWEN_EXPECTED),"--max-new-tokens","420","--four-bit")

rows=read_jsonl(checkpoint)
used_keys=[str(x["record_key"]) for x in rows]
want=set(queue.head(QWEN_EXPECTED)["record_key"])
if len(rows)!=QWEN_EXPECTED or len(set(used_keys))!=len(rows) or set(used_keys)!=want:
    raise RuntimeError("Qwen checkpoint incomplete or keys do not match the triage set")

valid=[x for x in rows if x.get("status")=="validated_schema"]
invalid=[x for x in rows if x.get("status")!="validated_schema"]
counts=dict(collections.Counter(x.get("annotation",{}).get("stance","unknown") for x in valid))
reason=dict(collections.Counter(x.get("status","missing") for x in rows))
out={
    "mode":MODE,
    "timestamp_utc":datetime.now(timezone.utc).isoformat(),
    "model":"Qwen/Qwen3-4B-Instruct-2507",
    "repository_revision":revision,
    "triage_seed":SEED,
    "sample_design":"purposive 60% uncertain/abstained NLI, 40% hash-ranked comparison; no prevalence",
    "triage_pool":int(len(queue)),
    "processed":int(len(rows)),
    "schema_valid_count":int(len(valid)),
    "schema_invalid_count":int(len(invalid)),
    "schema_valid_fraction":round(len(valid)/len(rows),5),
    "stance_on_schema_valid":counts,
    "status_counts":reason,
    "accuracy_measured":False,
    "no_human_ground_truth":True,
    "upstream_benchmark_n":int(summary["n_sampled"]),
    "sensitivity_warning":"Schema-valid does not establish correct vaccine stance."
}
(OUT/"qwen_validation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print("QWEN_VALIDATION_COUNTS",json.dumps({
    "mode":MODE,"processed":out["processed"],"valid":out["schema_valid_count"],
    "invalid":out["schema_invalid_count"]}),flush=True)

# Additional descriptive semantic comparison is part of the 300-case stage.
if MODE=="full":
    safe_python("-m","vaxbench.summarize",
        "--sample",sample,"--multilingual",xlmt,
        "--english",roberta,"--zeroshot",nli,
        "--qwen",checkpoint,"--output",OUT/"qwen_comparison.json")

# This threshold is a cost/parseability guard, NOT an accuracy threshold.
if out["schema_valid_fraction"]<0.70:
    raise RuntimeError("Qwen schema validation below 70% operational continuation threshold")
print("Qwen stage completed and passed the structural output gate.",flush=True)
