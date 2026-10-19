"""Out-of-domain proxy evaluation on published human-coded vaccination intentions.

VACMA-PUBLIC contains 200 English text-bearing, independently expert-coded
tweets. Only the 61 mutually-exclusive positive-intent or negative-intent
reference examples are evaluated, not all 200 as vaccine-stance gold.

No reference labels enter the Qwen prompt. No calibration, prompt selection,
threshold tuning, or training is performed on evaluation examples.
The six-way vaccination stance task is NOT identical to intention labels.
"""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen
from io import StringIO
import csv,hashlib,json,subprocess,sys

SOURCE="https://raw.githubusercontent.com/peregilk/VACMA-PUBLIC/2e30fac04ad3caece6ca361fde53ca828ef68543/tweets_200_labeled.csv"
OUT=Path("/kaggle/working/vax_outputs")
OUT.mkdir(parents=True,exist_ok=True)

def run(*args):
    subprocess.run([sys.executable,*map(str,args)],check=True)

r=Request(SOURCE,headers={"User-Agent":"vaccine-stance-research-benchmark"})
with urlopen(r,timeout=80) as response:
    data=response.read(200_001)
if len(data)>200_000:raise RuntimeError("Unexpectedly large reference CSV")
source_checksum=hashlib.sha256(data).hexdigest()
csvtext=data.decode("utf-8-sig")
rows=list(csv.DictReader(StringIO(csvtext)))
accept="Intent to accept Covid-19 vaccination"
reject="Intent to reject Covid-19 vaccination"
if len(rows)!=200 or any(accept not in x or reject not in x for x in rows):
    raise RuntimeError("Public reference version/schema does not match plan")
norm=[]
for row in rows:
    a=str(row[accept]).strip()=="1"
    b=str(row[reject]).strip()=="1"
    if a and b:raise RuntimeError("Conflicting public intention labels")
    if not (a or b):continue
    t=str(row["text"])
    if not t.strip(): raise RuntimeError("Missing text in held-out intention reference")
    norm.append({"record_key":hashlib.sha256(t.encode("utf-8")).hexdigest(),
                 "text_hash":hashlib.sha256(t.encode("utf-8")).hexdigest(),
                 "text":t,"reference":"support" if a else "oppose"})
counts=Counter(x["reference"] for x in norm)
if len(norm)!=61 or counts!={"support":43,"oppose":18}:
    raise RuntimeError("Expected exactly 43 accepting and 18 rejecting reference tweets")
if len(set(x["record_key"] for x in norm))!=61:
    raise RuntimeError("Duplicate text in 61-case public benchmark")
import pandas as pd
dataset=OUT/"vacma_reference_61.csv"
pd.DataFrame(norm).to_csv(dataset,index=False)
repo=Path("/kaggle/working/vax-eval-repo")
subprocess.run(["git","clone","--depth","1","--branch",
    "feature/kaggle-model-benchmark-v1",
    "https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination.git",
    str(repo)],capture_output=True,text=True,check=True)
run("-m","pip","install","-q","-e",str(repo))
run("-m","pip","install","-q","-r",str(repo/"requirements.txt"))
checkpoint=OUT/"vacma_qwen_v2_predictions.jsonl"
run("-m","vaxbench.cli","qwen","--input",dataset,
    "--output",checkpoint,"--max-records","61","--four-bit",
    "--max-new-tokens","420","--ontology-version","v2")
pred={}
with checkpoint.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():continue
        ob=json.loads(line)
        key=ob["record_key"]
        if key in pred:raise RuntimeError("Repeated output record key")
        pred[key]=ob
if len(pred)!=61 or set(pred)!=set(x["record_key"] for x in norm):
    raise RuntimeError("External v2 benchmark incomplete")

tab={reference:{target:0 for target in ("support","oppose","abstain")}
     for reference in ("support","oppose")}
subcats=Counter()
valid=0
for item in norm:
    x=pred[item["record_key"]]
    if x.get("status")=="validated_schema":
        valid+=1
        label=x["annotation"]["stance"]
        target=label if label in ("support","oppose") else "abstain"
        if target=="abstain":subcats[label]+=1
    else:
        target="abstain"
        subcats["invalid_schema"]+=1
    tab[item["reference"]][target]+=1

total=61
correct=tab["support"]["support"]+tab["oppose"]["oppose"]
covered=sum(tab[a][b] for a in ("support","oppose") for b in ("support","oppose"))
class_scores={}
for label in ("support","oppose"):
    tp=tab[label][label]
    actual=sum(tab[label].values())
    predicted=tab["support"][label]+tab["oppose"][label]
    precision=tp/predicted if predicted else 0.0
    recall=tp/actual if actual else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    class_scores[label]={"reference_n":actual,"predicted_n":predicted,
                         "precision":round(precision,6),
                         "recall_including_abstentions":round(recall,6),
                         "f1_abstentions_as_missed_reference":round(f1,6)}
report={
    "created_utc":datetime.now(timezone.utc).isoformat(),
    "reference_name":"VACMA-PUBLIC maternal and COVID-19 vaccination confidence, two independently labeled intention fields",
    "reference_url":"https://github.com/peregilk/VACMA-PUBLIC",
    "reference_commit":"2e30fac04ad3caece6ca361fde53ca828ef68543",
    "file_sha256":source_checksum,
    "input_labeled_tweets":200,
    "task":"out_of_domain_proxy_for_author_vaccination_intention_support_vs_oppose",
    "reference_positive_intent":43,
    "reference_negative_intent":18,
    "eligible_unique_tweets":61,
    "reference_rows_used_for_training":0,
    "reference_rows_used_for_prompt_tuning":0,
    "inference_model":"Qwen/Qwen3-4B-Instruct-2507",
    "ontology_version":"v2",
    "schema_valid":valid,
    "binary_model_coverage":covered,
    "abstention_count":total-covered,
    "abstention_breakdown":dict(subcats),
    "binary_confusion_reference_rows_prediction_cols":tab,
    "class_metrics":class_scores,
    "strict_reference_accuracy_including_abstentions":round(correct/total,6),
    "macro_f1_with_abstentions_as_errors":round(sum(v["f1_abstentions_as_missed_reference"]
                                                   for v in class_scores.values())/2,6),
    "majority_class_reference_baseline":round(43/61,6),
    "six_way_vaccine_stance_accuracy_measured":False,
    "out_of_domain":True,
    "generalizable_to_historical_corpus":False,
    "label_task_mismatch_note":"Acceptance/rejection intentions are not equivalent to all author-level six-way vaccination stance categories; negative references missing outside selected subset.",
    "no_tweet_text_or_row_data_published":True
}
(OUT/"vacma_intent_evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print("EXTERNAL_INTENT_PROXY_EVALUATED",json.dumps({
    "n":total,"v2_schema_valid":valid,"binary_covered":covered,
    "strict_accuracy":report["strict_reference_accuracy_including_abstentions"],
    "macro_f1":report["macro_f1_with_abstentions_as_errors"],
    "out_of_domain":True,"six_way_accuracy":False}),flush=True)
