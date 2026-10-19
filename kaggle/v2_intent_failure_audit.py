"""CPU-only post-hoc failure analysis of frozen VACMA intention-proxy outputs.

This diagnoses construct mismatch and evidence-generation failure, but is NOT
a new independent test; the reference set has already been inspected.
Private rows/text/predictions remain only in Kaggle controlled storage.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json,math
import pandas as pd

OUT=Path("/kaggle/working/vax_outputs"); OUT.mkdir(parents=True,exist_ok=True)
def one(filename,slug):
    a=[p for p in Path("/kaggle/input").rglob(filename)
       if slug in str(p).lower()]
    if len(a)!=1:raise RuntimeError("Upstream private output missing/ambiguous "+filename)
    return a[0]
report_path=one("vacma_intent_evaluation.json","vax-v2-intent-proxy-20261018")
original=json.loads(report_path.read_text(encoding="utf-8"))
if(original.get("eligible_unique_tweets")!=61 or
   original.get("strict_reference_accuracy_including_abstentions")!=0.196721):
    raise RuntimeError("Prior frozen proxy evaluation changed")
frame=pd.read_csv(report_path.parent/"vacma_reference_61.csv",
                  dtype=str,keep_default_na=False)
assert len(frame)==61 and frame.record_key.nunique()==61
pred={}
with (report_path.parent/"vacma_qwen_v2_predictions.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():continue
        r=json.loads(line);key=r.get("record_key")
        if not key or key in pred:raise RuntimeError("Duplicate/missing v2 ID")
        pred[key]=r
if set(pred)!=set(frame.record_key):raise RuntimeError("Prior 61 output join mismatch")
metrics=defaultdict(Counter)
margins=defaultdict(list)
for row in frame.to_dict("records"):
    key=row["record_key"]
    p=pred[key]
    ref=row["reference"]
    valid=p.get("status")=="validated_schema"
    s="invalid_schema"
    ann=p.get("annotation",{}) if valid else {}
    if valid:s=ann.get("stance","unexpected")
    # Safe only: aggregate metadata and lexical cohort counts, no individual
    # example excerpts or hashes, nor any LLM evidence text in logs.
    group=metrics[ref]
    group["total"]+=1
    group["v2_"+s]+=1
    group["schema_valid"]+=int(valid)
    if valid:
        group["basis_"+str(ann.get("interpretation_basis","missing"))]+=1
        group["relevance_"+str(ann.get("relevance","missing"))]+=1
        group["evidence_nonempty"]+=int(bool(ann.get("evidence")))
        group["evidence_empty"]+=int(not ann.get("evidence"))
        group["stance_support_predicted"]+=int(s=="support")
        group["stance_oppose_predicted"]+=int(s=="oppose")
        group["stance_no_expr"]+=int(s=="no_expressed_stance")
        group["stance_uncertain"]+=int(s=="uncertain")
        group["with_author_evidence_but_no_stance"]+=int(
            s=="no_expressed_stance" and bool(ann.get("evidence")))
    txt=str(row["text"]).lower()
    for name,pattern in (
        ("future_intent_language",r"\b(?:will|get(?:ting)?|plan|intend|want|book(?:ed|ing)?|going)\b"),
        ("personal_marker",r"\b(?:i|my|me|we|our)\b"),
        ("vaccination_term",r"\b(?:vaccin\w*|jab|shot|dose|pfizer|moderna|booster)\b"),
        ("mandate_term",r"\b(?:mandat\w*|passport|government)\b")):
        import re
        group["lexical_"+name]+=int(bool(re.search(pattern,txt)))
    margins[ref].append(len(txt))
overview={
 "created_utc":datetime.now(timezone.utc).isoformat(),
 "prior_reference":"VACMA-PUBLIC 61-case author vaccination intention proxy",
 "prior_score_12_of_61":True,
 "intention_reference_known_to_model_designer":True,
 "this_is_post_hoc_development_audit_not_new_test":True,
 "model_accuracy_newly_estimated":False,
 "n":61,
 "by_intention_reference":{k:dict(v) for k,v in sorted(metrics.items())},
 "mean_tweet_length_by_reference":{k:round(sum(v)/len(v),1)
                                   for k,v in sorted(margins.items())},
 "interpretation":"Compare how a generic author vaccine stance head is used on an author intention reference. Lexical flags are coarse heuristics, not annotations or causal findings.",
 "individual_text_or_predictions_exported":False,
}
(OUT/"v2_intent_failure_audit.json").write_text(json.dumps(overview,indent=2))
print("V2_INTENTION_DIAGNOSTIC_COMPLETE",json.dumps({
  "n":61,"by_reference":overview["by_intention_reference"],
  "new_accuracy":False,"private_aggregate_only":True,
}),flush=True)
