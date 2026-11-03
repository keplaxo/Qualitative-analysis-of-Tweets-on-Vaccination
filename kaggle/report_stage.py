"""CPU-only final aggregate research report using private Kaggle notebook inputs.

Strictly no original tweet text, tweet IDs, personal identifiers, or individual
row predictions are printed or exported by this report stage.
"""
import collections,json
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd

OUT=Path("/kaggle/working/vax_outputs")
OUT.mkdir(parents=True,exist_ok=True)

def one(filename,slug):
    matching=[p for p in Path("/kaggle/input").rglob(filename)
              if slug in str(p).lower()]
    if len(matching)!=1:
        raise RuntimeError("Private notebook input missing/ambiguous: "+filename)
    return matching[0]

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(v) for v in f if v.strip()]

baseline=one("summary.json","vax-sentiment-benchmark-full")
qreport=one("qwen_validation.json","vax-qwen-300")
base=json.loads(baseline.read_text(encoding="utf-8"))
q=json.loads(qreport.read_text(encoding="utf-8"))
if (base.get("n_sampled")!=2000 or
    q.get("processed")!=300 or
    q.get("schema_valid_count",0)<210 or
    q.get("mode")!="full"):
    raise RuntimeError("Final report prerequisites not met")

sample=pd.read_csv(baseline.parent/"benchmark_sample_screened.csv",
                   dtype=str,keep_default_na=False)
queue=pd.read_csv(qreport.parent/"qwen_triage.csv",
                  dtype=str,keep_default_na=False)
deberta=read_jsonl(baseline.parent/"deberta_stance.jsonl")
qrows=read_jsonl(qreport.parent/"qwen_semantics.jsonl")
assert len(queue)==300 and len(qrows)==300
assert len(set(queue.record_key))==len(queue)
assert len(set(x["record_key"] for x in qrows))==len(qrows)

keyed={x["record_key"]:x for x in deberta}
qa={x["record_key"]:x for x in qrows}
qtable=queue[["record_key","year","selection_reason"]].copy()
qtable=qtable.merge(sample[["record_key","language_screen"]],
                    on="record_key",how="left",validate="one_to_one")
counts=collections.Counter()
strata=collections.defaultdict(lambda:collections.Counter())
year_counts=collections.defaultdict(lambda:collections.Counter())
flags=collections.Counter()
for row in qtable.to_dict("records"):
    key=row["record_key"]
    a=qa[key]
    nli=keyed.get(key,{})
    valid=a.get("status")=="validated_schema"
    stats=strata[row["selection_reason"]]
    years=year_counts[str(row["year"])]
    for s in (stats,years):
        s["processed"]+=1
        s["schema_valid"]+=int(valid)
    if valid:
        ann=a["annotation"]
        agreement=nli.get("label")==ann.get("stance")
        for s in (stats,years):
            s["paired_valid"]+=1
            s["stance_match"]+=int(agreement)
        counts["stance_match"]+=int(agreement)
        counts["stance_disagree"]+=int(not agreement)
        for key2 in ("safety_concern","adverse_event_report",
                     "procedure_fear","access_barrier"):
            flags[key2+"_yes"]+=int(ann.get(key2) is True)
        counts["human_vaccine_relevant"]+=int(ann.get("relevance")=="human")
    else:
        counts["schema_unresolved"]+=1

def safe_counts(src):
    return {str(k):dict(v) for k,v in sorted(src.items())}

report={
 "generated_utc":datetime.now(timezone.utc).isoformat(),
 "method":"descriptive comparison of independent model predictions on purposive triage sample",
 "historical_dataset_scope":"2006–2019 year-covered 2,000-case development subset",
 "base_n":base["n_sampled"],
 "qwen_selected_n":len(queue),
 "qwen_schema_valid_n":q["schema_valid_count"],
 "qwen_schema_invalid_n":q["schema_invalid_count"],
 "schema_validation_rate":q["schema_valid_fraction"],
 "base_sentiment_agreement":base.get("sentiment_agreement"),
 "qwen_deberta_stance_paired_valid_n":counts["stance_match"]+counts["stance_disagree"],
 "qwen_deberta_stance_matching_n":counts["stance_match"],
 "qwen_deberta_stance_agreement_rate":(
     counts["stance_match"]/(counts["stance_match"]+counts["stance_disagree"])
     if counts["stance_match"]+counts["stance_disagree"] else None),
 "selection_strata":safe_counts(strata),
 "year_strata":safe_counts(year_counts),
 "qwen_stance_valid_schema_counts":q["stance_on_schema_valid"],
 "semantic_flag_counts_on_valid":dict(flags),
 "accuracy_measured":False,
 "human_approval_required":False,
 "generalizable_to_corpus":False,
 "limitations":[
    "Sentiment and vaccine-specific author stance are not interchangeable",
    "Sampling deliberately enriches low-margin and abstained DeBERTa predictions",
    "Automated labels are not verified against independent human gold labels",
    "Language eligibility relies on an automatic classifier",
    "Clinical claims and adverse-event statements are classified as discourse only"
 ]}
(OUT/"research_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
text=[
 "# Exploratory vaccination sentiment and stance model comparison",
 "",
 "## Completed computational coverage",
 "2,000 development tweets processed in the initial benchmark; 300 purposively selected for Qwen3 semantic review.",
 "",
 "## Model comparison",
 f"Qwen schema-valid classifications: {report['qwen_schema_valid_n']}/300.",
 f"Paired DeBERTa–Qwen stance agreement among schema-valid outputs: {report['qwen_deberta_stance_matching_n']}/{report['qwen_deberta_stance_paired_valid_n']}.",
 f"Base sentiment paired label agreement: {base['sentiment_agreement']['matching']}/{base['sentiment_agreement']['paired']}.",
 "",
 "## Interpretation restrictions",
 "Model agreement is not accuracy. This purposive comparison does not estimate vaccination sentiment prevalence.",
 "No independent, task-matched validity reference is available, so full-corpus inference is gated pending further methodological assessment.",
]
(OUT/"research_report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
print("AGGREGATE_REPORT_COMPLETE",json.dumps({
 "base_n":report["base_n"],"qwen_n":report["qwen_selected_n"],
 "qwen_valid":report["qwen_schema_valid_n"],
 "paired":report["qwen_deberta_stance_paired_valid_n"],
 "agreement":report["qwen_deberta_stance_agreement_rate"]
}),flush=True)
