"""Private final CPU synthesis of sixteen previously verified 4-shard waves.

Descriptive label frequencies only, with denominators and model coverage.
No raw tweets, IDs, individual records or private per-tweet predictions printed
or exported into GitHub.
"""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json

OUT=Path("/kaggle/working/vax_outputs");OUT.mkdir(parents=True,exist_ok=True)
def one(filename,slug):
    found=[p for p in Path("/kaggle/input").rglob(filename)
           if slug in str(p).lower()]
    if len(found)!=1:raise RuntimeError("Missing/ambiguous reporting dependency "+slug)
    return json.loads(found[0].read_text(encoding="utf-8"))
index=one("full_corpus_manifest.json","vax-full-corpus-index-20261019")
events=one("full_corpus_occurrence_manifest.json","vax-corpus-event-join-20261019-v2")
if (index.get("shard_count")!=64 or
    index.get("unique_texts")!=events.get("unique_texts") or
    index.get("raw_rows_total")-index.get("blank_rows")!=
    events.get("total_nonempty_events")):
    raise RuntimeError("Original source index/event totals changed")

def merge_group(dest,source):
    for k,v in source.items():
        if isinstance(v,dict):
            target=dest.setdefault(k,{})
            for label,n in v.items():
                target[label]=target.get(label,0)+int(n)
        else:
            dest[k]=dest.get(k,0)+int(v)
def merge_dimension(dest,data):
    for key,row in data.items():
        merge_group(dest.setdefault(key,{}),row)
def integrity(dataset,scope):
    total=sum(x.get("rows",0) for x in dataset.values())
    en=sum(x.get("english_eligible",0) for x in dataset.values())
    for label,g in dataset.items():
        if (sum(g.get("multilingual_sentiment",{}).values())!=g["rows"] or
            sum(g.get("english_sentiment",{}).values())!=g["english_eligible"] or
            sum(g.get("deberta_stance",{}).values())!=g["english_eligible"] or
            g.get("deberta_abstained",0)>g["english_eligible"]):
            raise RuntimeError("Model label coverage violated in "+scope+"/"+label)
    return total,en

unique=event_year=event_source=None
unique={};event_year={};event_source={}
revisions=defaultdict(set)
n_unique=n_raw=0
for idx in range(16):
    slug="vax-corpus-report-wave-%02d-20261019"%idx
    report=one("corpus_report_wave.json",slug)
    if (report.get("wave")!=idx or
        report.get("shards")!=list(range(idx*4,(idx+1)*4)) or
        report.get("entire_wave_hash_coverage_verified") is not True or
        report.get("tweet_content_or_hashes_exported") is not False or
        report.get("stance_unvalidated") is not True):
        raise RuntimeError("Missing full coverage/warning in reporting wave")
    n_unique+=report["unique_texts"];n_raw+=report["raw_events_nonempty"]
    merge_dimension(unique,report["by_year_unique"])
    merge_dimension(event_year,report["by_year_events"])
    merge_dimension(event_source,report["by_source_events"])
    for model,values in report.get("model_revisions",{}).items():
        revisions[model].update(values)
if n_unique!=index["unique_texts"] or n_raw!=events["total_nonempty_events"]:
    raise RuntimeError("Aggregated reports do not cover full source corpus")
count_unique,english_unique=integrity(unique,"unique_year")
count_raw,english_raw=integrity(event_year,"raw_year")
source_raw,source_english=integrity(event_source,"raw_source")
if (count_unique!=n_unique or count_raw!=n_raw or source_raw!=n_raw
    or source_english!=english_raw or
    {str(k):v["rows"] for k,v in event_year.items()}!=
        {str(k):int(n) for k,n in events["event_year_counts"].items()}):
    raise RuntimeError("Event/source/year coverage validation failed")
if any(len(rev)>1 for rev in revisions.values()):
    raise RuntimeError("Multiple model revisions across corpus partitions require assessment")

def labels(row,name):
    counts={}
    for value in row.values():
        for label,n in value.get(name,{}).items():
            counts[label]=counts.get(label,0)+n
    return dict(sorted(counts.items()))

result={
 "created_utc":datetime.now(timezone.utc).isoformat(),
 "title":"Model-predicted vaccination discourse across the indexed full historical Twitter corpus",
 "study_scope":"Entire supplied private corpus, both source CSVs, exact-text dedup with original occurrence weights",
 "n_original_source_rows":index["raw_rows_total"],
 "n_blank_text_rows":index["blank_rows"],
 "n_nonblank_tweet_events":n_raw,
 "n_sha256_unique_texts":n_unique,
 "duplicate_event_rows":index["duplicate_text_rows"],
 "shards_covered":64,
 "sentiment_kernels_complete":64,
 "stance_kernels_complete":64,
 "english_screened_unique":english_unique,
 "english_screened_raw_events":english_raw,
 "predicted_sentiment_unique":labels(unique,"multilingual_sentiment"),
 "predicted_sentiment_original_events":labels(event_year,"multilingual_sentiment"),
 "predicted_english_sentiment_unique":labels(unique,"english_sentiment"),
 "predicted_english_sentiment_original_events":labels(event_year,"english_sentiment"),
 "predicted_exploratory_stance_unique_english":labels(unique,"deberta_stance"),
 "predicted_exploratory_stance_original_events_english":labels(event_year,"deberta_stance"),
 "predicted_stance_abstained_unique":sum(v["deberta_abstained"] for v in unique.values()),
 "predicted_stance_abstained_raw_events":sum(v["deberta_abstained"] for v in event_year.values()),
 "yearly_unique_text_statistics":dict(sorted(unique.items())),
 "yearly_original_event_statistics":dict(sorted(event_year.items())),
 "source_original_event_statistics":dict(sorted(event_source.items())),
 "model_revisions":{k:sorted(v) for k,v in sorted(revisions.items())},
 "interpretation":"Model-generated labels are descriptive predictions; they are NOT independently verified vaccination sentiment or stance ground truth.",
 "accuracy_measured":False,
 "stance_prevalence_validated":False,
 "language_screen_independently_validated":False,
 "medical_causality_evaluated":False,
 "all_source_text_kept_private":True,
 "record_level_prediction_data_exported":False,
}
(OUT/"full_corpus_research_report.json").write_text(json.dumps(result,indent=2))
doc=[
"# Full vaccination Twitter corpus: model-predicted sentiment and stance",
"",
"## Source population and coverage",
f"Original rows: {result['n_original_source_rows']:,}; unique exact texts: {n_unique:,}; duplicate-text occurrences: {result['duplicate_event_rows']:,}.",
f"Complete model coverage across 64 sentiment and 64 exploratory stance shards.",
f"Automatic English screen: {english_unique:,} distinct texts; {english_raw:,} original nonblank events.",
"",
"## Multilingual emotional sentiment predictions",
"Unique-text labels: "+json.dumps(result["predicted_sentiment_unique"],sort_keys=True),
"Original-event-weighted labels: "+json.dumps(result["predicted_sentiment_original_events"],sort_keys=True),
"",
"## Exploratory English zero-shot vaccine stance predictions",
"Unique-text labels: "+json.dumps(result["predicted_exploratory_stance_unique_english"],sort_keys=True),
"Original-event-weighted labels: "+json.dumps(result["predicted_exploratory_stance_original_events_english"],sort_keys=True),
"",
"## Research limitations",
"The English language screen is automatic, not independently confirmed.",
"The vaccine stance model had substantial cross-model disagreement in development and did not receive a matched independent validation dataset.",
"The report counts predicted stance classes, NOT measured vaccine support/opposition prevalence or vaccine-related medical causality.",
"Counts for deduplicated texts and counts for repeated source events are intentionally separate; detailed data stay in private Kaggle."
]
(OUT/"full_corpus_research_report.md").write_text("\n".join(doc)+"\n")
print("FULL_CORPUS_RESEARCH_REPORT_COMPLETE",json.dumps({
    "raw_events":n_raw,"unique_texts":n_unique,
    "english_unique":english_unique,
    "sentiment_64_shards":True,"stance_64_shards":True,
    "accuracy_not_measured":True}),flush=True)
