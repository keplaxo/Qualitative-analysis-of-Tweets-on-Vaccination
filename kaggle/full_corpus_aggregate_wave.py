"""Private CPU reporting wave: four matching sentiment/stance shards plus raw events.

Prints only anonymous aggregate counts. All record hash lists, individual
predictions and event joins stay in Kaggle mounted private notebook storage.
"""
WAVE_INDEX=WAVE_PLACEHOLDER
assert 0<=WAVE_INDEX<16
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import gzip,json
import pandas as pd

OUT=Path("/kaggle/working/vax_outputs");OUT.mkdir(parents=True,exist_ok=True)
SHARDS=range(WAVE_INDEX*4,(WAVE_INDEX+1)*4)
EVENTS="vax-corpus-event-join-20261019-v2"
def file(name,slug):
    matches=[p for p in Path("/kaggle/input").rglob(name)
             if slug in str(p).lower()]
    if len(matches)!=1:raise RuntimeError("Private model/occurrence file missing/ambiguous "+
                                          name+" in "+slug)
    return matches[0]
def jsonl(path):
    d={}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row=json.loads(line);k=row.get("record_key")
                if not k or k in d:raise RuntimeError("Repeated or missing private model hash")
                d[k]=row
    return d
def group():
    return {"rows":0,"multilingual_sentiment":Counter(),
            "english_sentiment":Counter(),"deberta_stance":Counter(),
            "english_eligible":0,"deberta_abstained":0}
def add(d,labels,english,abstained):
    d["rows"]+=1
    d["multilingual_sentiment"][labels[0]]+=1
    d["english_eligible"]+=int(english)
    if english:
        d["english_sentiment"][labels[1]]+=1
        d["deberta_stance"][labels[2]]+=1
        d["deberta_abstained"]+=int(abstained)
def stringify(x):
    return {str(k):(dict(sorted(v.items())) if isinstance(v,Counter) else v)
            for k,v in x.items()}
def clean_group(g):
    return {name:(dict(sorted(val.items())) if isinstance(val,Counter) else val)
            for name,val in g.items()}

unique_year=defaultdict(group)
raw_year=defaultdict(group)
raw_source=defaultdict(group)
model_revisions=defaultdict(set)
total_u=total_e=0
for i in SHARDS:
    sid="%02d"%i
    senti="vax-corpus-sentiment-"+sid+"-20261019"
    stance="vax-corpus-stance-"+sid+"-20261019"
    sreport=json.loads(file("corpus_shard_report.json",senti).read_text())
    nreport=json.loads(file("corpus_shard_report.json",stance).read_text())
    if (sreport.get("mode")!="sentiment" or nreport.get("mode")!="stance" or
        sreport.get("shard")!=i or nreport.get("shard")!=i or
        not sreport.get("coverage_complete") or not nreport.get("coverage_complete")):
        raise RuntimeError("Invalid completed model stage "+sid)
    frame=pd.read_csv(file("shard_screened.csv",senti),
                      dtype=str,keep_default_na=False,
                      usecols=["record_key","year","language_screen"])
    if frame.record_key.nunique()!=len(frame):
        raise RuntimeError("Repeated screened text hash "+sid)
    sm=jsonl(file("xlmt_sentiment.jsonl",senti))
    se=jsonl(file("roberta_english_sentiment.jsonl",senti))
    sn=jsonl(file("deberta_stance.jsonl",stance))
    keys=set(frame.record_key)
    ekeys=set(frame.loc[frame.language_screen.str.lower().isin(
        ("en","eng","english")),"record_key"])
    if (set(sm)!=keys or set(se)!=ekeys or set(sn)!=ekeys or
        len(keys)!=sreport["input_unique_n"] or
        len(ekeys)!=nreport["english_screened_n"]):
        raise RuntimeError("Full model coverage/hash alignment failed "+sid)
    # Never print or export text/hash keys; local tuple map supports
    # independent source-event weighting with each text inferred once.
    labels={}
    for rec in frame.to_dict("records"):
        k=rec["record_key"]
        is_en=k in ekeys
        a=str(sm[k]["model_label"]).lower()
        b=str(se[k]["model_label"]).lower() if is_en else ""
        c=str(sn[k]["label"]).lower() if is_en else ""
        abstain=bool(sn[k].get("abstained")) if is_en else False
        yr=str(rec["year"])
        labels[k]=(a,b,c,is_en,abstain)
        add(unique_year[yr],(a,b,c),is_en,abstain)
        model_revisions["xlmt"].add(str(sm[k].get("revision","unknown")))
        if is_en:
            model_revisions["roberta"].add(str(se[k].get("revision","unknown")))
            model_revisions["deberta"].add(str(sn[k].get("revision","unknown")))
    events=pd.read_csv(file("occurrence_"+sid+".csv.gz",EVENTS),
                       dtype=str,keep_default_na=False)
    if (set(events.columns)!=set(("record_key","year","source")) or
        not set(events.record_key).issubset(keys) or
        len(events)<len(keys)):
        raise RuntimeError("Raw source events lack exact-text matched predictions "+sid)
    for row in events.to_dict("records"):
        k=row["record_key"]
        a,b,c,english,abstained=labels[k]
        add(raw_year[str(row["year"])],(a,b,c),english,abstained)
        add(raw_source[str(row["source"])],(a,b,c),english,abstained)
    total_u+=len(keys)
    total_e+=len(events)
    print("REPORT_WAVE_SHARD_COMPLETE",sid,"unique",len(keys),
          "events",len(events),flush=True)
report={
    "generated_utc":datetime.now(timezone.utc).isoformat(),
    "wave":WAVE_INDEX,"shards":[int(i) for i in SHARDS],
    "unique_texts":total_u,"raw_events_nonempty":total_e,
    "by_year_unique":{k:clean_group(v) for k,v in sorted(unique_year.items())},
    "by_year_events":{k:clean_group(v) for k,v in sorted(raw_year.items())},
    "by_source_events":{k:clean_group(v) for k,v in sorted(raw_source.items())},
    "model_revisions":{k:sorted(v) for k,v in sorted(model_revisions.items())},
    "entire_wave_hash_coverage_verified":True,
    "sentiment_predicted_not_truth":True,
    "stance_unvalidated":True,
    "no_medical_causality_claims":True,
    "tweet_content_or_hashes_exported":False,
    "counting_unit_definitions":{
        "unique":"SHA256-distinct exact UTF-8 text",
        "events":"all nonblank original source CSV records including duplicate text"
    },
}
(OUT/"corpus_report_wave.json").write_text(json.dumps(report,indent=2))
print("CORPUS_REPORT_WAVE_COMPLETE",json.dumps({
    "wave":WAVE_INDEX,"shards":4,"unique":total_u,"events":total_e,
    "no_model_accuracy_claims":True}),flush=True)
