"""Reconstruct raw tweet-event multiplicity for indexed, unique-text corpus.

This PRIVATE Kaggle CPU stage maps every nonblank source tweet to its full
SHA256 text key, source file and UTC year. No text or IDs appear in output.
Allows eventual aggregation of model predictions per raw tweet event and per
unique text without conflating the two denominators.
"""
import collections,csv,gzip,hashlib,json
from contextlib import ExitStack
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

ROOT=Path("/kaggle/input")
OUT=Path("/kaggle/working/vax_outputs")
SHARDS=64
SOURCES=("master.csv","vaccination2.csv")
TEXT_NAMES=("text","tweet","tweet_text","full_text","content","original_text",
            "tweet content","message","fulltext")
DATE_NAMES=("date","created_at","datetime","created","timestamp","tweet_date","time","createdat")
def find(filename,slug):
    a=[p for p in ROOT.rglob(filename) if slug in str(p).lower()]
    if len(a)!=1:raise RuntimeError("Missing/ambiguous private input "+filename)
    return a[0]
def findcol(columns,candidates):
    c={v.lower().strip():v for v in columns}
    return next((c[x] for x in candidates if x in c),None)
def parse_year(v):
    values=pd.to_datetime(v,format="mixed",utc=True,errors="coerce")
    return [int(x) if pd.notna(x) and 2000<=int(x)<=2026 else 0
            for x in values.dt.year]
def main():
    index=json.loads(find("full_corpus_manifest.json",
                          "vax-full-corpus-index-20261019").read_text(encoding="utf-8"))
    if (index.get("shard_count")!=64 or index.get("raw_rows_total")!=2285081 or
        index.get("unique_texts")!=1944361 or
        index.get("duplicate_text_rows")!=340720):
        raise RuntimeError("Unexpected frozen Kaggle full-corpus index identity")
    dest=OUT/"corpus_events"
    dest.mkdir(parents=True,exist_ok=True)
    src=collections.Counter();years=collections.Counter();bins=collections.Counter()
    with ExitStack() as stack:
        writers=[]
        for i in range(SHARDS):
            f=stack.enter_context(gzip.open(dest/("occurrence_%02d.csv.gz"%i),
                                            "wt",encoding="utf-8",
                                            newline="",compresslevel=4))
            w=csv.writer(f);w.writerow(("record_key","year","source"))
            writers.append(w)
        for filename in SOURCES:
            path=find(filename,"twitter-vaccination-dataset")
            cols=list(pd.read_csv(path,nrows=0).columns)
            text_col=findcol(cols,TEXT_NAMES)
            date_col=findcol(cols,DATE_NAMES)
            if not text_col:raise RuntimeError("Missing text column in "+filename)
            usecols=[text_col]+([date_col] if date_col and date_col!=text_col else [])
            for chunk in pd.read_csv(path,encoding="utf-8-sig",dtype=str,
                                     keep_default_na=False,usecols=usecols,
                                     chunksize=16000,low_memory=False,
                                     on_bad_lines="error"):
                src[filename+"_raw"]+=len(chunk)
                yr=parse_year(chunk[date_col]) if date_col else [0]*len(chunk)
                for text,year in zip(chunk[text_col].astype(str),yr):
                    if not text.strip():
                        src[filename+"_blank"]+=1
                        continue
                    digest=hashlib.sha256(text.encode("utf-8")).digest()
                    b=digest[0]>>2
                    writers[b].writerow((digest.hex(),year,filename[:-4]))
                    bins[b]+=1
                    src[filename+"_nonempty"]+=1
                    years[str(year)]+=1
    total=sum(bins.values())
    if total!=index["raw_rows_total"]-index["blank_rows"]:
        raise RuntimeError("Raw tweet event counts differ from full index")
    if any(bins[i]<int(index["shard_unique_counts"]["%02d"%i])
           for i in range(SHARDS)):
        raise RuntimeError("Raw event shard has fewer rows than unique-text shard")
    result={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "original_raw_rows":index["raw_rows_total"],
        "unique_texts":index["unique_texts"],
        "total_nonempty_events":total,
        "duplicate_text_rows":index["duplicate_text_rows"],
        "shard_count":SHARDS,
        "event_counts_by_shard":{"%02d"%k:int(v) for k,v in sorted(bins.items())},
        "event_year_counts":dict(sorted(years.items(),key=lambda a:int(a[0]))),
        "source_row_coverage":dict(src),
        "exact_sha256_text_join":True,
        "future_aggregation_can_report_event_weighted_counts":True,
        "model_labels_not_reference_gold":True,
        "source_text_included_in_event_files":False,
        "record_level_rows_private":True,
        "github_only_aggregates":True}
    (OUT/"full_corpus_occurrence_manifest.json").write_text(
        json.dumps(result,indent=2),encoding="utf-8")
    print("FULL_CORPUS_OCCURRENCE_BACKFILL",json.dumps({
        "nonempty_event_rows":total,"unique_texts":result["unique_texts"],
        "shards":SHARDS,"duplicate_rows":result["duplicate_text_rows"]
    }),flush=True)
if __name__=="__main__":main()
