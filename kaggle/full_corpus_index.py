"""Full-corpus private Kaggle CPU index: exact text dedup, 64 hash shards.

Reads BOTH hosted source CSVs once using bounded pandas chunks; all original
tweet text remains PRIVATE Kaggle output, never transmitted to GitHub. This
stage performs no inference or external data transfers. Master rows take
precedence over duplicates in supplementary vaccination2.csv.
"""
import collections,csv,gzip,hashlib,json,os,re,sys
from contextlib import ExitStack
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

DATASET="twitter-vaccination-dataset"
SHARDS=64
READ_CHUNK=16000
OUT=Path("/kaggle/working/vax_outputs")
B=OUT/"corpus_shards"
B.mkdir(parents=True,exist_ok=True)
TEXT_NAMES=("text","tweet","tweet_text","full_text","content","original_text",
            "tweet content","message","fulltext")
DATE_NAMES=("date","created_at","datetime","created","timestamp","tweet_date",
            "time","createdat")
FILE_NAMES=("master.csv","vaccination2.csv")
COLUMNS=("record_key","text_hash","text","year","source","language_screen")

def source_csv(name):
    paths=[p for p in Path("/kaggle/input").rglob(name)
           if DATASET in str(p).lower()]
    if len(paths)!=1:
        raise RuntimeError("Expected one Kaggle private source "+name+
                           "; mounted candidates="+str(len(paths)))
    return paths[0]

def column(headers,options):
    lower={str(x).casefold().strip():x for x in headers}
    return next((lower[x] for x in options if x in lower),None)

def sanitized_years(values):
    try:
        date=pd.to_datetime(values,format="mixed",utc=True,errors="coerce")
    except (TypeError,ValueError):
        date=pd.to_datetime(values,utc=True,errors="coerce")
    years=date.dt.year.fillna(0).astype(int)
    # Out of research scope/year, anomalous or missing dates: keep records
    # but store year=0 for conservative year-stratified reporting.
    return [int(y) if 2000<=int(y)<=2026 else 0 for y in years]

def main():
    seen=set()
    stats=collections.Counter()
    yearly=collections.Counter()
    by_source=collections.defaultdict(collections.Counter)
    shards=collections.Counter()
    files=[]
    with ExitStack() as stack:
        writers=[]
        for i in range(SHARDS):
            fn=B/("shard_%02d.csv.gz"%i)
            f=stack.enter_context(gzip.open(fn,"wt",encoding="utf-8",
                                            newline="",compresslevel=4))
            writer=csv.writer(f)
            writer.writerow(COLUMNS)
            writers.append(writer)
        for name in FILE_NAMES:
            path=source_csv(name)
            headers=list(pd.read_csv(path,nrows=0,encoding="utf-8-sig").columns)
            text_col=column(headers,TEXT_NAMES)
            date_col=column(headers,DATE_NAMES)
            if text_col is None:
                raise RuntimeError("Unable to find tweet text in "+name+
                                   "; user review required")
            record_source=name[:-4]
            src=collections.Counter()
            usecols=[text_col]+([date_col] if date_col and date_col!=text_col else [])
            for chunk in pd.read_csv(
                path,dtype=str,keep_default_na=False,encoding="utf-8-sig",
                usecols=usecols,
                chunksize=READ_CHUNK,low_memory=False,on_bad_lines="error"
            ):
                src["raw_rows"]+=len(chunk)
                all_text=chunk[text_col].astype(str)
                years=(sanitized_years(chunk[date_col])
                       if date_col else [0]*len(chunk))
                for tweet,yr in zip(all_text,years):
                    if not tweet.strip():
                        src["empty_text"]+=1
                        continue
                    h=hashlib.sha256(tweet.encode("utf-8")).digest()
                    if h in seen:
                        src["duplicate_text"]+=1
                        continue
                    seen.add(h)
                    bucket=h[0]>>2 # first six digest bits -> 64 shards
                    digest=h.hex()
                    writers[bucket].writerow((digest,digest,tweet,yr,
                                              record_source,"not_screened"))
                    src["unique_texts"]+=1
                    shards[str(bucket)]+=1
                    yearly[str(yr)]+=1
            stats.update(src)
            by_source[name].update(src)
            files.append({
                "filename":name,
                "bytes":path.stat().st_size,
                "source_headers":headers,
                "text_column":text_col,
                "date_column":date_col,
                "date_absent":date_col is None,
                "raw_rows":src["raw_rows"],
                "unique_texts":src["unique_texts"],
                "duplicate_text":src["duplicate_text"],
                "empty_text":src["empty_text"]
            })
            print("FULL_INDEX_SOURCE",json.dumps({
                "source":name,"rows":src["raw_rows"],
                "unique":src["unique_texts"],"duplicates":src["duplicate_text"],
                "blank":src["empty_text"]}),flush=True)
    total=sum(shards.values())
    if total!=len(seen) or total!=stats["unique_texts"]:
        raise RuntimeError("Dedup/partition coverage mismatch")
    sizes=[(B/("shard_%02d.csv.gz"%i)).stat().st_size for i in range(SHARDS)]
    if any(v==0 for v in shards.values()):
        raise RuntimeError("Corpus partition empty: verify hash spread")
    manifest={
       "created_utc":datetime.now(timezone.utc).isoformat(),
       "input_dataset":"keplaxo/twitter-vaccination-dataset",
       "input_files":files,
       "raw_rows_total":int(stats["raw_rows"]),
       "unique_texts":int(total),
       "blank_rows":int(stats["empty_text"]),
       "duplicate_text_rows":int(stats["duplicate_text"]),
       "sha256_exact_text_dedup":True,
       "raw_duplicate_multiplicity_preserved":False,
       "source_order":list(FILE_NAMES),
       "shard_count":SHARDS,
       "shard_key":"top six bits of SHA256 exact UTF-8 tweet text",
       "shard_unique_counts":{"%02d"%i:int(shards[str(i)]) for i in range(SHARDS)},
       "shard_compressed_bytes":{"%02d"%i:int(sizes[i]) for i in range(SHARDS)},
       "year_counts":dict(sorted(yearly.items(),key=lambda x:int(x[0]))),
       "per_source":{name:dict(by_source[name]) for name in FILE_NAMES},
       "date_convention":"UTC parse, year 0 for missing/out-of-range, no row dropped solely for date",
       "sentiment_model":"cardiffnlp/twitter-xlm-roberta-base-sentiment",
       "english_model":"cardiffnlp/twitter-roberta-base-sentiment-latest",
       "stance_model":"MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
       "further_stage_requires_capacity_measurement":True,
       "source_records_private":True,
       "individual_tweet_data_exported_to_github":False,
       "no_classification_accuracy_measured":True,
       "no_full_corpus_prevalence_claims_yet":True,
    }
    (OUT/"full_corpus_manifest.json").write_text(
        json.dumps(manifest,indent=2),encoding="utf-8")
    print("FULL_CORPUS_INDEX_COMPLETE",json.dumps({
       "raw":manifest["raw_rows_total"],"unique":total,
       "shards":SHARDS,"dup":manifest["duplicate_text_rows"],
       "blank":manifest["blank_rows"]}),flush=True)

if __name__=="__main__":main()
