"""CPU-only, metadata-only check of independent published vaccine stance labels.

Public VaxxHesitancy dataset (Mu et al., ICWSM 2023), Zenodo record 7601328,
V2, CC BY 4.0. No raw tweets or labels per row are printed in public CI.
The held-out TEST file is not used to tune prompts or pick model settings.
"""
import csv,io,json,urllib.request
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

URLS={
 "test":"https://zenodo.org/records/7601328/files/VaccineHesitancy_test.csv?download=1",
 "training":"https://zenodo.org/records/7601328/files/VaccineHesitancy_train_v2.csv?download=1",
}
OUT=Path("/kaggle/working/vax_outputs")
OUT.mkdir(parents=True,exist_ok=True)
FILES={}
for key,url in URLS.items():
    req=urllib.request.Request(url,headers={"User-Agent":"vaccination-sentiment-research/1.0"})
    with urllib.request.urlopen(req,timeout=75) as response:
        data=response.read(1_500_001)
    if len(data)>1_500_000:raise RuntimeError("Public benchmark file unexpectedly large")
    text=data.decode("utf-8-sig",errors="strict")
    reader=csv.DictReader(io.StringIO(text))
    cols=list(reader.fieldnames or [])
    if not cols:raise RuntimeError("Public benchmark file missing header")
    rows=list(reader)
    if len(rows)<30:raise RuntimeError("Public benchmark download unexpectedly small")
    # Only structural metadata. No example tweet, personal identifiers,
    # individual annotations or private benchmark examples are exported.
    counts={}
    candidates=[]
    for column in cols:
        name=column.lower()
        if any(v in name for v in ("label","stance","annotat","category","class")):
            candidates.append(column)
            # Use counts only when bounded categorical diversity is likely.
            tally=Counter(str(row.get(column,"")).strip() for row in rows)
            if len(tally)<=30:
                counts[column]=dict(sorted(tally.items()))
            else:
                counts[column]={"cardinality":len(tally)}
    FILES[key]={
        "filename":url.split("/")[-1].split("?")[0],
        "columns":cols,
        "n_rows":len(rows),
        "possible_label_columns":candidates,
        "possible_label_distributions":counts,
        "row_level_data_exported":False,
        "source_type":"published_independently_annotated_public_benchmark",
    }
report={
 "dataset":"VaxxHesitancy (Mu et al., ICWSM 2023)",
 "url":"https://zenodo.org/records/7601328",
 "doi":"10.5281/zenodo.7601328",
 "license":"CC BY 4.0",
 "version":"V2",
 "language":"English",
 "created_utc":datetime.now(timezone.utc).isoformat(),
 "files":FILES,
 "evaluation_plan":"Use held-out test ONLY with frozen v2 prompt. Restrict exact stance scoring to aligned labels and report nonaligned hesitant/irrelevant separately.",
 "historical_domain_shift":"2020s COVID-19 versus historical vaccination tweets 2006–2019",
 "accuracy_evaluated":False,
 "raw_text_exported":False,
}
(OUT/"external_benchmark_schema.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print("EXTERNAL_REFERENCE_SCHEMA_VALIDATED",json.dumps({
 "doi":report["doi"],
 "file_rows":{name:x["n_rows"] for name,x in FILES.items()},
 "possible_label_columns":{name:x["possible_label_columns"] for name,x in FILES.items()},
}),flush=True)
