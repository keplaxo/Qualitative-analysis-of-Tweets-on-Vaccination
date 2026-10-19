"""Find vaccination datasets owned by authenticated Kaggle user using metadata only."""
import csv, io, json, os, re, subprocess, sys
HINTS = re.compile(r"vaccin|vax|immuniz|immunis|twitter|tweets?", re.I)
EXACT = re.compile(r"^master\.csv$|vaccin.*\.(csv|parquet)$",re.I)

def cli(*args):
    proc=subprocess.run(["kaggle",*args],capture_output=True,text=True,
                        timeout=90,env=os.environ.copy())
    if proc.returncode:
        raise RuntimeError("Kaggle command failed: "+args[0]+" "+args[1]+
                           " (exit "+str(proc.returncode)+"). Check credentials.")
    return proc.stdout

def csv_rows(data):
    lines=data.strip().splitlines()
    pos=next((i for i,v in enumerate(lines)
          if v.lower().lstrip().startswith(("ref,","reference,","name,"))),None)
    if pos is None:
        if not data.strip() or "no datasets" in data.lower(): return []
        raise RuntimeError("Unexpected Kaggle metadata format")
    return list(csv.DictReader(io.StringIO("\n".join(lines[pos:]))))

def normal(row):
    d={str(k).strip().lower():str(v or "").strip() for k,v in row.items()}
    return {"ref":d.get("ref") or d.get("reference") or d.get("name"),
            "title":d.get("title","")}

def main():
    if not (os.environ.get("KAGGLE_API_TOKEN") or
            (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        raise RuntimeError("No Kaggle secret detected: use KAGGLE_API_TOKEN or legacy KAGGLE_USERNAME/KAGGLE_KEY")
    listed=[]
    for page in range(1,11):
        rows=csv_rows(cli("datasets","list","--mine","--csv","--sort-by","updated","--page",str(page)))
        if not rows: break
        listed.extend(map(normal,rows))
        if len(rows)<20: break
    favorites=[r for r in listed if HINTS.search(r["ref"]+" "+r["title"])]
    fallbacks=[r for r in listed if r not in favorites][:35]
    found=[]
    for item in favorites+fallbacks:
        if "/" not in item["ref"]: continue
        try:
            rows=csv_rows(cli("datasets","files",item["ref"],"--csv"))
        except (RuntimeError,ValueError):
            rows=[]
        names=[]
        for r in rows:
            d={str(k).strip().lower():str(v or "").strip() for k,v in r.items()}
            name=d.get("name") or d.get("filename") or d.get("ref")
            if name: names.append(name)
        hits=[name for name in names if EXACT.search(name)]
        if item in favorites or hits:
            found.append({"ref":item["ref"],"title":item["title"],
                          "file_count":len(names),"master_or_vaccine_files":hits[:12],
                          "relevant_filenames":[v for v in names if HINTS.search(v)][:12],
                          "title_matched":item in favorites})
    found=sorted(found,key=lambda v:(not bool(v["master_or_vaccine_files"]),
                    not v["title_matched"],v["ref"]))
    print(json.dumps({"owned_datasets_listed":len(listed),"candidate_count":len(found),
                      "candidates":found[:15],"downloaded_raw_data":False},indent=2))
    if len(found)==1:
        with open(os.environ["GITHUB_OUTPUT"],"a") as out:
            out.write("dataset_ref="+found[0]["ref"]+"\n")
    elif len(found)>1:
        print("Multiple matching datasets; choose based on filenames.")
    else:
        print("No matching dataset metadata. Confirm account and dataset visibility.")

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(type(exc).__name__+": "+str(exc),file=sys.stderr)
        sys.exit(1)
