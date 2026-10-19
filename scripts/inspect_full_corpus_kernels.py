"""Safe Kaggle registry look-up (operational metadata only)."""
import csv,io,os,re,subprocess,sys
TARGETS=("corpus","event","vax-full","vaccination")
def main():
    if not (os.getenv("KAGGLE_API_TOKEN") or
            (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("Kaggle auth not set")
    p=subprocess.run(["kaggle","kernels","list","--mine","--csv"],
                     capture_output=True,text=True,timeout=100)
    if p.returncode:raise RuntimeError("Unable to read owned Kaggle notebooks")
    reader=csv.DictReader(io.StringIO(p.stdout))
    if not reader.fieldnames:raise RuntimeError("Missing Kaggle registry columns")
    fields=tuple(reader.fieldnames)
    print("SAFE_KERNEL_REGISTRY_FIELDS",fields,flush=True)
    shown=0
    for row in reader:
        info=(row.get("ref","")+" "+row.get("title","")).lower()
        if not any(t in info for t in TARGETS):continue
        public={key:str(val)[:125] for key,val in row.items() if
                key and key.lower() in ("ref","title","status","last_run_time",
                                         "lastrun","last_run")}
        # Never print user-created notebook content or results.
        if public:
            print("SAFE_OWNED_NOTEBOOK",public,flush=True)
            shown+=1
    print("SAFE_OWNED_NOTEBOOKS_MATCHED",shown,flush=True)
if __name__=="__main__":
    try:main()
    except Exception as e:
        print("SAFE_REGISTRY_BLOCKED",type(e).__name__,file=sys.stderr)
        sys.exit(1)
