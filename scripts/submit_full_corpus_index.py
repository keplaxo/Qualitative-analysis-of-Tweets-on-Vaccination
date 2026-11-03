"""One-time Kaggle CPU index launch; never overwrite previous corpus shards."""
import json,os,pathlib,re,subprocess,sys,tempfile
KERNEL="keplaxo/vax-full-corpus-index-20261019"
DATASET="keplaxo/twitter-vaccination-dataset"
def cmd(*args,timeout=140):
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)
def main():
    if not (os.getenv("KAGGLE_API_TOKEN") or
            (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("Missing existing Kaggle credentials")
    p=cmd("kernels","status",KERNEL)
    if p.returncode==0:
        m=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout)
        print("FULL_INDEX_ALREADY_REGISTERED",m[-1] if m else "UNKNOWN")
        return
    owned=cmd("kernels","list","--mine","--csv")
    if owned.returncode:raise RuntimeError("Kaggle own-kernel lookup failed")
    if KERNEL.lower() in owned.stdout.lower():
        raise RuntimeError("Registered kernel inaccessible; no duplicate submission")
    source=pathlib.Path("kaggle/full_corpus_index.py").read_text(encoding="utf-8")
    if "SHARDS=64" not in source or 'FILE_NAMES=("master.csv","vaccination2.csv")' not in source:
        raise RuntimeError("Full-corpus source/partition contract changed")
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp)
        (root/"full_corpus_index.py").write_text(source,encoding="utf-8")
        metadata={
           "id":KERNEL,"title":"Vax Full Corpus Index 20261019",
           "code_file":"full_corpus_index.py","language":"python",
           "kernel_type":"script","is_private":True,
           "enable_gpu":False,"enable_internet":False,
           "dataset_sources":[DATASET],"kernel_sources":[],
           "competition_sources":[],"model_sources":[]}
        (root/"kernel-metadata.json").write_text(json.dumps(metadata),encoding="utf-8")
        p=cmd("kernels","push","-p",str(root),"--timeout","10800",timeout=240)
        if p.returncode:
            msg=(p.stdout+" "+p.stderr).lower()
            cat="conflict" if "409" in msg else "quota" if "quota" in msg else "other"
            raise RuntimeError("Private corpus index rejected: "+cat)
    print("FULL_INDEX_SUBMITTED",KERNEL,"gpu=False",
          "source_files=2","shards=64",flush=True)
if __name__=="__main__":
    try:main()
    except Exception as err:
        print("FULL_INDEX_BLOCKED",type(err).__name__,str(err),file=sys.stderr)
        sys.exit(1)
