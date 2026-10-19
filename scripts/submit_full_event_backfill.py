"""Submit private CPU raw-event frequency backfill, idempotent."""
import json,os,pathlib,re,subprocess,sys,tempfile
REF="keplaxo/vax-full-corpus-occurrences-20261019"
def call(*a,timeout=110):
    return subprocess.run(["kaggle",*a],capture_output=True,text=True,timeout=timeout)
def main():
    if not (os.getenv("KAGGLE_API_TOKEN") or
            (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("Kaggle credentials unavailable")
    p=call("kernels","status",REF)
    if not p.returncode:
        m=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout)
        print("CORPUS_EVENTS_ALREADY_REGISTERED",m[-1] if m else "UNKNOWN")
        return
    owned=call("kernels","list","--mine","--csv")
    if owned.returncode or REF.lower() in owned.stdout.lower():
        raise RuntimeError("Ambiguous existing raw events kernel; no duplicate submission")
    p=call("kernels","status","keplaxo/vax-full-corpus-index-20261019")
    if p.returncode or "KernelWorkerStatus.COMPLETE" not in p.stdout:
        raise RuntimeError("Full-corpus index not complete")
    script=pathlib.Path("kaggle/full_corpus_occurrence_backfill.py").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        d=pathlib.Path(tmp);f="full_corpus_occurrence_backfill.py"
        (d/f).write_text(script,encoding="utf-8")
        meta={"id":REF,"title":"Vax Full Corpus Raw Event Weights 20261019",
              "code_file":f,"language":"python","kernel_type":"script",
              "is_private":True,"enable_gpu":False,"enable_internet":False,
              "dataset_sources":["keplaxo/twitter-vaccination-dataset"],
              "kernel_sources":["keplaxo/vax-full-corpus-index-20261019"],
              "competition_sources":[],"model_sources":[]}
        (d/"kernel-metadata.json").write_text(json.dumps(meta))
        p=call("kernels","push","-p",str(d),"--timeout","10800",timeout=240)
        if p.returncode:raise RuntimeError("Private event backfill rejected")
    print("FULL_EVENT_BACKFILL_SUBMITTED",REF,"gpu=False",flush=True)
if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("FULL_EVENT_BACKFILL_BLOCKED",type(exc).__name__,str(exc),file=sys.stderr)
        sys.exit(1)
