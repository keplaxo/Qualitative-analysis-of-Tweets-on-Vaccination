"""Check private Kaggle run status, retrieving only non-text research summaries."""
import json, os, pathlib, re, subprocess, sys, tempfile, time

TASKS={
  "schema_probe":{"ref":"keplaxo/vax-schema-probe-20261017","report":"schema_probe.json"},
  "benchmark_smoke":{"ref":"keplaxo/vax-sentiment-benchmark-smoke","report":"summary.json"},
  "benchmark_full":{"ref":"keplaxo/vax-sentiment-benchmark-full","report":"summary.json"},
}

def command(*args,timeout=90):
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)

def one(label,cfg):
    proc=command("kernels","status",cfg["ref"])
    state=(proc.stdout+"\n"+proc.stderr).strip()
    if proc.returncode:
        return {"run":label,"state":"not_available"}
    found=re.findall(r"KernelWorkerStatus\.([A-Za-z_]+)",state)
    state=found[-1] if found else (
        "complete" if re.search(r"\bcomplete\b",state,re.I) else "unknown")
    record={"run":label,"state":state,"notebook":"https://www.kaggle.com/code/"+cfg["ref"]}
    if state.casefold() in ("error","failed"):
        # Never emit raw kernel stderr: it could contain private tweet content.
        logs=command("kernels","logs",cfg["ref"],timeout=75)
        if logs.returncode == 0:
            raw=logs.stdout.lower()
            if "duplicate prediction key" in raw:
                record["failure_category"]="duplicate_processing_keys"
            elif "master.csv not mounted" in raw:
                record["failure_category"]="dataset_mount_path"
            elif "no module named" in raw or "modulenotfounderror" in raw:
                record["failure_category"]="python_package_missing"
            elif "out of memory" in raw or "cuda error" in raw:
                record["failure_category"]="gpu_memory_or_cuda"
            else:
                record["failure_category"]="unknown_review_private_kaggle_logs"
    if state.casefold() in ("complete","completed","success","succeeded"):
        with tempfile.TemporaryDirectory() as td:
            dest=pathlib.Path(td)
            p=command("kernels","output",cfg["ref"],"-p",str(dest),
                      "--file-pattern",cfg["report"],timeout=120)
            located=list(dest.rglob(cfg["report"]))
            if p.returncode==0 and located:
                data=json.loads(located[0].read_text(encoding="utf-8"))
                if label=="schema_probe":
                    record["report"]=data
                else:
                    # Safe report is counts and model IDs, never per-tweet text.
                    record["report"]=data
            else:
                record["report"]="summary report not yet available"
    return record

def main():
    if not (os.environ.get("KAGGLE_API_TOKEN") or
            (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        raise RuntimeError("Kaggle credentials missing")
    # Re-check all private runs after the verified schema-preflight completion.
    records=[one(label,cfg) for label,cfg in TASKS.items()]
    print(json.dumps(records,indent=2))
    summary=os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary,"a") as f:
            f.write("## Vaccination Kaggle run status\n\n")
            for rec in records:
                f.write("- "+rec["run"]+": **"+rec["state"]+"**\n")
            f.write("\nNo tweet text, model prompts, identifiers or API secrets were exported.\n")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(type(e).__name__+": "+str(e),file=sys.stderr)
        sys.exit(1)
