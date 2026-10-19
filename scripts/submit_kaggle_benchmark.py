"""Submit safe Kaggle benchmark from GitHub Actions, never uploading source tweets."""
import json, os, pathlib, shutil, subprocess, sys, tempfile, time

DATASET="keplaxo/twitter-vaccination-dataset"
MODES={
  "smoke":{"kernel":"keplaxo/vax-sentiment-benchmark-smoke",
           "title":"Vax Sentiment Benchmark Smoke","main_n":100,
           "enrich_n":0,"qwen":False},
  "full":{"kernel":"keplaxo/vax-sentiment-benchmark-full",
          "title":"Vax Sentiment Benchmark Full","main_n":2000,
          "enrich_n":0,"qwen":False},
}

def main():
    mode=os.environ.get("BENCHMARK_MODE","smoke").strip().lower()
    if mode not in MODES:
        raise ValueError("Unknown mode")
    if not (os.environ.get("KAGGLE_API_TOKEN") or
            (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        raise RuntimeError("Kaggle API authentication not configured")
    cfg=MODES[mode]
    if mode=="full":
        # Require the prior Kaggle smoke run to have completed and saved
        # all three outputs. Submission is not evidence of completion.
        prior=MODES["smoke"]["kernel"]
        check=subprocess.run(["kaggle","kernels","status",prior],
                             capture_output=True,text=True,timeout=60)
        if check.returncode or "KernelWorkerStatus.COMPLETE" not in check.stdout:
            raise RuntimeError("2,000-record stage blocked: smoke notebook not complete")
        with tempfile.TemporaryDirectory() as td:
            output=subprocess.run(
                ["kaggle","kernels","output",prior,"-p",td,
                 "--file-pattern","summary.json"],
                capture_output=True,text=True,timeout=120)
            summaries=list(pathlib.Path(td).rglob("summary.json"))
            if output.returncode or not summaries:
                raise RuntimeError("2,000-record stage blocked: smoke summary missing")
            prior_result=json.loads(summaries[0].read_text())
            coverage=prior_result.get("stage_coverage",{})
            if (prior_result.get("n_sampled")!=100 or
                coverage.get("multilingual_sentiment")!=100 or
                coverage.get("english_sentiment",0)<1 or
                coverage.get("zero_shot_stance",0)<1):
                raise RuntimeError("2,000-record stage blocked: incomplete model coverage")
        print("Validated 100-record smoke checkpoint; approved 2,000-record stage.")
    src=pathlib.Path("kaggle/vaccination_benchmark.ipynb")
    doc=json.loads(src.read_text(encoding="utf-8"))
    config_idx=next((i for i,c in enumerate(doc["cells"])
                     if "MAIN_N = 2000" in "".join(c["source"])),None)
    if config_idx is None:
        raise ValueError("Missing expected benchmark config cell")
    code="".join(doc["cells"][config_idx]["source"])
    code=code.replace("MAIN_N = 2000","MAIN_N = "+str(cfg["main_n"]))
    code=code.replace("ENRICHMENT_N = 1000","ENRICHMENT_N = "+str(cfg["enrich_n"]))
    code=code.replace("RUN_QWEN = False","RUN_QWEN = "+str(cfg["qwen"]))
    doc["cells"][config_idx]["source"]=code.splitlines(keepends=True)
    # Kaggle notebook must be private; inputs are mounted by Kaggle.
    with tempfile.TemporaryDirectory() as tmp:
        d=pathlib.Path(tmp)
        (d/"vaccination_benchmark.ipynb").write_text(json.dumps(doc,indent=1))
        meta={
            "id":cfg["kernel"],"title":cfg["title"],
            "code_file":"vaccination_benchmark.ipynb",
            "language":"python","kernel_type":"notebook",
            "is_private":True,
            "enable_gpu":True,"enable_internet":True,
            "dataset_sources":[DATASET],
            "competition_sources":[],"kernel_sources":[],"model_sources":[]
        }
        (d/"kernel-metadata.json").write_text(json.dumps(meta))
        cmd=["kaggle","kernels","push","-p",str(d),"--timeout","3600"]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
        if p.returncode:
            raise RuntimeError("Kaggle benchmark submission failed (exit "+
                               str(p.returncode)+"); inspect Kaggle account.")
        print("Successfully submitted private Kaggle "+mode+" GPU job.")
        print("Notebook URL: https://www.kaggle.com/code/"+cfg["kernel"])
        print("Input dataset: "+DATASET)
        print("Sample cap: "+str(cfg["main_n"])+" + enrichment "+
              str(cfg["enrich_n"])+"; Qwen enabled: "+str(cfg["qwen"]))
        # Best-effort immediate status, but do not confuse submission with completion.
        time.sleep(12)
        p=subprocess.run(["kaggle","kernels","status",cfg["kernel"]],
              capture_output=True,text=True,timeout=90)
        if p.returncode==0:
            print("Initial Kaggle status: "+p.stdout.strip()[:350])
        else:
            print("Notebook submitted; status not yet retrievable.")

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(type(exc).__name__+": "+str(exc),file=sys.stderr)
        sys.exit(1)
