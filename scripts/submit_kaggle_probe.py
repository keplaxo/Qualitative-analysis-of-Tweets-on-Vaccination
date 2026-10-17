"""Create a private Kaggle schema probe, poll it and display safe metadata."""
import json, os, pathlib, re, shutil, subprocess, sys, tempfile, time

DATASET="keplaxo/twitter-vaccination-dataset"
KERNEL="keplaxo/vax-schema-probe-20261017"

def invoke(*args,timeout=120,check=True):
    p=subprocess.run(["kaggle",*args],text=True,capture_output=True,timeout=timeout)
    if check and p.returncode:
        raise RuntimeError("Kaggle "+args[0]+" "+args[1]+
                           " failed (code "+str(p.returncode)+").")
    return p

def main():
    if not (os.environ.get("KAGGLE_API_TOKEN") or
            (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        raise RuntimeError("Kaggle secret unavailable")
    with tempfile.TemporaryDirectory() as tmp:
        d=pathlib.Path(tmp)
        shutil.copy2("kaggle/schema_probe.ipynb",d/"schema_probe.ipynb")
        metadata={
            "id":KERNEL,"title":"Vaccination Research Dataset Schema Probe",
            "code_file":"schema_probe.ipynb","language":"python",
            "kernel_type":"notebook","is_private":True,
            "enable_gpu":False,"enable_internet":False,
            "dataset_sources":[DATASET],"competition_sources":[],
            "kernel_sources":[],"model_sources":[]}
        (d/"kernel-metadata.json").write_text(json.dumps(metadata))
        invoke("kernels","push","-p",str(d),timeout=180)
        print("Private schema probe submitted. Notebook: https://www.kaggle.com/code/"+KERNEL)
        end=time.monotonic()+540
        while time.monotonic()<end:
            time.sleep(20)
            p=invoke("kernels","status",KERNEL,check=False)
            status=(p.stdout+" "+p.stderr).strip()
            if p.returncode: continue
            if re.search(r"\b(error|failed|cancelled|canceled)\b",status,re.I):
                raise RuntimeError("Kaggle probe failed with state: "+
                                   status[:220])
            if re.search(r"\b(complete|completed|success|succeeded)\b",status,re.I):
                print("Kaggle schema probe completed.")
                break
        else:
            raise TimeoutError("Kaggle schema probe still pending after 9min; check Kaggle notebook page.")
        out=d/"output"
        out.mkdir()
        invoke("kernels","output",KERNEL,"-p",str(out),
               "--file-pattern","schema_probe.json",timeout=120)
        matches=list(out.rglob("schema_probe.json"))
        if not matches:
            raise RuntimeError("Probe completed but schema JSON output is unavailable.")
        obj=json.loads(matches[0].read_text())
        print("Schema results (no record text):")
        print(json.dumps(obj,indent=2))
        master=next((v for v in obj["files"] if v["filename"]=="master.csv"),None)
        if not master or not master.get("readable"):
            raise RuntimeError("Dataset master.csv was not readable in Kaggle.")
        print("Master file attached and readable.")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(type(e).__name__+": "+str(e),file=sys.stderr)
        sys.exit(1)
