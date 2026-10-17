"""Diagnose Kaggle notebook errors without dumping tweet rows into public CI logs."""
import os,re,subprocess,sys,csv,io,json,tempfile,pathlib
REFS=("keplaxo/vax-schema-probe-20261017",
      "keplaxo/vax-sentiment-benchmark-smoke")
PAT=re.compile(r"Traceback|Exception|Error|No module named|No such file|not found|"
               r"ModuleNotFound|ImportError|OSError|RuntimeError|ValueError|"
               r"FileNotFound|SyntaxError|KeyError|AssertionError|"
               r"^\\s*File |Failed to|error:|out of memory|Cannot find column",re.I)
def main():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN secret unavailable")
    p=subprocess.run(["kaggle","kernels","list","--mine","--csv"],
         capture_output=True,text=True,timeout=60)
    if p.returncode==0:
        for line in p.stdout.splitlines():
            if re.search(r"vax|vaccinat|schema",line,re.I):
                print("OWNED_KERNEL: "+line[:400])
    for ref in REFS:
        print("NOTEBOOK: "+ref)
        p=subprocess.run(["kaggle","kernels","status",ref],
                         capture_output=True,text=True,timeout=60)
        if p.returncode==0:
            state=re.search(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout)
            print("STATE: "+(state.group(1) if state else "unknown"))
        else:
            print("STATE: not_available")
        p=subprocess.run(["kaggle","kernels","logs",ref],
                         capture_output=True,text=True,timeout=90)
        if p.returncode:
            print("LOGS: unavailable (exit "+str(p.returncode)+")")
            print("DIAGNOSTIC: "+(p.stderr or p.stdout).strip()[:280])
            continue
        lines=[v for v in p.stdout.splitlines() if PAT.search(v)]
        print("DIAGNOSTIC_LINES: "+str(len(lines)))
        for line in lines[-35:]:
            # Avoid printing source text, credentials and long model generated text.
            if len(line)>420: line=line[:420]+"...(truncated)"
            print(line)
        if not lines:
            print("No exception or error markers found in Kaggle logs.")
        print("DIAGNOSTICS_CHECKED: most recent Kaggle run, 17 October")
        try:
            events=json.loads(p.stdout)
            for event in events:
                if event.get("stream_name")=="stdout":
                    for line in str(event.get("data","")).splitlines():
                        if ("Available input CSVs" in line or
                            "/kaggle/input/" in line or
                            "Dataset mounted" in line):
                            print("SAFE_PATH_DIAGNOSTIC: "+line[:300])
        except (ValueError,TypeError):
            pass
        with tempfile.TemporaryDirectory() as folder:
            pull=subprocess.run(["kaggle","kernels","pull",ref,"-p",folder,"-m"],
                capture_output=True,text=True,timeout=90)
            metas=list(pathlib.Path(folder).rglob("kernel-metadata.json"))
            if pull.returncode==0 and metas:
                obj=json.loads(metas[0].read_text())
                print("KERNEL_METADATA:",json.dumps({
                    "id":obj.get("id"),"title":obj.get("title"),
                    "dataset_sources":obj.get("dataset_sources"),
                    "is_private":obj.get("is_private"),
                    "enable_gpu":obj.get("enable_gpu")}))

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(type(exc).__name__+": "+str(exc),file=sys.stderr)
        sys.exit(1)
