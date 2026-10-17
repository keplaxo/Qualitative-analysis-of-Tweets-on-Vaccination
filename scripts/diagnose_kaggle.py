"""Diagnose Kaggle notebook errors without dumping tweet rows into public CI logs."""
import os,re,subprocess,sys,csv,io,json,tempfile,pathlib
REFS=("keplaxo/vax-schema-probe-20261017",
      "keplaxo/vax-sentiment-benchmark-smoke",
      "keplaxo/vax-sentiment-benchmark-full")
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
        if ref.endswith("-full"):
            try:
                events=json.loads(p.stdout)
                raw="\n".join(str(e.get("data","")) for e in events if isinstance(e,dict))
                marks=("GPU runtime:","Dataset mounted:","Repository revision:",
                       "Downstream models will use the language-screened copy",
                       "Running:","xlmt_sentiment.jsonl","roberta_english_sentiment.jsonl",
                       "deberta_stance.jsonl","summary.json","Traceback","Error",
                       "out of memory","quota","cancel","interrupt")
                print("FULL_NOTEBOOK_LOG_EVENTS:",len(events))
                print("FULL_NOTEBOOK_MILESTONES:",[v for v in marks if v.lower() in raw.lower()])
                print("FULL_NOTEBOOK_LAST_EVENT_TIME_SECONDS:",max([float(e.get("time",0)) for e in events if isinstance(e,dict)] or [0]))
                print("FULL_NOTEBOOK_STREAMS:",{n:sum(e.get("stream_name")==n for e in events if isinstance(e,dict)) for n in ("stdout","stderr")})
            except (ValueError,TypeError):
                raw=p.stdout
                print("FULL_NOTEBOOK_LOG_RESPONSE:",{
                    "characters":len(raw),
                    "line_count":len(raw.splitlines()),
                    "has_traceback":"Traceback" in raw,
                    "has_python_exception":"Exception encountered" in raw,
                    "has_gpu_runtime_marker":"GPU runtime:" in raw,
                    "has_dataset_mount_marker":"Dataset mounted:" in raw,
                    "has_repository_revision_marker":"Repository revision:" in raw,
                    "has_language_screen_marker":"Downstream models will use" in raw,
                    "has_model_inference_marker":"Running: " in raw,
                    "has_summary_file_marker":"summary.json" in raw,
                    "contains_queued_message":"queued" in raw.lower(),
                    "contains_cancel_message":"cancel" in raw.lower(),
                    "contains_no_logs_message":"no logs" in raw.lower(),
                })
                # Only request an allowed non-sensitive aggregate summary.
                with tempfile.TemporaryDirectory() as tmp:
                    out=subprocess.run(["kaggle","kernels","output",ref,
                                        "-p",tmp,"--file-pattern","summary.json"],
                        capture_output=True,text=True,timeout=75)
                    found=list(pathlib.Path(tmp).rglob("summary.json"))
                    print("FULL_NOTEBOOK_SUMMARY_FILE:",bool(found))
                    print("FULL_NOTEBOOK_OUTPUT_API_SUCCESS:",out.returncode==0)
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
