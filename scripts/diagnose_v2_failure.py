"""Read-only SAFE Kaggle v2 failure diagnostics, no tweet or prompt logging."""
import json,os,re,subprocess,sys
REF="keplaxo/vax-stance-v2-pilot-20261018"
ERRORS=("RuntimeError","ValueError","AssertionError","KeyError","IndexError",
        "TypeError","AttributeError","ImportError","ModuleNotFoundError",
        "CalledProcessError","OutOfMemoryError","CUDAError","OSError",
        "FileNotFoundError","SyntaxError","SystemExit","KeyboardInterrupt")
MARKERS=("V2_STAGE","Qwen 10/","Qwen 12/","V2_AGGREGATE","V2_SCHEMA_GATE",
         "Incomplete versioned Qwen v2 checkpoint",
         "v2 join and exact-text SHA-256 checks failed",
         "Expected exactly one","Insufficient frozen records",
         "V2_SCHEMA_GATE_BLOCKED","Traceback","CUDA out of memory",
         "out of memory","OOM","SIGKILL","SIGTERM","Killed",
         "DeadlineExceeded","exceeded time","AssertionError",
         "GPU","torch","vaxbench")
def call(*args):
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=140)
def main():
    if not os.getenv("KAGGLE_API_TOKEN"):
        raise RuntimeError("private credentials unavailable")
    s=call("kernels","status",REF)
    if s.returncode: raise RuntimeError("status API unavailable")
    state=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",s.stdout)
    print("KAGGLE_V2_STATE",state[-1] if state else "unknown")
    p=call("kernels","logs",REF)
    if p.returncode: raise RuntimeError("logs API unavailable")
    logs=json.loads(p.stdout)
    if not isinstance(logs,list):raise RuntimeError("unexpected Kaggle log format")
    messages=[str(x.get("data","")) for x in logs if isinstance(x,dict)]
    whole="\n".join(messages)
    print("SAFE_ANALYSIS",json.dumps({
      "events":len(logs),
      "elapsed_seconds":round(max([float(x.get("time",0)) for x in logs if isinstance(x,dict)] or [0]),1),
      "streams":{t:sum(str(x.get("stream_name"))==t for x in logs if isinstance(x,dict))
                 for t in sorted({str(x.get("stream_name")) for x in logs if isinstance(x,dict)})},
      "error_types":{err:len(re.findall(err,whole)) for err in ERRORS if err in whole},
      "milestones":{k:whole.count(k) for k in MARKERS if k in whole},
      "last_progress":[list(map(int,v)) for v in re.findall(r"Qwen (\d+)/(\d+)",whole)[-6:]],
      "project_traceback_lines":[int(n) for n in
        re.findall(r"(?:v2_experiment|cli|v2_selection|ontology_v2)\.py[^\\n]{0,50}?line (\\d+)",whole)],
      "tail_events":[{"stream":str(x.get("stream_name")),
                      "time":round(float(x.get("time",0)),1),
                      "size":len(str(x.get("data",""))),
                      "has_error":any(e in str(x.get("data","")) for e in ERRORS),
                      "has_traceback":"Traceback" in str(x.get("data","")),
                      "has_progress":"Qwen " in str(x.get("data",""))}
                    for x in logs[-8:] if isinstance(x,dict)]
    },sort_keys=True))
    # No raw Kaggle exception, image, text or prompt is ever printed.

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("DIAGNOSTIC_BLOCKED",type(e).__name__)
        sys.exit(1)
