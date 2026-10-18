"""Submit exactly one private, CPU-only diagnostic kernel, fail closed."""
import json,os,pathlib,re,subprocess,sys,tempfile
KERNEL="keplaxo/vax-stance-audit-20261018-v1"
SOURCES=["keplaxo/vax-sentiment-benchmark-full","keplaxo/vax-qwen-300"]
def run(*parts):
    return subprocess.run(["kaggle",*parts],capture_output=True,text=True,timeout=180)
def main():
    if not (os.getenv("KAGGLE_API_TOKEN") or
            (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise RuntimeError("Missing private Kaggle API credentials")
    state=run("kernels","status",KERNEL)
    if state.returncode==0:
        print("Audit notebook already exists. No resubmission. State:",
              re.findall(r"KernelWorkerStatus\.([A-Z_]+)",state.stdout))
        return
    for ref in SOURCES:
        status=run("kernels","status",ref)
        if status.returncode or "KernelWorkerStatus.COMPLETE" not in status.stdout:
            raise RuntimeError("Verified predecessor has not completed: "+ref)
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp)
        code_file="stance_disagreement_audit.py"
        content=pathlib.Path("kaggle/"+code_file).read_text(encoding="utf-8")
        (root/code_file).write_text(content,encoding="utf-8")
        meta={"id":KERNEL,"title":"Vaccination Stance Disagreement Audit",
              "code_file":code_file,"language":"python","kernel_type":"script",
              "is_private":True,"enable_gpu":False,"enable_internet":False,
              "kernel_sources":SOURCES,"dataset_sources":[],
              "competition_sources":[],"model_sources":[]}
        (root/"kernel-metadata.json").write_text(json.dumps(meta))
        p=run("kernels","push","-p",str(root),"--timeout","1800")
        # Kaggle may return a successful process exit even if its response
        # names an access, quota, or metadata failure. Print only classified
        # operational messages; never raw CLI payload or credentials.
        joined=(p.stdout+" "+p.stderr).lower()
        category=("permission" if "permission" in joined or "denied" in joined else
                  "resource_or_quota" if "quota" in joined or "limit" in joined else
                  "conflict" if "already" in joined or "exist" in joined else
                  "metadata" if "metadata" in joined or "invalid" in joined else
                  "unclassified")
        print("KAGGLE_PUSH_DIAGNOSTIC",{
            "exit_code":p.returncode,"response_chars":len(joined),
            "category":category,
            "mentions_success":"success" in joined,
            "mentions_url":"kaggle.com/code/" in joined,
            "mentions_kernel_slug":KERNEL in joined},flush=True)
        if p.returncode:
            # Only source/metadata submission messages, never private research
            # content. Remove URLs, long opaque tokens, and e-mail addresses.
            message=(p.stderr or p.stdout)[:700]
            message=re.sub(r"https?://\\S+","[url]",message)
            message=re.sub(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9_.-]+","[email]",message)
            message=re.sub(r"[A-Za-z0-9_-]{28,}","[token]",message)
            print("SANITIZED_SUBMISSION_FAILURE",repr(message[:550]),flush=True)
            raise RuntimeError("Private Kaggle CPU audit submission failed: exit "+
                               str(p.returncode)+" category="+category)
    print("CPU_ONLY_AUDIT_SUBMITTED",KERNEL)
    status=run("kernels","status",KERNEL)
    if status.returncode==0:
        print("INITIAL_STATE:",re.findall(r"KernelWorkerStatus\.([A-Z_]+)",status.stdout))

if __name__=="__main__":
    try:main()
    except Exception as e:
        print(type(e).__name__+": "+str(e),file=sys.stderr)
        sys.exit(1)
