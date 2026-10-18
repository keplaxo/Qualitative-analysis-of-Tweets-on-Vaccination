"""Poll private Kaggle CPU diagnostic and print only aggregate-safe JSON."""
import json,os,pathlib,re,subprocess,sys,tempfile,time
KERNEL="keplaxo/vax-stance-disagreement-audit"
def cli(*args,timeout=150):
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)

def main():
    deadline=time.monotonic()+420
    while True:
        p=cli("kernels","status",KERNEL,timeout=75)
        if p.returncode:
            raise RuntimeError("Audit status request unavailable")
        m=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout)
        state=m[-1] if m else "UNKNOWN"
        if state in ("ERROR","FAILED","CANCELLED","CANCELED","CANCEL_ACKNOWLEDGED"):
            raise RuntimeError("Audit stopped: "+state+"; inspect private Kaggle logs")
        if state=="COMPLETE": break
        if time.monotonic()>deadline:
            print("AUDIT_PENDING",state,"No duplicate submission. Poll later.")
            return
        time.sleep(15)
    with tempfile.TemporaryDirectory() as tmp:
        p=cli("kernels","output",KERNEL,"-p",tmp,
              "--file-pattern","stance_disagreement_audit.json")
        files=list(pathlib.Path(tmp).rglob("stance_disagreement_audit.json"))
        if p.returncode or len(files)!=1:
            raise RuntimeError("Audit completed but aggregate output is not yet available")
        report=json.loads(files[0].read_text(encoding="utf-8"))
        if (report.get("valid_pair_n")!=290 or
            report.get("sha256_text_lineage_pass_count")!=300 or
            report.get("record_key_coverage_passed") is not True or
            report.get("measured_accuracy") is not False or
            report.get("individual_tweet_content_exported") is not False):
            raise RuntimeError("Audit output sanity check failed")
        safe=["created_utc","valid_pair_n","sha256_text_lineage_pass_count",
              "agreement_all","confusion_rows_deberta_columns_qwen",
              "top_disagreement_directions","by_selection_reason",
              "by_nli_abstention","by_qwen_relevance","by_qwen_interpretation",
              "by_nli_margin","substantive_both",
              "nli_support_oppose_vs_qwen_nonexpressed",
              "qwen_relevance","qwen_interpretation_basis",
              "qwen_evidence","invalid_qwen_outputs"]
        out={k:report.get(k) for k in safe}
        print("STANCE_DIAGNOSTIC_COMPLETE")
        print(json.dumps(out,sort_keys=True))
        summary=os.getenv("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary,"a",encoding="utf-8") as f:
                f.write("## Stance disagreement audit complete\n\n")
                f.write("Verified SHA-256 text matches on 300 records.\n")
                v=report["agreement_all"]
                f.write("Paired raw-label agreement: **"+str(v["matches"])+
                        "/"+str(v["n"])+"** (not accuracy).\n")
                f.write("All outputs aggregate only; no tweet text exported.\n")

if __name__=="__main__":
    try:main()
    except Exception as e:
        print(type(e).__name__+": "+str(e),file=sys.stderr)
        sys.exit(1)
