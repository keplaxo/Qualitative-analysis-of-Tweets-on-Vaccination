"""Private, CPU-only aggregate audit of discordant vaccine-stance models.

Loads prior Kaggle notebook outputs; never uploads individual records or tweet
content to GitHub. This script performs NO further model inference.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import pandas as pd

BASE_SLUG="vax-sentiment-benchmark-full"
QWEN_SLUG="vax-qwen-300"
OUT=Path("/kaggle/working/vax_outputs")
OUT.mkdir(parents=True,exist_ok=True)
LABELS=("support","oppose","uncertain","mixed",
        "no_expressed_stance","insufficient_context")
SUBSTANTIVE=set(LABELS[:4])

def only(filename,slug):
    found=[p for p in Path("/kaggle/input").rglob(filename)
           if slug in str(p).lower()]
    if len(found)!=1:
        raise RuntimeError("Upstream file missing or ambiguous: "+slug+"/"+filename)
    return found[0]

def jsonl(path):
    result={}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line)
            key=row.get("record_key")
            if not key or key in result:
                raise RuntimeError("Missing or duplicated private prediction key")
            if row.get("text_hash")!=key:
                raise RuntimeError("Prediction hash/key mismatch")
            result[key]=row
    return result

def add(d,key,value=1):
    d[key]=d.get(key,0)+value

def summarize(rows):
    n=len(rows)
    matches=sum(x["deberta"]==x["qwen"] for x in rows)
    return {"n":n,"matches":matches,"agreement":round(matches/n,6) if n else None,
            "deberta":dict(sorted(Counter(x["deberta"] for x in rows).items())),
            "qwen":dict(sorted(Counter(x["qwen"] for x in rows).items()))}

def main():
    baseline=only("summary.json",BASE_SLUG)
    qsummary=only("qwen_validation.json",QWEN_SLUG)
    full=json.loads(baseline.read_text(encoding="utf-8"))
    qval=json.loads(qsummary.read_text(encoding="utf-8"))
    if (full.get("n_sampled")!=2000 or
        full.get("stage_coverage",{}).get("zero_shot_stance")!=1666 or
        qval.get("processed")!=300 or
        qval.get("schema_valid_count")!=290):
        raise RuntimeError("Upstream run/version differs from verified study")

    sample=pd.read_csv(baseline.parent/"benchmark_sample_screened.csv",
                       dtype=str,keep_default_na=False)
    queue=pd.read_csv(qsummary.parent/"qwen_triage.csv",
                      dtype=str,keep_default_na=False)
    nli=jsonl(baseline.parent/"deberta_stance.jsonl")
    llm=jsonl(qsummary.parent/"qwen_semantics.jsonl")
    assert len(sample)==2000 and sample["record_key"].nunique()==2000
    assert len(queue)==300 and queue["record_key"].nunique()==300
    assert len(nli)==1666 and len(llm)==300

    # Check deterministic one-to-one lineage without publishing any keys or text.
    sample_by_key=sample.set_index("record_key")
    assert set(queue.record_key).issubset(set(sample.record_key))
    assert set(queue.record_key)==set(llm)
    assert set(queue.record_key).issubset(set(nli))
    text_hash_checks=0
    for key in queue.record_key:
        row=sample_by_key.loc[key]
        digest=sha256(str(row["text"]).encode("utf-8")).hexdigest()
        if digest!=str(key) or str(row["text_hash"])!=str(key):
            raise RuntimeError("Private text SHA-256 lineage check failed")
        text_hash_checks+=1
    valid=[]
    invalid=Counter()
    stratum=defaultdict(list)
    years=defaultdict(list)
    confusion={x:{y:0 for y in LABELS} for x in LABELS}
    qrel=Counter()
    basis=Counter()
    nli_flags=Counter()
    qev=Counter()
    qtarget=Counter()
    flag_counts=Counter()
    nli_margin=defaultdict(list)
    discordance=Counter()
    missing=0
    for r in queue.to_dict("records"):
        key=r["record_key"]
        a=llm[key]
        d=nli[key]
        if a.get("status")!="validated_schema":
            err=str(a.get("error","unknown")).split(":",1)[0]
            invalid[err if err.isidentifier() else "unclassified"]+=1
            continue
        an=a.get("annotation",{})
        dl=d.get("label")
        ql=an.get("stance")
        if dl not in LABELS or ql not in LABELS:
            raise RuntimeError("Unexpected model stance label")
        if a.get("text_hash")!=key or d.get("text_hash")!=key:
            raise RuntimeError("Upstream prediction identity mismatch")
        relevance=an.get("relevance","unknown")
        interpretation=an.get("interpretation_basis","unknown")
        abstain=bool(d.get("abstained"))
        margin=float(d.get("top_margin"))
        score=float(d.get("top_score"))
        if not 0 <= margin <= 1 or not 0 <= score <= 1:
            raise RuntimeError("Invalid NLI score or margin")
        evidence=an.get("evidence","")
        if evidence and (not isinstance(evidence,str) or evidence not in str(sample_by_key.loc[key,"text"])):
            raise RuntimeError("Qwen quote-evidence integrity failure")
        obj={"deberta":dl,"qwen":ql,"human":relevance=="human",
             "basis":interpretation,"abstained":abstain,
             "score":score,"margin":margin,
             "target_generic":str(an.get("vaccine_target"))=="generic_vaccination",
             "selection":r["selection_reason"]}
        valid.append(obj)
        stratum[str(r["selection_reason"])].append(obj)
        years[str(r["year"])].append(obj)
        confusion[dl][ql]+=1
        qrel[relevance]+=1
        basis[interpretation]+=1
        nli_flags["abstained" if abstain else "not_abstained"]+=1
        qev["evidence_present" if evidence else "evidence_empty"]+=1
        qtarget[str(an.get("vaccine_target")) in ("unknown","generic_vaccination") and str(an.get("vaccine_target")) or "named_or_other"]+=1
        nli_margin["matching" if dl==ql else "discordant"].append(margin)
        for k in ("safety_concern","adverse_event_report","procedure_fear","access_barrier"):
            if an.get(k) is True: flag_counts[k]+=1
        if dl!=ql:
            discordance[dl+"__TO__"+ql]+=1
    if len(valid)!=290 or sum(sum(x.values()) for x in confusion.values())!=290:
        raise RuntimeError("Paired case coverage invalid")

    report={
      "created_utc":datetime.now(timezone.utc).isoformat(),
      "purpose":"mechanistic investigation of low DeBERTa–Qwen stance agreement",
      "full_benchmark_n":2000,"triaged_n":300,"valid_pair_n":len(valid),
      "invalid_qwen_outputs":dict(sorted(invalid.items())),
      "sha256_text_lineage_pass_count":text_hash_checks,
      "record_key_coverage_passed":True,
      "exact_text_source_identity_verified":True,
      "agreement_all":summarize(valid),
      "confusion_rows_deberta_columns_qwen":confusion,
      "top_disagreement_directions":sorted(discordance.items(),key=lambda v:-v[1])[:15],
      "by_selection_reason":{k:summarize(v) for k,v in sorted(stratum.items())},
      "by_year":{k:summarize(v) for k,v in sorted(years.items())},
      "by_nli_abstention":{
        "abstained":summarize([x for x in valid if x["abstained"]]),
        "not_abstained":summarize([x for x in valid if not x["abstained"]])},
      "by_qwen_relevance":{
        "human":summarize([x for x in valid if x["human"]]),
        "animal_unrelated_unclear":summarize([x for x in valid if not x["human"]])},
      "by_qwen_interpretation":{
        k:summarize([x for x in valid if x["basis"]==k]) for k in sorted(basis)},
      "by_nli_margin":{
        "below_0_12":summarize([x for x in valid if x["margin"]<.12]),
        "0_12_to_0_30":summarize([x for x in valid if .12<=x["margin"]<.30]),
        "at_least_0_30":summarize([x for x in valid if x["margin"]>=.30])},
      "substantive_both":summarize([x for x in valid
                          if x["deberta"] in SUBSTANTIVE and x["qwen"] in SUBSTANTIVE]),
      "nli_support_oppose_vs_qwen_nonexpressed":{
        "nli_support_or_oppose":sum(x["deberta"] in {"support","oppose"} for x in valid),
        "qwen_no_or_insufficient_among_those":sum(
          x["deberta"] in {"support","oppose"} and
          x["qwen"] in {"no_expressed_stance","insufficient_context"} for x in valid)},
      "qwen_relevance":dict(sorted(qrel.items())),
      "qwen_interpretation_basis":dict(sorted(basis.items())),
      "nli_abstention_counts":dict(sorted(nli_flags.items())),
      "qwen_evidence":dict(sorted(qev.items())),
      "qwen_target_grouping":dict(sorted(qtarget.items())),
      "qwen_mention_flags":dict(sorted(flag_counts.items())),
      "mean_nli_top_margin_by_match":{
        k:round(sum(v)/len(v),5) if v else None for k,v in sorted(nli_margin.items())},
      "quality_control":"Model label definitions and schema validation are not independent correctness tests",
      "measured_accuracy":False,
      "population_prevalence_estimated":False,
      "individual_tweet_content_exported":False,
      "clinical_causality_evaluated":False
    }
    dest=OUT/"stance_disagreement_audit.json"
    dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("STANCE_AUDIT",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
