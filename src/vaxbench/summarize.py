"""Report inference coverage and disagreement without calling it accuracy."""
import argparse
import json
from collections import Counter
from pathlib import Path
from .triage import load_predictions

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True)
    p.add_argument("--multilingual", required=True)
    p.add_argument("--english", required=True)
    p.add_argument("--zeroshot", required=True)
    p.add_argument("--qwen", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    import pandas as pd
    sampled = pd.read_csv(args.sample, dtype=str, keep_default_na=False)
    keys = set(sampled["record_key"])
    stage_paths = {
        "multilingual_sentiment": args.multilingual,
        "english_sentiment": args.english,
        "zero_shot_stance": args.zeroshot,
    }
    if args.qwen and Path(args.qwen).exists():
        stage_paths["qwen_semantics"] = args.qwen
    stages = {name:load_predictions(path) for name,path in stage_paths.items()}
    result = {
        "n_sampled":len(keys),
        "method":"exploratory model coverage and prediction agreement",
        "accuracy_measured":False,
        "sampling_not_prevalence_representative":True,
        "stage_coverage":{},
        "stage_labels":{},
        "stage_revisions":{},
    }
    for name, pred in stages.items():
        valid = {k:v for k,v in pred.items() if k in keys}
        result["stage_coverage"][name] = len(valid)
        result["stage_revisions"][name] = sorted(
            set(str(v.get("revision")) for v in valid.values()))
        label_key = ("model_label" if "sentiment" in name else
                     "label" if name == "zero_shot_stance" else None)
        if label_key:
            result["stage_labels"][name] = dict(Counter(
                str(v.get(label_key,"missing")).lower()
                for v in valid.values()))
        else:
            result["stage_labels"][name] = dict(Counter(
                str(v.get("status","missing")) for v in valid.values()))
    if "english_sentiment" in stages and "multilingual_sentiment" in stages:
        left,right=stages["english_sentiment"],stages["multilingual_sentiment"]
        both=keys&set(left)&set(right)
        agrees=sum(str(left[k].get("model_label")).lower() ==
                   str(right[k].get("model_label")).lower() for k in both)
        result["sentiment_agreement"]={
            "paired":len(both),"matching":agrees,
            "rate":(agrees/len(both) if both else None),
            "not_accuracy":True}
    if "qwen_semantics" in stages:
        x,y=stages["zero_shot_stance"],stages["qwen_semantics"]
        both=[k for k in keys&set(x)&set(y)
              if y[k].get("status")=="validated_schema"]
        agree=sum(x[k].get("label")==y[k].get("annotation",{}).get("stance")
                  for k in both)
        result["stance_agreement"]={
            "paired_valid_schema":len(both),"matching":agree,
            "rate":(agree/len(both) if both else None),"not_accuracy":True}
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
