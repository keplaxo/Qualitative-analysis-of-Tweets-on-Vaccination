"""Select a year-covered set of ambiguous and routine records for LLM adjudication.

Selection is exploratory and deliberately enriched, not a validity sample.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from .core import rank_digest

def load_predictions(path):
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("record_key") in out:
                    raise ValueError("duplicate prediction key: " + record["record_key"])
                out[record["record_key"]] = record
    return out

def year_cover(df, n, priority_column):
    if n <= 0 or df.empty:
        return df.head(0).copy()
    years = sorted(df["year"].astype(str).unique())
    quota = n // max(1, len(years))
    selections = []
    for year in years:
        part = df.loc[df["year"].astype(str).eq(str(year))]
        selections.append(part.sort_values([priority_column, "rank"]).head(quota))
    out = pd.concat(selections) if selections else df.head(0).copy()
    remain = df.loc[~df["record_key"].isin(out["record_key"])]
    if len(out) < n:
        out = pd.concat(
            [out, remain.sort_values([priority_column, "rank"]).head(n-len(out))])
    return out.head(n)

def main(argv=None):
    p = argparse.ArgumentParser(description="Qwen review queue from NLI uncertainty")
    p.add_argument("--input", required=True)
    p.add_argument("--zeroshot", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-records", type=int, default=300)
    p.add_argument("--hard-fraction", type=float, default=.6)
    p.add_argument("--seed", default="20261017-qwen")
    args = p.parse_args(argv)
    if not 0 <= args.hard_fraction <= 1 or args.max_records < 1:
        p.error("hard-fraction must be in [0,1], max-records positive")
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    preds = load_predictions(args.zeroshot)
    df = df.loc[df["record_key"].isin(preds)].copy()
    if df.empty:
        raise ValueError("No matching zero-shot predictions")
    df["rank"] = df["text_hash"].map(lambda v: rank_digest(v,args.seed))
    df["margin"] = df["record_key"].map(
        lambda k: float(preds[k]["top_margin"]))
    df["abstained"] = df["record_key"].map(
        lambda k: bool(preds[k]["abstained"]))
    df["hard_score"] = df["margin"] - df["abstained"].astype(int)
    hard_n = min(round(args.max_records * args.hard_fraction), len(df))
    hard = year_cover(df, hard_n, "hard_score").copy()
    hard["selection_reason"] = "low_margin_or_abstention"
    other = df.loc[~df["record_key"].isin(hard["record_key"])].copy()
    other["random_priority"] = other["rank"]
    ordinary = year_cover(other, min(args.max_records-len(hard),len(other)),
                          "random_priority").copy()
    ordinary["selection_reason"] = "hash_ranked_comparison"
    chosen = pd.concat((hard,ordinary)).drop_duplicates("record_key")
    chosen = chosen.drop(columns=["rank","hard_score","random_priority",
                                  "margin","abstained"],errors="ignore")
    target = Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    chosen.to_csv(target,index=False)
    print(json.dumps({
        "queued_for_qwen":int(len(chosen)),
        "hard":int((chosen["selection_reason"]=="low_margin_or_abstention").sum()),
        "comparison":int((chosen["selection_reason"]=="hash_ranked_comparison").sum()),
        "population_prevalence_eligible":False,
        "independent_gold":False
    },indent=2))

if __name__ == "__main__":
    main()
