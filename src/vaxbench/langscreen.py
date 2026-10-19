"""Audit-friendly language prediction on the prepared sample only.

langid is a screening heuristic; multilingual/very short tweets can be wrong.
"""
import argparse
import re
from pathlib import Path
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+",re.IGNORECASE)

def main(argv=None):
    parser=argparse.ArgumentParser(description="Optional prediction of missing tweet languages")
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    import langid
    df=pd.read_csv(args.input,dtype=str,keep_default_na=False)
    if "text" not in df or "language_screen" not in df:
        raise ValueError("Input must be prepared benchmark sample")
    origins=[]
    labels=[]
    for text, previous in zip(df["text"],df["language_screen"]):
        old=previous.strip().lower()
        if old not in ("", "not_screened", "unknown", "missing"):
            labels.append(previous)
            origins.append("source_language_column_unverified")
            continue
        stripped=URL_RE.sub(" ",text)
        # Do not guess for tweets with too little natural-language text.
        if len(re.findall(r"[^\W\d_]",stripped,flags=re.UNICODE)) < 12:
            labels.append("unknown")
            origins.append("insufficient_text_for_language_screen")
            continue
        lang,_=langid.classify(stripped)
        labels.append(lang)
        origins.append("langid_1.1.6_predicted_not_validated")
    df["language_screen"]=labels
    df["language_screen_origin"]=origins
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest,index=False)
    print("Language-screen records:",len(df))
    print(df.groupby(["language_screen","language_screen_origin"]).size().to_string())
    print("These are predictions, not independently confirmed language labels.")

if __name__ == "__main__":
    main()
