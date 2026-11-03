import json
import pandas as pd
from vaxbench.triage import main
from vaxbench.summarize import main as summary_main

def test_triage_and_summary(tmp_path):
    sample = tmp_path/"sample.csv"
    zpath = tmp_path/"zero.jsonl"
    dest = tmp_path/"triage.csv"
    pd.DataFrame([
        {"record_key":str(i),"text_hash":str(i)+"abc",
         "text":"Vaccination message "+str(i),"year":str(2006+i%3),
         "source":"year_balanced_development"}
        for i in range(30)
    ]).to_csv(sample,index=False)
    with zpath.open("w") as f:
        for i in range(30):
            f.write(json.dumps({"record_key":str(i),"text_hash":str(i)+"abc",
                "label":"support" if i%2 else "insufficient_context",
                "abstained":i%2==0,"top_margin":(i/30.0),
                "revision":"demo"})+"\n")
    main(["--input",str(sample),"--zeroshot",str(zpath),
          "--output",str(dest),"--max-records","12"])
    t=pd.read_csv(dest,dtype=str)
    assert len(t)==12
    assert t["record_key"].nunique()==12
    assert set(t["selection_reason"])=={
        "low_margin_or_abstention","hash_ranked_comparison"}
    a=tmp_path/"sent.jsonl"
    with a.open("w") as f:
        f.write(json.dumps({"record_key":"0","model_label":"negative",
                             "revision":"dummy"})+"\n")
    output=tmp_path/"summary.json"
    summary_main(["--sample",str(sample),"--multilingual",str(a),
                  "--english",str(a),"--zeroshot",str(zpath),
                  "--output",str(output)])
    s=json.loads(output.read_text())
    assert s["accuracy_measured"] is False
    assert s["stage_coverage"]["zero_shot_stance"]==30
    assert s["sentiment_agreement"]["rate"]==1.
