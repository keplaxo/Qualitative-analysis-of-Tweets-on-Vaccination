import pandas as pd
from vaxbench.cli import main

def test_scientific_notation_id_collisions_do_not_merge_distinct_tweets(tmp_path):
    source=tmp_path/"problematic.csv"
    output=tmp_path/"sample.csv"
    pd.DataFrame([
        {"id":"2.44E+17","tweet":"Vaccine access was limited yesterday","date":"2018-08-01"},
        {"id":"2.44E+17","tweet":"Vaccine access improved today","date":"2018-08-02"},
        {"id":"2.44E+17","tweet":"What does vaccination cost?","date":"2018-08-03"},
        {"id":"2.44E+17","tweet":"Vaccines prevent many diseases","date":"2018-08-04"},
    ]).to_csv(source,index=False)
    main(["prepare","--input",str(source),"--text-col","tweet",
          "--id-col","id","--date-col","date",
          "--output",str(output),"--main-n","4"])
    data=pd.read_csv(output,dtype=str)
    assert len(data)==4
    assert data["record_key"].nunique()==4
    assert all(data["record_key"]==data["text_hash"])
