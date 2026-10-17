import json
import pandas as pd
from vaxbench.cli import main
from vaxbench.core import text_digest

def test_prepare_year_coverage_and_dedup(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "sample.csv"
    pd.DataFrame([
        {"tweet_id": "1234567890123456789", "text": "Vaccination appointment today",
         "date": "2006-12-20", "language": "en"},
        {"tweet_id": "1234567890123456780", "text": "Vaccination appointment today",
         "date": "2006-12-20", "language": "en"},
        {"tweet_id": "9876543210987654321", "text": "Vaccines help prevent disease",
         "date": "2007-01-01", "language": "en"},
        {"tweet_id": "3444678999912345611", "text": "A flu jab is available",
         "date": "2007-01-01", "language": "fr"},
        {"tweet_id": "0001", "text": "I got vaccinated this morning",
         "date": "2006-12-22", "language": "en"},
    ]).to_csv(source, index=False)
    params = [
        "prepare", "--input", str(source), "--output", str(output),
        "--id-col", "tweet_id", "--main-n", "4", "--seed", "test-seed",
    ]
    main(params)
    df = pd.read_csv(output, dtype=str)
    assert len(df) == 4
    assert set(df["year"]) == {"2006", "2007"}
    assert df["text"].nunique() == 4
    assert text_digest("Vaccination appointment today") in set(df["record_key"])
    assert set(df["record_key"]) == set(df["text_hash"])
    assert df["record_key"].nunique() == len(df)
    assert json.loads((tmp_path/"sample.csv.manifest.json").read_text())["n_records"] == 4
    before = output.read_bytes()
    main(params)
    assert output.read_bytes() == before

def test_prepare_rejects_unexpected_date_column(tmp_path):
    source = tmp_path/"missing.csv"
    pd.DataFrame({"text":["vaccine"], "not_date":["2019-01-01"]}).to_csv(source,index=False)
    try:
        main(["prepare","--input",str(source),"--output",str(tmp_path/"out.csv")])
    except ValueError as exc:
        assert "Cannot find column" in str(exc)
    else:
        raise AssertionError("Expected date-column error")
