"""Contract tests for v2 selection using synthetic, non-research keys."""
from vaxbench.v2_selection import select_v2_cases, GROUPS

def make_replay():
    nli,qwen={},{}
    groups=(
        ("insufficient_context","uncertain",140),
        ("support","uncertain",40),
        ("support","oppose",12),
        ("oppose","oppose",20),
    )
    for i,(old,new,n) in enumerate(groups):
        for j in range(n):
            k=f"synthetic-{i}-{j:03}"
            nli[k]={"label":old}
            qwen[k]={"status":"validated_schema","annotation":{"stance":new}}
    return nli,qwen

def test_balanced_interleaved_reproducible_selection():
    d,q=make_replay()
    got=select_v2_cases(d,q)
    again=select_v2_cases(dict(reversed(list(d.items()))),q)
    assert got==again
    assert len(got)==60
    assert len(set(x["record_key"] for x in got))==60
    counts={name:sum(r["selection_reason"]==name for r in got)
            for name,_ in GROUPS}
    assert counts==dict(GROUPS)
    preview=[r["selection_reason"] for r in got[:12]]
    assert all(preview.count(name)==3 for name,_ in GROUPS)

def test_selection_requires_prespecified_strata():
    d,q=make_replay()
    for k,v in list(q.items()):
        if v["annotation"]["stance"]=="oppose":
            v["status"]="needs_reassessment"
    import pytest
    with pytest.raises(ValueError,match="Insufficient frozen"):
        select_v2_cases(d,q)


def test_v2_selection_merge_preserves_v2_strata_without_suffixes():
    import pandas as pd
    from vaxbench.v2_selection import merge_v2_selection
    selected=[{"record_key":"a","selection_reason":"exact_agreement_controls"},
              {"record_key":"b","selection_reason":"abstention_vs_uncertain"}]
    triage=pd.DataFrame([
        {"record_key":"b","selection_reason":"low_margin_or_abstention","text":"B"},
        {"record_key":"a","selection_reason":"hash_ranked_comparison","text":"A"}
    ])
    joined=merge_v2_selection(selected,triage)
    assert joined["selection_reason"].tolist()==[
        "exact_agreement_controls","abstention_vs_uncertain"]
    assert joined["text"].tolist()==["A","B"]
    assert "selection_reason_x" not in joined
    assert "selection_reason_y" not in joined

def test_v2_merge_rejects_missing_and_duplicate_input_text():
    import pandas as pd
    import pytest
    from vaxbench.v2_selection import merge_v2_selection
    cases=[{"record_key":"a","selection_reason":"exact_agreement_controls"}]
    with pytest.raises(ValueError,match="not joined"):
        merge_v2_selection(cases,pd.DataFrame([
            {"record_key":"b","selection_reason":"old","text":"B"}]))
    with pytest.raises(ValueError,match="not unique"):
        merge_v2_selection(cases,pd.DataFrame([
            {"record_key":"a","selection_reason":"old","text":"A"},
            {"record_key":"a","selection_reason":"old","text":"A"}]))
