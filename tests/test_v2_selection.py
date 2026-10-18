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
