"""Prespecified, deterministic 60-record controlled ontology comparison.

Requires previously frozen v1 annotations. This selected development subset is
NOT a representative sample and carries no independent reference labels.
"""
from .core import rank_digest

GROUPS=(
    ("abstention_vs_uncertain",20),
    ("support_vs_uncertain",15),
    ("polarity_or_opposition_conflict",10),
    ("exact_agreement_controls",15),
)
SEED="vax-ontology-v2-20261018"

def select_v2_cases(nli_by_key, qwen_by_key, limit=60):
    """Return 60 distinct source keys in interleaved, preregistered strata."""
    if limit!=60: raise ValueError("v2 controlled design fixes sample at 60")
    options={name:[] for name,_ in GROUPS}
    for key,q in qwen_by_key.items():
        d=nli_by_key.get(key)
        if not d or q.get("status")!="validated_schema":
            continue
        ann=q.get("annotation") or {}
        old=ann.get("stance")
        prior=d.get("label")
        if not old or not prior: continue
        if prior=="insufficient_context" and old=="uncertain":
            options["abstention_vs_uncertain"].append(key)
        if prior=="support" and old=="uncertain":
            options["support_vs_uncertain"].append(key)
        if ((prior=="support" and old=="oppose") or
            (prior=="oppose" and old=="support") or
            (prior=="insufficient_context" and old=="oppose")):
            options["polarity_or_opposition_conflict"].append(key)
        if prior==old:
            options["exact_agreement_controls"].append(key)
    strata={}
    for name,quota in GROUPS:
        candidates=sorted(set(options[name]),
                          key=lambda k:(rank_digest(k,SEED+"-"+name),k))
        if len(candidates)<quota:
            raise ValueError("Insufficient frozen records in prespecified stratum: "+name)
        strata[name]=candidates[:quota]
    # Interleave group ordering to let the 12-case preflight include 3 per
    # stratum while the full 60 contain the exact same first 12 records.
    selected=[]
    used=set()
    for ix in range(max(q for _,q in GROUPS)):
        for name,quota in GROUPS:
            if ix>=quota: continue
            key=strata[name][ix]
            if key in used:
                raise ValueError("Overlapping sampling strata")
            selected.append({"record_key":key,"selection_reason":name})
            used.add(key)
    assert len(selected)==60 and len(used)==60
    return selected


def merge_v2_selection(selected, triage):
    """Preserve frozen v2 design strata when joining earlier triage data.

    Both tables have a selection_reason column. Dropping the original triage
    reason prevents pandas from creating selection_reason_x and
    selection_reason_y, which would break downstream aggregate reporting.
    The earlier triage reason is represented by the frozen v1 artifact and
    is not used for v2 stratification.
    """
    import pandas as pd
    design=pd.DataFrame(selected)
    if (design.empty or design["record_key"].duplicated().any()
        or design["selection_reason"].isna().any()):
        raise ValueError("Invalid v2 design case selection")
    if triage["record_key"].duplicated().any():
        raise ValueError("Original triage is not unique by exact text key")
    original=triage.drop(columns=["selection_reason"],errors="ignore")
    joined=design.merge(original,on="record_key",how="left",
                        validate="one_to_one")
    if "text" not in joined or joined["text"].isna().any():
        raise ValueError("v2 case text not joined from original triage")
    if len(joined)!=len(design) or "selection_reason" not in joined:
        raise ValueError("v2 design selection reason lost during merge")
    return joined
