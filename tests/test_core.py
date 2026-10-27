from vaxbench.core import (
    text_digest, rank_digest, tweet_model_text,
    parse_json_object, validate_llm_label, pick_label,
    ZERO_SHOT_CANDIDATES,
)
import pytest

def test_deterministic_keys():
    assert text_digest("hello") == text_digest("hello")
    assert rank_digest("a", "42") != rank_digest("a", "43")

def test_preprocess_preserves_negation():
    t = tweet_model_text("Not against vaccination @someone https://t.co/foo")
    assert t.startswith("Not against vaccination")
    assert t.endswith("@user http")

def test_parse_json_code_fence():
    fence = chr(96) * 3
    assert parse_json_object(fence + 'json\n{"stance":"support"}\n' + fence)["stance"] == "support"

def test_reject_invalid_json():
    with pytest.raises(ValueError):
        parse_json_object("No JSON returned")

def test_validate_exact_evidence():
    base = {
        "relevance": "human", "stance": "no_expressed_stance",
        "interpretation_basis": "not_expressed",
        "evidence": "", "vaccine_target": "influenza",
        "safety_concern": False, "adverse_event_report": False,
        "procedure_fear": False, "access_barrier": False,
        "mandate_stance": "no_expressed_stance",
    }
    assert validate_llm_label(base, "Had my flu vaccine today")
    with pytest.raises(ValueError):
        validate_llm_label({**base, "evidence":"words not present"}, "Had my flu vaccine today")

def test_abstention_on_near_tie():
    scores = dict.fromkeys(ZERO_SHOT_CANDIDATES, 0.0)
    scores["support"] = .38
    scores["oppose"] = .35
    out = pick_label(scores)
    assert out["abstained"]
    assert out["label"] == "insufficient_context"

def test_high_margin_not_abstain():
    scores = dict.fromkeys(ZERO_SHOT_CANDIDATES, 0.01)
    scores["support"] = .90
    assert pick_label(scores)["label"] == "support"
