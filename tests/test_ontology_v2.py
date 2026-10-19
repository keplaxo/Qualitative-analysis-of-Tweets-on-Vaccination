"""Synthetic contract tests. No user tweets or human-labeled research data."""
import pytest
from vaxbench.ontology_v2 import validate_llm_label_v2, SYSTEM_PROMPT_V2

TEXT="I am still deciding if I should get the vaccine."

def annotation(stance="uncertain",basis="explicit",evidence="still deciding",
               relevance="human"):
    return {
      "relevance":relevance,"vaccine_target":"generic_vaccination",
      "stance":stance,"interpretation_basis":basis,"evidence":evidence,
      "safety_concern":False,"adverse_event_report":False,
      "procedure_fear":False,"access_barrier":False,
      "mandate_stance":"no_expressed_stance"
    }

def test_v2_explicit_hesitancy_is_valid():
    assert validate_llm_label_v2(annotation(),TEXT)["stance"]=="uncertain"

@pytest.mark.parametrize("basis,evidence",[
    ("unresolved",""),("inferred","still deciding"),("explicit","")
])
def test_v2_uncertainty_cannot_be_generic_abstention(basis,evidence):
    with pytest.raises(ValueError):
        validate_llm_label_v2(annotation(basis=basis,evidence=evidence),TEXT)

def test_v2_missing_context_uses_unresolved_basis():
    a=annotation("insufficient_context","unresolved","")
    assert validate_llm_label_v2(a,TEXT)["stance"]=="insufficient_context"

def test_v2_neutral_on_topic_uses_not_expressed():
    assert validate_llm_label_v2(
      annotation("no_expressed_stance","not_expressed",""),TEXT
    )["stance"]=="no_expressed_stance"

@pytest.mark.parametrize("stance,basis,evidence",[
    ("support","inferred",""),("oppose","unresolved",""),
    ("no_expressed_stance","unresolved",""),
    ("insufficient_context","not_expressed","")
])
def test_v2_rejects_internally_contradictory_labels(stance,basis,evidence):
    with pytest.raises(ValueError):
        validate_llm_label_v2(annotation(stance,basis,evidence),TEXT)

def test_v2_refuses_nonhuman_vaccine_stance():
    with pytest.raises(ValueError):
        validate_llm_label_v2(
          annotation("support","explicit","vaccine",relevance="animal"),TEXT)

def test_v2_exact_text_evidence_guard():
    with pytest.raises(ValueError):
        validate_llm_label_v2(annotation(evidence="unseen sentence"),TEXT)

def test_prompt_distinguishes_ontology_categories():
    for label in ("uncertain","no_expressed_stance","insufficient_context"):
        assert label in SYSTEM_PROMPT_V2
