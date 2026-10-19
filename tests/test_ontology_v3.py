"""Synthetic examples for independent v3 stance and personal intention axes."""
import pytest
from vaxbench.ontology_v3 import validate_llm_label_v3,SYSTEM_PROMPT_V3

def row(stance="no_expressed_stance",basis="not_expressed",evidence="",
        intention="accept",ibasis="explicit",ievidence="I plan to get vaccinated",
        relevance="human"):
    return {
     "relevance":relevance,"vaccine_target":"generic_vaccination",
     "stance":stance,"interpretation_basis":basis,"evidence":evidence,
     "safety_concern":False,"adverse_event_report":False,
     "procedure_fear":False,"access_barrier":False,
     "mandate_stance":"no_expressed_stance",
     "personal_intention":intention,"intention_basis":ibasis,
     "intention_evidence":ievidence,
    }

def test_v3_distinct_intention_vs_ideological_stance():
    text="I plan to get vaccinated but have no opinion about vaccination."
    obj=row()
    assert validate_llm_label_v3(obj,text)["personal_intention"]=="accept"
    assert obj["stance"]=="no_expressed_stance"

def test_v3_can_have_stance_without_personal_intent():
    t="Vaccines save lives, everyone should consider them."
    obj=row(stance="support",basis="explicit",evidence="Vaccines save lives",
            intention="not_expressed",ibasis="not_expressed",ievidence="")
    assert validate_llm_label_v3(obj,t)["stance"]=="support"

def test_v3_past_vaccine_receipt_does_not_force_intent():
    t="I received my vaccine yesterday."
    obj=row(intention="not_expressed",ibasis="not_expressed",ievidence="")
    assert validate_llm_label_v3(obj,t)["personal_intention"]=="not_expressed"

@pytest.mark.parametrize("intention,basis,evidence",[
    ("accept","explicit",""),("reject","unresolved","decline"),
    ("undecided","inferred","thinking"),
    ("not_expressed","explicit",""),
    ("insufficient_context","not_expressed",""),
    ("accept","explicit","not present"),
])
def test_v3_invalid_intent_evidence_and_basis(intention,basis,evidence):
    text="I might get vaccinated, still thinking."
    with pytest.raises(ValueError):
        validate_llm_label_v3(row(intention=intention,ibasis=basis,
                                  ievidence=evidence),text)

def test_v3_rejects_animal_vaccination_personal_intention():
    with pytest.raises(ValueError):
        validate_llm_label_v3(row(relevance="animal"),
                              "I plan to get vaccinated")

def test_v3_prompt_has_two_independent_target_axes():
    for text in ("personal_intention","intention_evidence","stance:",
                 "receipt","mandate_stance","not_expressed"):
        assert text in SYSTEM_PROMPT_V3
