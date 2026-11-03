"""Deterministic sampling and result validation; no external inference dependency."""
import hashlib
import json
import re
from typing import Any

STANCE_LABELS = (
    "support", "oppose", "uncertain", "mixed",
    "no_expressed_stance", "insufficient_context",
)
RELEVANCE_LABELS = ("human", "animal", "unrelated", "unclear")
ZERO_SHOT_CANDIDATES = {
    "support": "the author supports vaccination",
    "oppose": "the author opposes vaccination",
    "uncertain": "the author is uncertain about getting vaccinated",
    "mixed": "the author expresses both support and opposition to vaccination",
    "no_expressed_stance": "the author does not express a position on vaccination",
    "insufficient_context": "the author's position on vaccination cannot be determined",
}

def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def rank_digest(record_key: str, seed: str) -> str:
    return hashlib.sha256((seed + "\0" + record_key).encode("utf-8")).hexdigest()

def tweet_model_text(text: str) -> str:
    """CardiffNLP-style @user/http token substitution, preserving negation."""
    return " ".join(
        "@user" if token.startswith("@") else
        "http" if token.lower().startswith(("http://", "https://")) else token
        for token in str(text).split()
    )

def parse_json_object(generation: str) -> dict[str, Any]:
    """Parse a single assistant JSON object, tolerating a markdown fence."""
    s = generation.strip()
    fence = chr(96) * 3
    if s.startswith(fence):
        s = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*" + re.escape(fence) + r"$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        i = s.find("{")
        if i < 0:
            raise ValueError("model returned no JSON object")
        obj, end = json.JSONDecoder().raw_decode(s[i:])
        if s[i + end:].strip() and not s[i + end:].strip().startswith(fence):
            raise ValueError("extraneous content after JSON object")
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj

def validate_llm_label(obj: dict[str, Any], text: str) -> dict[str, Any]:
    """Schema and evidence checks do not establish semantic accuracy."""
    if obj.get("relevance") not in RELEVANCE_LABELS:
        raise ValueError("invalid relevance")
    if obj.get("stance") not in STANCE_LABELS:
        raise ValueError("invalid stance")
    if obj.get("interpretation_basis") not in (
        "explicit", "inferred", "unresolved", "not_expressed"):
        raise ValueError("invalid interpretation_basis")
    evidence = obj.get("evidence", "")
    if not isinstance(evidence, str) or (evidence and evidence not in text):
        raise ValueError("evidence must be an exact substring of the input")
    for name in ("vaccine_target", "safety_concern", "adverse_event_report",
                 "procedure_fear", "access_barrier", "mandate_stance"):
        if name not in obj:
            raise ValueError("missing " + name)
    if obj["mandate_stance"] not in STANCE_LABELS:
        raise ValueError("invalid mandate stance")
    for name in ("safety_concern", "adverse_event_report",
                 "procedure_fear", "access_barrier"):
        if not isinstance(obj[name], bool):
            raise ValueError("invalid boolean for " + name)
    return obj

def pick_label(scores: dict[str, float], min_top: float = 0.36,
               min_margin: float = 0.12) -> dict[str, Any]:
    """Heuristic abstention thresholds. Never treat scores as calibrated."""
    if set(scores) != set(ZERO_SHOT_CANDIDATES):
        raise ValueError("incomplete candidate score set")
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ordered[:2]
    abstain = top[1] < min_top or (top[1] - second[1]) < min_margin
    return {
        "candidate_label": top[0],
        "label": "insufficient_context" if abstain else top[0],
        "abstained": bool(abstain),
        "top_score": float(top[1]),
        "top_margin": float(top[1] - second[1]),
        "raw_scores": scores,
        "thresholds_heuristic_not_calibrated": True,
    }
