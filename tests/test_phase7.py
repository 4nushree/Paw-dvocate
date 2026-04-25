# tests/test_phase7.py
"""
Phase 7 tests — Open Paws Alignment + Weighted Ensemble
Run:  python -m pytest tests/test_phase7.py -v
Live: pytest tests/test_phase7.py -v -m live
"""

import pytest
from unittest.mock import patch, MagicMock

from src.classifier.openpaws_scorer import (
    score_bill_alignment,
    _call_groq_alignment,
    get_already_scored,
    ALIGNMENT_PROMPT,
)
from src.classifier.ensemble import (
    compute_ensemble,
    _label_to_direction,
    _direction_to_label,
)
from config.settings import ENSEMBLE_WEIGHTS


# ═══════════════════════════════════════════════════════
# OPEN PAWS SCORER TESTS
# ═══════════════════════════════════════════════════════

def test_alignment_prompt_structure():
    assert "alignment_score" in ALIGNMENT_PROMPT
    assert "framing_summary" in ALIGNMENT_PROMPT
    assert "+1.0" in ALIGNMENT_PROMPT
    assert "-1.0" in ALIGNMENT_PROMPT


def test_get_already_scored_returns_set():
    result = get_already_scored()
    assert isinstance(result, set)


def test_score_bill_alignment_hf_failure_falls_back_to_groq():
    """When HF fails, the function should try Groq fallback."""
    bill = {
        "title": "Animal Cruelty Act",
        "description": "Bans animal cruelty",
        "subjects": "Animals",
    }
    # Mock HF to fail, Groq to succeed
    hf_fail = {"success": False, "error": "HF unavailable"}
    groq_ok = {
        "success": True,
        "data": {"alignment_score": 0.8, "framing_summary": "Pro-animal framing"},
    }
    with patch("src.classifier.openpaws_scorer._call_hf_api", return_value=hf_fail):
        with patch("src.classifier.openpaws_scorer._call_groq_alignment", return_value=groq_ok):
            result = score_bill_alignment(bill)

    assert result["success"] is True
    assert result["backend"] == "groq"
    assert result["openpaws_alignment_score"] == 0.8


def test_score_bill_alignment_both_fail():
    bill = {"title": "Test", "description": "", "subjects": ""}
    hf_fail = {"success": False, "error": "HF down"}
    groq_fail = {"success": False, "error": "Groq down"}
    with patch("src.classifier.openpaws_scorer._call_hf_api", return_value=hf_fail):
        with patch("src.classifier.openpaws_scorer._call_groq_alignment", return_value=groq_fail):
            result = score_bill_alignment(bill)

    assert result["success"] is False
    assert result["openpaws_alignment_score"] == 0.0


def test_score_clamp_range():
    """Alignment score must be clamped to [-1, +1]."""
    bill = {"title": "Test", "description": "", "subjects": ""}
    extreme = {
        "success": True,
        "data": {"alignment_score": 5.0, "framing_summary": "test"},
    }
    with patch("src.classifier.openpaws_scorer._call_hf_api", return_value=extreme):
        result = score_bill_alignment(bill)
    assert result["openpaws_alignment_score"] == 1.0


# ═══════════════════════════════════════════════════════
# ENSEMBLE TESTS
# ═══════════════════════════════════════════════════════

def test_label_to_direction():
    assert _label_to_direction("pro_animal") == +1
    assert _label_to_direction("anti_animal") == -1
    assert _label_to_direction("neutral") == 0
    assert _label_to_direction("") == 0
    assert _label_to_direction(None) == 0


def test_direction_to_label():
    assert _direction_to_label(0.5) == "pro_animal"
    assert _direction_to_label(-0.5) == "anti_animal"
    assert _direction_to_label(0.0) == "neutral"
    assert _direction_to_label(0.01) == "neutral"   # below threshold


def test_weights_sum_to_one():
    total = sum(ENSEMBLE_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, should be 1.0"


def test_ensemble_strong_pro():
    row = {
        "keyword_score": 0.8, "keyword_match": 1,
        "keywords_found": "animal cruelty|pro|strong",
        "embedding_similarity": 0.75, "embedding_label": "pro_animal",
        "groq_label": "pro_animal", "groq_confidence": 0.92,
        "openpaws_alignment_score": 0.85,
    }
    result = compute_ensemble(row)
    assert result["final_label"] == "pro_animal"
    assert result["final_confidence"] > 0.5
    assert result["risk_level"] == "high"
    assert result["relevance_score"] > 0.5


def test_ensemble_strong_anti():
    row = {
        "keyword_score": 0.6, "keyword_match": 1,
        "keywords_found": "ag-gag|anti|strong",
        "embedding_similarity": 0.65, "embedding_label": "anti_animal",
        "groq_label": "anti_animal", "groq_confidence": 0.88,
        "openpaws_alignment_score": -0.75,
    }
    result = compute_ensemble(row)
    assert result["final_label"] == "anti_animal"
    assert result["final_confidence"] > 0.5
    assert result["risk_level"] == "high"


def test_ensemble_neutral():
    row = {
        "keyword_score": 0.0, "keyword_match": 0, "keywords_found": "",
        "embedding_similarity": 0.15, "embedding_label": "neutral",
        "groq_label": "neutral", "groq_confidence": 0.95,
        "openpaws_alignment_score": 0.0,
    }
    result = compute_ensemble(row)
    assert result["final_label"] == "neutral"
    assert result["risk_level"] == "low"


def test_ensemble_empty_row():
    """Should not crash on empty / default row."""
    row = {
        "keyword_score": 0.0, "keyword_match": 0, "keywords_found": "",
        "embedding_similarity": 0.0, "embedding_label": "",
        "groq_label": "", "groq_confidence": 0.0,
        "openpaws_alignment_score": 0.0,
    }
    result = compute_ensemble(row)
    assert result["final_label"] == "neutral"
    assert result["final_confidence"] == 0.0
    assert result["risk_level"] == "low"


def test_ensemble_return_format():
    row = {
        "keyword_score": 0.5, "keyword_match": 1,
        "keywords_found": "animal|pro|weak",
        "embedding_similarity": 0.4, "embedding_label": "pro_animal",
        "groq_label": "pro_animal", "groq_confidence": 0.7,
        "openpaws_alignment_score": 0.3,
    }
    result = compute_ensemble(row)

    assert "final_label" in result
    assert "final_confidence" in result
    assert "relevance_score" in result
    assert "risk_level" in result

    assert result["final_label"] in ("pro_animal", "anti_animal", "neutral")
    assert 0.0 <= result["final_confidence"] <= 1.0
    assert 0.0 <= result["relevance_score"] <= 1.0
    assert result["risk_level"] in ("high", "medium", "low")


def test_ensemble_mixed_signals_groq_wins():
    """Groq has 45% weight — should dominate when others are weak."""
    row = {
        "keyword_score": 0.1, "keyword_match": 1,
        "keywords_found": "animal|pro|weak",
        "embedding_similarity": 0.3, "embedding_label": "neutral",
        "groq_label": "anti_animal", "groq_confidence": 0.85,
        "openpaws_alignment_score": -0.4,
    }
    result = compute_ensemble(row)
    assert result["final_label"] == "anti_animal"


# ═══════════════════════════════════════════════════════
# LIVE TESTS (hit real APIs)
# ═══════════════════════════════════════════════════════

@pytest.mark.live
def test_live_alignment_pro():
    bill = {
        "title": "Animal Cruelty Prevention and Puppy Mill Ban Act",
        "description": "Bans puppy mills and increases cruelty penalties to felony",
        "subjects": "Animals",
    }
    result = score_bill_alignment(bill)
    assert result["success"] is True
    assert result["openpaws_alignment_score"] > 0.0


@pytest.mark.live
def test_live_alignment_anti():
    bill = {
        "title": "Agricultural Operations Protection Act",
        "description": "Criminalizes unauthorized recording at agricultural facilities",
        "subjects": "Agriculture",
    }
    result = score_bill_alignment(bill)
    assert result["success"] is True
    assert result["openpaws_alignment_score"] < 0.0
