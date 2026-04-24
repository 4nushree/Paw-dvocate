# tests/test_phase5.py

"""
Phase 5 tests — Groq classifier.
Run with:  python -m pytest tests/test_phase5.py -v

Note: Tests that hit the live API are marked with @pytest.mark.live
and skipped by default. Run them with: pytest -m live
"""

import pytest
from src.classifier.groq_classifier import (
    classify_bill_groq,
    _error_result,
    GROQ_API_URL,
    SYSTEM_PROMPT,
)
from config.settings import GROQ_API_KEY


# ─────────────────────────────────────────────────────
# Offline tests (no API calls)
# ─────────────────────────────────────────────────────

def test_error_result_format():
    result = _error_result("test error")
    assert result["success"] is False
    assert result["error"] == "test error"
    assert result["groq_label"] == "neutral"
    assert result["groq_confidence"] == 0.0
    assert result["groq_reasoning"] == ""


def test_system_prompt_exists():
    assert len(SYSTEM_PROMPT) > 100
    assert "pro_animal" in SYSTEM_PROMPT
    assert "anti_animal" in SYSTEM_PROMPT
    assert "neutral" in SYSTEM_PROMPT


def test_api_url():
    assert "groq.com" in GROQ_API_URL
    assert "chat/completions" in GROQ_API_URL


def test_api_key_loaded():
    assert GROQ_API_KEY, "GROQ_API_KEY not set in .env"
    assert GROQ_API_KEY.startswith("gsk_")


# ─────────────────────────────────────────────────────
# Live API tests (require network + valid key)
# ─────────────────────────────────────────────────────

@pytest.mark.live
def test_classify_pro_animal_live():
    bill = {
        "title": "Animal Cruelty Prevention Act",
        "description": "Increases penalties for animal cruelty to felony level",
        "subjects": "Animals",
    }
    result = classify_bill_groq(bill)
    assert result["success"] is True
    assert result["groq_label"] == "pro_animal"
    assert result["groq_confidence"] > 0.5


@pytest.mark.live
def test_classify_neutral_live():
    bill = {
        "title": "Highway Bridge Repair Funding Act",
        "description": "Allocates funds for bridge repair",
        "subjects": "Transportation",
    }
    result = classify_bill_groq(bill)
    assert result["success"] is True
    assert result["groq_label"] == "neutral"


@pytest.mark.live
def test_return_format_live():
    bill = {"title": "Test Bill", "description": "A test"}
    result = classify_bill_groq(bill)

    assert "groq_label" in result
    assert "groq_confidence" in result
    assert "groq_reasoning" in result
    assert "groq_classified_at" in result
    assert "success" in result

    if result["success"]:
        assert result["groq_label"] in ("pro_animal", "anti_animal", "neutral")
        assert 0.0 <= result["groq_confidence"] <= 1.0
        assert len(result["groq_reasoning"]) > 0
        assert len(result["groq_classified_at"]) > 0
