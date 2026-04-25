# tests/test_phase6.py
"""
Phase 6 tests — Groq LLM Classifier
Run:  python -m pytest tests/test_phase6.py -v
Live tests (hit real API): pytest tests/test_phase6.py -v -m live
"""

import pytest
from unittest.mock import patch, MagicMock
from src.classifier.groq_classifier import (
    load_api_key,
    build_prompt,
    classify_bill,
    store_results,
    resume_progress,
    _error_result,
)
from config.settings import GROQ_API_KEY, GROQ_MODEL


# ─────────────────────────────────────────────────────
# TEST 1: API key loading
# ─────────────────────────────────────────────────────

def test_load_api_key_present():
    key = load_api_key()
    assert key.startswith("gsk_")
    assert len(key) > 20


def test_load_api_key_raises_when_missing():
    with patch("src.classifier.groq_classifier.GROQ_API_KEY", ""):
        with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
            load_api_key()


# ─────────────────────────────────────────────────────
# TEST 2: Prompt builder
# ─────────────────────────────────────────────────────

def test_build_prompt_includes_title():
    bill = {"title": "Animal Cruelty Act", "description": "", "subjects": "",
            "committee": "", "state": "CA"}
    prompt = build_prompt(bill)
    assert "Animal Cruelty Act" in prompt
    assert "CA" in prompt


def test_build_prompt_includes_scores():
    bill = {"title": "Test", "description": "", "subjects": "", "committee": "", "state": "NY"}
    prompt = build_prompt(bill, kw_score=0.75, emb_score=0.62)
    assert "0.75" in prompt
    assert "0.62" in prompt


def test_build_prompt_truncates_long_description():
    bill = {
        "title": "Test Bill",
        "description": "X" * 1000,
        "subjects": "",
        "committee": "",
        "state": "TX",
    }
    prompt = build_prompt(bill)
    # Description capped at 500 chars
    assert "X" * 501 not in prompt


def test_build_prompt_all_fields():
    bill = {
        "title": "Wildlife Protection Act",
        "description": "Protects endangered species",
        "subjects": "Wildlife, Environment",
        "committee": "Natural Resources",
        "state": "CA",
    }
    prompt = build_prompt(bill, kw_score=0.9, emb_score=0.8)
    assert "Wildlife Protection Act" in prompt
    assert "Protects endangered species" in prompt
    assert "Wildlife, Environment" in prompt
    assert "Natural Resources" in prompt
    assert "0.90" in prompt


# ─────────────────────────────────────────────────────
# TEST 3: Error result format
# ─────────────────────────────────────────────────────

def test_error_result_structure():
    r = _error_result("some error")
    assert r["success"] is False
    assert r["error"] == "some error"
    assert r["groq_label"] == "neutral"
    assert r["groq_confidence"] == 0.0
    assert r["groq_reasoning"] == ""
    assert r["groq_classified_at"] == ""


# ─────────────────────────────────────────────────────
# TEST 4: classify_bill with mocked HTTP
# ─────────────────────────────────────────────────────

def _mock_groq_response(label="pro_animal", confidence=0.9, reasoning="Test reason"):
    """Creates a mock requests.Response that mimics Groq API output."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": f'{{"label": "{label}", "confidence": {confidence}, "reasoning": "{reasoning}"}}'
            }
        }]
    }
    return mock_resp


def test_classify_bill_pro_animal_mock():
    bill = {"title": "Animal Cruelty Prevention Act", "description": "Bans puppy mills",
            "subjects": "Animals", "committee": "", "state": "CA"}

    with patch("requests.post", return_value=_mock_groq_response("pro_animal", 0.92)):
        result = classify_bill(bill, kw_score=0.8, emb_score=0.7, api_key="gsk_fake")

    assert result["success"] is True
    assert result["groq_label"] == "pro_animal"
    assert result["groq_confidence"] == 0.92
    assert "Test reason" in result["groq_reasoning"]
    assert result["groq_classified_at"] != ""


def test_classify_bill_anti_animal_mock():
    bill = {"title": "Right to Hunt Act", "description": "Expands hunting rights",
            "subjects": "Hunting", "committee": "", "state": "TX"}

    with patch("requests.post", return_value=_mock_groq_response("anti_animal", 0.85)):
        result = classify_bill(bill, api_key="gsk_fake")

    assert result["groq_label"] == "anti_animal"
    assert result["success"] is True


def test_classify_bill_invalid_label_defaults_neutral():
    """If the LLM returns a garbage label, it must be caught and defaulted."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"label": "GARBAGE", "confidence": 0.5, "reasoning": "x"}'}}]
    }
    bill = {"title": "Test", "description": "", "subjects": "", "committee": "", "state": "CA"}
    with patch("requests.post", return_value=mock_resp):
        result = classify_bill(bill, api_key="gsk_fake")
    assert result["groq_label"] == "neutral"
    assert result["success"] is True


def test_classify_bill_confidence_clamped():
    """Confidence > 1.0 from LLM must be clamped to 1.0."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"label": "neutral", "confidence": 1.5, "reasoning": "x"}'}}]
    }
    bill = {"title": "Test", "description": "", "subjects": "", "committee": "", "state": "CA"}
    with patch("requests.post", return_value=mock_resp):
        result = classify_bill(bill, api_key="gsk_fake")
    assert result["groq_confidence"] == 1.0


def test_classify_bill_rate_limit_returns_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    bill = {"title": "Test", "description": "", "subjects": "", "committee": "", "state": "CA"}
    with patch("requests.post", return_value=mock_resp):
        result = classify_bill(bill, api_key="gsk_fake")
    assert result["success"] is False
    assert "rate_limited_429" in result["error"]


def test_classify_bill_network_error():
    import requests as req_lib
    bill = {"title": "Test", "description": "", "subjects": "", "committee": "", "state": "CA"}
    with patch("requests.post", side_effect=req_lib.exceptions.ConnectionError("timeout")):
        result = classify_bill(bill, api_key="gsk_fake")
    assert result["success"] is False
    assert "Network error" in result["error"]


# ─────────────────────────────────────────────────────
# TEST 5: resume_progress
# ─────────────────────────────────────────────────────

def test_resume_progress_returns_set():
    done = resume_progress()
    assert isinstance(done, set)


# ─────────────────────────────────────────────────────
# LIVE API tests (touch real Groq API)
# Run with: pytest tests/test_phase6.py -v -m live
# ─────────────────────────────────────────────────────

@pytest.mark.live
def test_live_pro_animal():
    bill = {
        "title": "Animal Cruelty Prevention and Puppy Mill Ban Act",
        "description": "Bans commercial puppy mills, increases cruelty penalties to felony",
        "subjects": "Animals, Criminal Law",
        "committee": "Judiciary",
        "state": "CA",
    }
    result = classify_bill(bill, kw_score=0.9, emb_score=0.78, api_key=load_api_key())
    assert result["success"] is True
    assert result["groq_label"] == "pro_animal"
    assert result["groq_confidence"] >= 0.7


@pytest.mark.live
def test_live_anti_animal():
    bill = {
        "title": "Agricultural Operations Protection Act",
        "description": "Criminalizes unauthorized recording at farms and slaughterhouses",
        "subjects": "Agriculture, Criminal Law",
        "committee": "Agriculture",
        "state": "TX",
    }
    result = classify_bill(bill, kw_score=0.8, emb_score=0.71, api_key=load_api_key())
    assert result["success"] is True
    assert result["groq_label"] == "anti_animal"


@pytest.mark.live
def test_live_neutral():
    bill = {
        "title": "State Highway Infrastructure Improvement Act",
        "description": "Allocates $500M for interstate bridge repair and maintenance",
        "subjects": "Transportation, Budget",
        "committee": "Transportation",
        "state": "NY",
    }
    result = classify_bill(bill, kw_score=0.0, emb_score=0.12, api_key=load_api_key())
    assert result["success"] is True
    assert result["groq_label"] == "neutral"
