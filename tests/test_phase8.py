# tests/test_phase8.py
"""
Phase 8 tests — Digest Generator + Pipeline Orchestrator
Run:  python -m pytest tests/test_phase8.py -v
"""

import os
import pytest
from datetime import datetime, timezone, timedelta

from src.digest.generator import (
    generate_digest_markdown,
    _risk_emoji,
    _label_emoji,
    _label_display,
    get_digest_bills,
)


# ─────────────────────────────────────────────────────
# TEST 1: Helper functions
# ─────────────────────────────────────────────────────

def test_risk_emoji():
    assert _risk_emoji("high") == "🔴"
    assert _risk_emoji("medium") == "🟡"
    assert _risk_emoji("low") == "🟢"
    assert _risk_emoji("unknown") == "⚪"


def test_label_emoji():
    assert _label_emoji("pro_animal") == "🐾"
    assert _label_emoji("anti_animal") == "⚠️"
    assert _label_emoji("neutral") == "➖"


def test_label_display():
    assert _label_display("pro_animal") == "Pro-Animal"
    assert _label_display("anti_animal") == "Anti-Animal"
    assert _label_display("neutral") == "Neutral"


# ─────────────────────────────────────────────────────
# TEST 2: Markdown generation
# ─────────────────────────────────────────────────────

def _sample_bills():
    return [
        {
            "bill_id": "CA_001", "state": "CA", "bill_number": "AB 100",
            "title": "Animal Cruelty Prevention Act",
            "description": "Increases penalties for animal cruelty",
            "status": "Introduced", "status_date": "2025-01-15",
            "introduced_date": "2025-01-15", "last_action": "Referred to committee",
            "last_action_date": "2025-02-01", "url": "https://example.com/ab100",
            "sponsors": "Jane Smith", "committee": "Judiciary",
            "subjects": "Animals", "keyword_score": 0.8, "keyword_match": 1,
            "keywords_found": "animal cruelty|pro|strong",
            "embedding_similarity": 0.75, "embedding_label": "pro_animal",
            "groq_label": "pro_animal", "groq_confidence": 0.92,
            "groq_reasoning": "Bill directly increases penalties for animal cruelty.",
            "openpaws_alignment_score": 0.85,
            "openpaws_framing_summary": "Frames animals as sentient beings.",
            "final_label": "pro_animal", "final_confidence": 0.85,
            "relevance_score": 0.92, "risk_level": "high",
        },
        {
            "bill_id": "TX_001", "state": "TX", "bill_number": "HB 200",
            "title": "Right to Hunt Constitutional Amendment",
            "description": "Establishes right to hunt",
            "status": "Introduced", "status_date": "2025-03-01",
            "introduced_date": "2025-03-01", "last_action": "Filed",
            "last_action_date": "2025-03-01", "url": "",
            "sponsors": "John Doe", "committee": "Wildlife",
            "subjects": "Hunting", "keyword_score": 0.6, "keyword_match": 1,
            "keywords_found": "right to hunt|anti|strong",
            "embedding_similarity": 0.65, "embedding_label": "anti_animal",
            "groq_label": "anti_animal", "groq_confidence": 0.88,
            "groq_reasoning": "Expands hunting rights.",
            "openpaws_alignment_score": -0.75,
            "openpaws_framing_summary": "Frames wildlife as a resource.",
            "final_label": "anti_animal", "final_confidence": 0.77,
            "relevance_score": 0.88, "risk_level": "high",
        },
        {
            "bill_id": "NY_001", "state": "NY", "bill_number": "S 300",
            "title": "Highway Repair Act",
            "description": "Funds bridge repairs",
            "status": "Introduced", "status_date": "2025-02-10",
            "introduced_date": "2025-02-10", "last_action": "Referred",
            "last_action_date": "2025-02-15", "url": "",
            "sponsors": "", "committee": "Transportation",
            "subjects": "Transportation", "keyword_score": 0.0, "keyword_match": 0,
            "keywords_found": "",
            "embedding_similarity": 0.12, "embedding_label": "neutral",
            "groq_label": "neutral", "groq_confidence": 0.95,
            "groq_reasoning": "Unrelated to animal welfare.",
            "openpaws_alignment_score": 0.0,
            "openpaws_framing_summary": "",
            "final_label": "neutral", "final_confidence": 0.46,
            "relevance_score": 0.0, "risk_level": "low",
        },
    ]


def test_digest_markdown_contains_header():
    md = generate_digest_markdown(_sample_bills())
    assert "Paw-dvocate" in md
    assert "Executive Summary" in md


def test_digest_markdown_contains_stats():
    md = generate_digest_markdown(_sample_bills())
    assert "Pro-animal" in md or "pro_animal" in md.lower()
    assert "Anti-animal" in md or "anti_animal" in md.lower()


def test_digest_markdown_contains_bill_cards():
    md = generate_digest_markdown(_sample_bills())
    assert "AB 100" in md
    assert "HB 200" in md
    assert "Animal Cruelty Prevention" in md


def test_digest_markdown_contains_methodology():
    md = generate_digest_markdown(_sample_bills())
    assert "Methodology" in md
    assert "Keyword Filter" in md
    assert "Semantic Embedding" in md
    assert "LLM Classification" in md


def test_digest_markdown_high_priority_section():
    md = generate_digest_markdown(_sample_bills())
    assert "High Priority" in md


def test_digest_markdown_empty_bills():
    md = generate_digest_markdown([])
    assert "Paw-dvocate" in md
    assert "Total classified bills" in md


def test_digest_markdown_is_valid_markdown():
    md = generate_digest_markdown(_sample_bills())
    # Check basic markdown structure
    assert md.startswith("#")
    assert "---" in md
    assert "|" in md    # tables


# ─────────────────────────────────────────────────────
# TEST 3: Database query
# ─────────────────────────────────────────────────────

def test_get_digest_bills_returns_list():
    result = get_digest_bills(days_back=0)
    assert isinstance(result, list)


def test_get_digest_bills_with_state_filter():
    result = get_digest_bills(days_back=0, states=["CA"])
    assert isinstance(result, list)
    # All returned bills should be CA
    for b in result:
        assert b["state"] == "CA"
