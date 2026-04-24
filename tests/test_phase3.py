# tests/test_phase3.py

"""
Phase 3 tests — keyword filter.
Run with:  python -m pytest tests/test_phase3.py -v
"""

from src.classifier.keyword_filter import (
    classify_bill_keywords,
    _build_search_text,
    _find_matches,
    _score_matches,
)


# ─────────────────────────────────────────────────────
# TEST 1: Search text builder
# ─────────────────────────────────────────────────────

def test_build_search_text_combines_fields():
    bill = {
        "title": "Animal Cruelty Prevention Act",
        "description": "Increases penalties for animal abuse",
        "subjects": "Animals, Criminal Law",
    }
    text = _build_search_text(bill)
    assert "animal cruelty prevention act" in text
    assert "increases penalties for animal abuse" in text
    assert "animals, criminal law" in text


def test_build_search_text_handles_missing_fields():
    bill = {"title": "Some Bill"}
    text = _build_search_text(bill)
    assert "some bill" in text
    assert text == "some bill"


def test_build_search_text_empty_bill():
    text = _build_search_text({})
    assert text == ""


# ─────────────────────────────────────────────────────
# TEST 2: Keyword matching
# ─────────────────────────────────────────────────────

def test_find_matches_exact():
    text = "this bill addresses animal cruelty in the state"
    matches = _find_matches(text, ["animal cruelty", "puppy mill"])
    assert "animal cruelty" in matches
    assert "puppy mill" not in matches


def test_find_matches_word_boundary():
    """'animal' should NOT match inside 'animals' unless 'animals' is in the list."""
    text = "protects animals from harm"
    matches_singular = _find_matches(text, ["animal"])
    matches_plural   = _find_matches(text, ["animals"])
    assert "animal" not in matches_singular   # word boundary prevents partial match
    assert "animals" in matches_plural


def test_find_matches_no_match():
    text = "a bill about tax reform"
    matches = _find_matches(text, ["animal cruelty", "wildlife"])
    assert matches == []


def test_find_matches_special_chars():
    """ag-gag has a hyphen — should still match."""
    text = "this is an ag-gag bill"
    matches = _find_matches(text, ["ag-gag"])
    assert "ag-gag" in matches


# ─────────────────────────────────────────────────────
# TEST 3: Score calculation
# ─────────────────────────────────────────────────────

def test_score_zero_when_no_matches():
    score = _score_matches([], [], [], [], [], [])
    assert score == 0.0


def test_score_one_strong():
    # 1 strong = 1.0 / 3.0 = 0.3333
    score = _score_matches(["animal cruelty"], [], [], [], [], [])
    assert 0.33 <= score <= 0.34


def test_score_caps_at_one():
    # 5 strong = 5.0 / 3.0 = capped at 1.0
    score = _score_matches(
        ["a", "b", "c"], [], [],
        ["d", "e"], [], [],
    )
    assert score == 1.0


def test_score_moderate_and_weak():
    # 1 moderate (0.6) + 1 weak (0.3) = 0.9 / 3.0 = 0.30
    score = _score_matches([], ["companion animal"], ["wildlife"], [], [], [])
    assert 0.29 <= score <= 0.31


# ─────────────────────────────────────────────────────
# TEST 4: Full classify_bill_keywords function
# ─────────────────────────────────────────────────────

def test_classify_pro_animal_bill():
    bill = {
        "title": "Animal Cruelty Prevention and Puppy Mill Regulation Act",
        "description": "Bans puppy mills and increases animal cruelty penalties",
        "subjects": "Animals, Animal Welfare",
    }
    result = classify_bill_keywords(bill)

    assert result["keyword_match"] is True
    assert result["keyword_score"] > 0
    assert len(result["keywords_found"]) > 0

    # Should find strong pro keywords
    kw_strings = result["keywords_found"]
    directions = [kw.split("|")[1] for kw in kw_strings]
    assert "pro" in directions


def test_classify_anti_animal_bill():
    bill = {
        "title": "Right to Hunt Constitutional Amendment",
        "description": "Establishes constitutional right to hunt and trap wildlife",
        "subjects": "Hunting, Wildlife Management",
    }
    result = classify_bill_keywords(bill)

    assert result["keyword_match"] is True
    assert result["keyword_score"] > 0

    kw_strings = result["keywords_found"]
    directions = [kw.split("|")[1] for kw in kw_strings]
    assert "anti" in directions


def test_classify_irrelevant_bill():
    bill = {
        "title": "Highway Infrastructure Improvement Act",
        "description": "Allocates funding for road repairs in rural counties",
        "subjects": "Transportation, Budget",
    }
    result = classify_bill_keywords(bill)

    assert result["keyword_match"] is False
    assert result["keywords_found"] == []
    assert result["keyword_score"] == 0.0


def test_classify_returns_correct_format():
    bill = {
        "title": "Test Bill",
        "description": "",
        "subjects": "",
    }
    result = classify_bill_keywords(bill)

    # Check all 3 required keys exist
    assert "keyword_match" in result
    assert "keywords_found" in result
    assert "keyword_score" in result

    # Check types
    assert isinstance(result["keyword_match"], bool)
    assert isinstance(result["keywords_found"], list)
    assert isinstance(result["keyword_score"], float)

    # Score is in [0, 1]
    assert 0.0 <= result["keyword_score"] <= 1.0


def test_keywords_found_format():
    """Each entry in keywords_found should be 'keyword|direction|tier'."""
    bill = {
        "title": "Animal welfare improvement act",
        "description": "Improves animal shelter standards",
        "subjects": "",
    }
    result = classify_bill_keywords(bill)

    for kw in result["keywords_found"]:
        parts = kw.split("|")
        assert len(parts) == 3, f"Bad format: {kw}"
        assert parts[1] in ("pro", "anti"), f"Bad direction: {parts[1]}"
        assert parts[2] in ("strong", "moderate", "weak"), f"Bad tier: {parts[2]}"
