# tests/test_phase4.py

"""
Phase 4 tests — embedding scorer.
Run with:  python -m pytest tests/test_phase4.py -v
"""

import numpy as np
from src.classifier.embedding_scorer import (
    load_model,
    build_reference_embeddings,
    score_bill,
    _build_embedding_text,
)


# ─────────────────────────────────────────────────────
# TEST 1: Text builder
# ─────────────────────────────────────────────────────

def test_build_embedding_text_with_description():
    bill = {"title": "Animal Welfare Act", "description": "Protects farm animals"}
    text = _build_embedding_text(bill)
    assert text == "Animal Welfare Act. Protects farm animals"


def test_build_embedding_text_without_description():
    bill = {"title": "Animal Welfare Act", "description": ""}
    text = _build_embedding_text(bill)
    assert text == "Animal Welfare Act"


def test_build_embedding_text_empty():
    text = _build_embedding_text({})
    assert text == ""


# ─────────────────────────────────────────────────────
# TEST 2: Model loading
# ─────────────────────────────────────────────────────

def test_model_loads():
    model = load_model()
    assert model is not None
    assert model.get_embedding_dimension() == 384


def test_model_cached():
    """Second call should return the same object (no reload)."""
    model1 = load_model()
    model2 = load_model()
    assert model1 is model2


# ─────────────────────────────────────────────────────
# TEST 3: Reference centroids
# ─────────────────────────────────────────────────────

def test_reference_embeddings_shape():
    centroids = build_reference_embeddings()
    assert "pro" in centroids
    assert "anti" in centroids
    assert "neutral" in centroids

    for cat, vec in centroids.items():
        assert vec.shape == (1, 384), f"{cat} shape is {vec.shape}"


def test_centroids_are_different():
    """Each category centroid should be distinct."""
    centroids = build_reference_embeddings()
    pro  = centroids["pro"].flatten()
    anti = centroids["anti"].flatten()
    neut = centroids["neutral"].flatten()

    # They should NOT be identical
    assert not np.allclose(pro, anti, atol=1e-3)
    assert not np.allclose(pro, neut, atol=1e-3)
    assert not np.allclose(anti, neut, atol=1e-3)


# ─────────────────────────────────────────────────────
# TEST 4: Single bill scoring
# ─────────────────────────────────────────────────────

def test_score_pro_animal_bill():
    bill = {
        "title": "Animal Cruelty Prevention and Humane Treatment Act",
        "description": "Increases felony penalties for animal abuse and bans puppy mills",
    }
    model = load_model()
    centroids = build_reference_embeddings()
    result = score_bill(bill, model, centroids)

    assert result["embedding_label"] == "pro_animal"
    assert result["embedding_similarity"] > 0
    assert result["similarities"]["pro"] > result["similarities"]["neutral"]


def test_score_anti_animal_bill():
    bill = {
        "title": "Right to Hunt and Fish Constitutional Amendment",
        "description": "Establishes a constitutional right to hunt, trap, and fish wildlife",
    }
    model = load_model()
    centroids = build_reference_embeddings()
    result = score_bill(bill, model, centroids)

    assert result["embedding_label"] == "anti_animal"
    assert result["similarities"]["anti"] > result["similarities"]["neutral"]


def test_score_neutral_bill():
    bill = {
        "title": "Highway Infrastructure Improvement and Bridge Repair Act",
        "description": "Allocates state funding for interstate highway maintenance",
    }
    model = load_model()
    centroids = build_reference_embeddings()
    result = score_bill(bill, model, centroids)

    assert result["embedding_label"] == "neutral"
    assert result["similarities"]["neutral"] > result["similarities"]["pro"]
    assert result["similarities"]["neutral"] > result["similarities"]["anti"]


def test_score_empty_bill():
    bill = {"title": "", "description": ""}
    model = load_model()
    centroids = build_reference_embeddings()
    result = score_bill(bill, model, centroids)

    assert result["embedding_label"] == "neutral"
    assert result["embedding_similarity"] == 0.0


# ─────────────────────────────────────────────────────
# TEST 5: Return format
# ─────────────────────────────────────────────────────

def test_return_format():
    bill = {"title": "Test Bill", "description": "Some description"}
    model = load_model()
    centroids = build_reference_embeddings()
    result = score_bill(bill, model, centroids)

    # Required keys
    assert "embedding_similarity" in result
    assert "embedding_label" in result
    assert "similarities" in result

    # Types
    assert isinstance(result["embedding_similarity"], float)
    assert isinstance(result["embedding_label"], str)
    assert isinstance(result["similarities"], dict)

    # Label must be one of the 3
    assert result["embedding_label"] in ("pro_animal", "anti_animal", "neutral")

    # Similarity in valid range
    assert -1.0 <= result["embedding_similarity"] <= 1.0

    # All 3 category scores present
    assert "pro" in result["similarities"]
    assert "anti" in result["similarities"]
    assert "neutral" in result["similarities"]
