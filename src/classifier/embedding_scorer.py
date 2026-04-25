# src/classifier/embedding_scorer.py

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import EMBEDDING_MODEL, EMBEDDINGS_DIR
from config.reference_texts import (
    PRO_ANIMAL_REFERENCES,
    ANTI_ANIMAL_REFERENCES,
    NEUTRAL_REFERENCES,
)


# ─────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────
# The model is loaded ONCE and reused across all calls.
# First load downloads ~80 MB from HuggingFace, then
# it's cached locally in ~/.cache/huggingface/
# ─────────────────────────────────────────────────────

_model = None   # Module-level cache


def load_model():
    """
    Loads the sentence-transformers model into memory.
    Call this once at startup — subsequent calls return
    the cached model instantly.

    Model: all-MiniLM-L6-v2
      - 384-dimensional vectors
      - Fast (~14K sentences/sec on GPU, ~500/sec on CPU)
      - Good quality for short texts like bill titles
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model: {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  Model loaded. Vector dim: {_model.get_sentence_embedding_dimension()}")
    return _model


# ─────────────────────────────────────────────────────
# REFERENCE EMBEDDINGS (CATEGORY CENTROIDS)
# ─────────────────────────────────────────────────────
# We embed the reference texts from config/reference_texts.py,
# average them per category, and cache to disk as .npy files.
# These centroids are the "targets" we compare each bill against.
# ─────────────────────────────────────────────────────

CENTROID_CACHE = {
    "pro":     os.path.join(EMBEDDINGS_DIR, "centroid_pro.npy"),
    "anti":    os.path.join(EMBEDDINGS_DIR, "centroid_anti.npy"),
    "neutral": os.path.join(EMBEDDINGS_DIR, "centroid_neutral.npy"),
}


def build_reference_embeddings(force_rebuild: bool = False) -> dict:
    """
    Builds or loads cached centroid vectors for each category.

    Returns:
        dict with keys "pro", "anti", "neutral"
        Each value is a numpy array of shape (1, 384)

    The centroids are saved to data/embeddings/ so we
    don't recompute them every time the pipeline runs.
    """
    # Check if all 3 cached files exist
    all_cached = all(os.path.exists(p) for p in CENTROID_CACHE.values())

    if all_cached and not force_rebuild:
        print("  Loading cached reference centroids...")
        return {
            cat: np.load(path).reshape(1, -1)
            for cat, path in CENTROID_CACHE.items()
        }

    # Need to build — load model and embed reference texts
    print("  Building reference centroids from scratch...")
    model = load_model()

    categories = {
        "pro":     PRO_ANIMAL_REFERENCES,
        "anti":    ANTI_ANIMAL_REFERENCES,
        "neutral": NEUTRAL_REFERENCES,
    }

    centroids = {}
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    for cat, texts in categories.items():
        # Embed all reference texts for this category
        vectors = model.encode(texts, show_progress_bar=False)
        # Average to get the centroid
        centroid = np.mean(vectors, axis=0)
        # Save to disk
        np.save(CENTROID_CACHE[cat], centroid)
        # Reshape to (1, dim) for cosine_similarity
        centroids[cat] = centroid.reshape(1, -1)
        print(f"    {cat:8s}: {len(texts)} references → centroid saved")

    return centroids


# ─────────────────────────────────────────────────────
# TEXT PREPARATION
# ─────────────────────────────────────────────────────

def _build_embedding_text(bill: dict) -> str:
    """
    Builds the text string to embed for a single bill.
    Uses title + description (same fields as keyword filter).

    We keep this SHORT because MiniLM works best with
    sentences under ~128 tokens. Long text gets truncated
    by the model anyway, so concatenating huge full_text
    would just waste the first 128 tokens on boilerplate.
    """
    title = bill.get("title", "")
    description = bill.get("description", "")

    if description:
        return f"{title}. {description}"
    return title


# ─────────────────────────────────────────────────────
# SINGLE BILL SCORING
# ─────────────────────────────────────────────────────

def score_bill(bill: dict, model, centroids: dict) -> dict:
    """
    Scores a single bill against all 3 category centroids.

    Parameters:
        bill (dict):       Bill row from database
        model:             Loaded SentenceTransformer model
        centroids (dict):  From build_reference_embeddings()

    Returns:
        dict with:
          - embedding_similarity (float): Highest similarity score (0–1)
          - embedding_label (str):        "pro_animal", "anti_animal", or "neutral"
          - similarities (dict):          Raw scores for all 3 categories
    """
    text = _build_embedding_text(bill)

    if not text.strip():
        return {
            "embedding_similarity": 0.0,
            "embedding_label":      "neutral",
            "similarities": {"pro": 0.0, "anti": 0.0, "neutral": 0.0},
        }

    # Embed the bill text → shape (1, 384)
    bill_vector = model.encode([text], show_progress_bar=False)

    # Compute cosine similarity to each centroid
    sims = {}
    for cat, centroid in centroids.items():
        sim = cosine_similarity(bill_vector, centroid)[0][0]
        sims[cat] = round(float(sim), 4)

    # The category with highest similarity wins
    best_cat = max(sims, key=sims.get)
    best_sim = sims[best_cat]

    # Map category names to classification labels
    label_map = {
        "pro":     "pro_animal",
        "anti":    "anti_animal",
        "neutral": "neutral",
    }

    return {
        "embedding_similarity": round(best_sim, 4),
        "embedding_label":      label_map[best_cat],
        "similarities":         sims,
    }


# ─────────────────────────────────────────────────────
# BATCH PROCESSOR
# ─────────────────────────────────────────────────────

def run_embedding_scorer(
    bills: list[dict],
    batch_size: int = 64,
    verbose: bool = True,
) -> list[dict]:
    """
    Scores a list of bills using embedding similarity.

    Parameters:
        bills (list[dict]):  Bills from db.get_all_bills()
        batch_size (int):    How many bills to embed at once (GPU memory)
        verbose (bool):      Print progress

    Returns:
        list[dict] — Each dict has:
            bill_id, embedding_similarity, embedding_label, similarities
    """
    total = len(bills)
    if verbose:
        print(f"\n  🧠 Running embedding scorer on {total:,} bills...")

    # ── Load model + centroids ──
    model = load_model()
    centroids = build_reference_embeddings()

    if verbose:
        print(f"  Embedding bills in batches of {batch_size}...\n")

    # ── Prepare all texts ──
    texts = [_build_embedding_text(b) for b in bills]

    # ── Batch encode all bills at once (much faster than one-by-one) ──
    all_vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # Pre-normalize for faster cosine
    )

    # ── Stack centroids into a matrix for vectorized comparison ──
    # Shape: (3, 384) — one row per category
    cat_names = ["pro", "anti", "neutral"]
    centroid_matrix = np.vstack([centroids[c] for c in cat_names])

    # ── Compute all similarities at once ──
    # Shape: (num_bills, 3)
    sim_matrix = cosine_similarity(all_vectors, centroid_matrix)

    label_map = {
        "pro":     "pro_animal",
        "anti":    "anti_animal",
        "neutral": "neutral",
    }

    # ── Build results ──
    results = []
    counts = {"pro_animal": 0, "anti_animal": 0, "neutral": 0}

    for i, bill in enumerate(bills):
        sims = {
            cat_names[j]: round(float(sim_matrix[i][j]), 4)
            for j in range(3)
        }
        best_idx = int(np.argmax(sim_matrix[i]))
        best_cat = cat_names[best_idx]
        best_sim = round(float(sim_matrix[i][best_idx]), 4)

        label = label_map[best_cat]
        counts[label] += 1

        results.append({
            "bill_id":              bill["bill_id"],
            "embedding_similarity": best_sim,
            "embedding_label":      label,
            "similarities":         sims,
        })

        # Progress logging
        if verbose and (i % 2000 == 0 or i == 0 or i == total - 1):
            print(
                f"  [{i + 1:>6,}/{total:,}]  "
                f"pro={counts['pro_animal']:,}  "
                f"anti={counts['anti_animal']:,}  "
                f"neutral={counts['neutral']:,}"
            )

    if verbose:
        print(f"\n  ✅ Embedding scoring complete.")
        print(f"  📊 Results:")
        print(f"     Pro-animal:  {counts['pro_animal']:,}  "
              f"({counts['pro_animal'] / total * 100:.1f}%)")
        print(f"     Anti-animal: {counts['anti_animal']:,}  "
              f"({counts['anti_animal'] / total * 100:.1f}%)")
        print(f"     Neutral:     {counts['neutral']:,}  "
              f"({counts['neutral'] / total * 100:.1f}%)")
        print()

    return results


# ─────────────────────────────────────────────────────
# SAVE BILL EMBEDDING TO DISK
# ─────────────────────────────────────────────────────

def save_bill_embedding(bill_id: str, vector: np.ndarray) -> str:
    """
    Saves a single bill's embedding vector to disk as .npy file.
    Returns the file path (relative).

    Used for later retrieval / similarity search across bills.
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    # Sanitize bill_id for filename (replace special chars)
    safe_id = bill_id.replace("/", "_").replace("\\", "_")
    filepath = os.path.join(EMBEDDINGS_DIR, f"{safe_id}.npy")
    np.save(filepath, vector)
    return filepath


# ─────────────────────────────────────────────────────
# STANDALONE TEST
# Run: python -m src.classifier.embedding_scorer
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.db import get_all_bills

    print("=" * 55)
    print("  PHASE 4 — EMBEDDING SCORER TEST")
    print("=" * 55)

    # ── Load bills ──
    bills = get_all_bills()
    print(f"\n  Loaded {len(bills):,} bills from database")

    # ── Run scorer ──
    results = run_embedding_scorer(bills)

    # ── Show top examples per category ──
    for label in ["pro_animal", "anti_animal"]:
        subset = [r for r in results if r["embedding_label"] == label]
        top_5 = sorted(subset, key=lambda r: r["embedding_similarity"], reverse=True)[:5]

        print(f"\n  ── TOP 5 {label.upper()} ──\n")
        for r in top_5:
            bill = next(b for b in bills if b["bill_id"] == r["bill_id"])
            title = bill["title"][:65] + "..." if len(bill["title"]) > 65 else bill["title"]
            print(f"  {r['bill_id']:15s}  sim={r['embedding_similarity']:.4f}  {title}")
            print(f"  {'':15s}  pro={r['similarities']['pro']:.4f}  "
                  f"anti={r['similarities']['anti']:.4f}  "
                  f"neutral={r['similarities']['neutral']:.4f}")
            print()

    # ── Score distribution ──
    all_sims = [r["embedding_similarity"] for r in results]
    print(f"  ── SIMILARITY DISTRIBUTION ──")
    print(f"  Mean:   {np.mean(all_sims):.4f}")
    print(f"  Median: {np.median(all_sims):.4f}")
    print(f"  Min:    {np.min(all_sims):.4f}")
    print(f"  Max:    {np.max(all_sims):.4f}")
    print()
