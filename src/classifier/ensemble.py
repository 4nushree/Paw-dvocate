# src/classifier/ensemble.py
#
# Phase 7b — Weighted Ensemble Classifier
#
# Combines all 4 pipeline stages into a final classification:
#   1. keyword_score    (0–1)   → coarse text matching
#   2. embedding_sim    (0–1)   → semantic similarity
#   3. groq_confidence  (0–1)   → LLM reasoning
#   4. openpaws_score   (-1–+1) → alignment framing
#
# Outputs: final_label, final_confidence, relevance_score, risk_level

from src.utils.db import get_connection, utcnow
from config.settings import ENSEMBLE_WEIGHTS


# ─────────────────────────────────────────────────────
# LABEL DIRECTION MAPPING
# ─────────────────────────────────────────────────────
# We convert every stage's output into a signed direction
# so the ensemble can combine them mathematically.
#
#   +1 = pro_animal
#   -1 = anti_animal
#    0 = neutral
# ─────────────────────────────────────────────────────

def _label_to_direction(label: str) -> int:
    """Maps a label string to a signed direction."""
    label = (label or "").lower().strip()
    if "pro" in label:
        return +1
    if "anti" in label:
        return -1
    return 0


def _direction_to_label(direction: float) -> str:
    """Maps a signed score back to a label."""
    if direction > 0.05:
        return "pro_animal"
    if direction < -0.05:
        return "anti_animal"
    return "neutral"


# ─────────────────────────────────────────────────────
# COMPUTE FINAL CLASSIFICATION FOR ONE BILL
# ─────────────────────────────────────────────────────

def compute_ensemble(row: dict) -> dict:
    """
    Takes a classification row (from the DB) and computes the
    final ensemble result.

    Parameters:
        row (dict): A classifications table row with keys:
            keyword_score, keyword_match, keywords_found,
            embedding_similarity, embedding_label,
            groq_label, groq_confidence,
            openpaws_alignment_score

    Returns:
        dict:
          final_label       (str):   "pro_animal", "anti_animal", "neutral"
          final_confidence  (float): 0.0–1.0
          relevance_score   (float): 0.0–1.0 (how relevant is this bill to animals)
          risk_level        (str):   "high", "medium", "low"
    """
    w = ENSEMBLE_WEIGHTS

    # ── Extract raw signals ──
    kw_score   = float(row.get("keyword_score", 0.0) or 0.0)
    kw_match   = int(row.get("keyword_match", 0) or 0)
    emb_sim    = float(row.get("embedding_similarity", 0.0) or 0.0)
    emb_label  = str(row.get("embedding_label", "") or "")
    groq_label = str(row.get("groq_label", "") or "")
    groq_conf  = float(row.get("groq_confidence", 0.0) or 0.0)
    op_score   = float(row.get("openpaws_alignment_score", 0.0) or 0.0)

    # ── Determine keyword direction from keywords_found ──
    kw_found = str(row.get("keywords_found", "") or "")
    kw_pro = kw_found.count("|pro|")
    kw_anti = kw_found.count("|anti|")
    if kw_pro > kw_anti:
        kw_direction = +1
    elif kw_anti > kw_pro:
        kw_direction = -1
    else:
        kw_direction = 0

    # ── Convert each stage to a signed direction × magnitude ──
    # keyword:   direction × score
    kw_signal = kw_direction * kw_score

    # embedding: direction × similarity (capped to relevant range)
    emb_direction = _label_to_direction(emb_label)
    emb_signal = emb_direction * emb_sim

    # groq:      direction × confidence
    groq_direction = _label_to_direction(groq_label)
    groq_signal = groq_direction * groq_conf

    # openpaws:  already signed (-1 to +1), use directly
    op_signal = op_score

    # ── Weighted combination ──
    # This gives a signed score: positive = pro, negative = anti
    combined = (
        w["keyword"]   * kw_signal +
        w["embedding"] * emb_signal +
        w["groq"]      * groq_signal +
        w["openpaws"]  * op_signal
    )

    # ── Final label ──
    final_label = _direction_to_label(combined)

    # ── Final confidence ──
    # Weighted average of each stage's confidence magnitude
    confidence = (
        w["keyword"]   * kw_score +
        w["embedding"] * min(emb_sim, 1.0) +
        w["groq"]      * groq_conf +
        w["openpaws"]  * abs(op_score)
    )
    confidence = round(min(confidence, 1.0), 4)

    # ── Relevance score ──
    # How relevant is this bill to animal welfare AT ALL?
    # High relevance = any stage thinks it's about animals
    relevance = max(
        kw_score,
        emb_sim if emb_label != "neutral" else emb_sim * 0.3,
        groq_conf if groq_label != "neutral" else 0.0,
        abs(op_score),
    )
    relevance = round(min(relevance, 1.0), 4)

    # ── Risk level ──
    # "high" = high-confidence non-neutral bill (needs advocacy attention)
    if final_label != "neutral" and confidence >= 0.6:
        risk_level = "high"
    elif final_label != "neutral" and confidence >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "final_label":      final_label,
        "final_confidence": confidence,
        "relevance_score":  relevance,
        "risk_level":       risk_level,
    }


# ─────────────────────────────────────────────────────
# BATCH: RUN ENSEMBLE ON ALL CLASSIFIED BILLS
# ─────────────────────────────────────────────────────

def run_ensemble(verbose: bool = True) -> dict:
    """
    Loads all classification rows from SQLite, computes ensemble
    scores, and writes final_label / final_confidence / relevance_score /
    risk_level back to the classifications table.

    Returns summary stats.
    """
    conn = get_connection()
    rows = conn.execute("SELECT * FROM classifications").fetchall()
    conn.close()

    total = len(rows)
    if verbose:
        print(f"\n  ⚖️  Running weighted ensemble on {total:,} classified bills...")
        print(f"     Weights: keyword={ENSEMBLE_WEIGHTS['keyword']:.2f}  "
              f"embedding={ENSEMBLE_WEIGHTS['embedding']:.2f}  "
              f"groq={ENSEMBLE_WEIGHTS['groq']:.2f}  "
              f"openpaws={ENSEMBLE_WEIGHTS['openpaws']:.2f}\n")

    counts = {"pro_animal": 0, "anti_animal": 0, "neutral": 0}
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    updated = 0

    for i, row in enumerate(rows):
        row_dict = dict(row)
        bill_id = row_dict["bill_id"]

        # Compute ensemble
        result = compute_ensemble(row_dict)

        # Write back to DB
        conn = get_connection()
        conn.execute("""
            UPDATE classifications
            SET final_label      = :final_label,
                final_confidence = :final_confidence,
                relevance_score  = :relevance_score,
                risk_level       = :risk_level,
                updated_at       = :updated_at
            WHERE bill_id = :bill_id
        """, {
            "bill_id":          bill_id,
            "final_label":      result["final_label"],
            "final_confidence": result["final_confidence"],
            "relevance_score":  result["relevance_score"],
            "risk_level":       result["risk_level"],
            "updated_at":       utcnow(),
        })
        conn.commit()
        conn.close()

        counts[result["final_label"]] += 1
        risk_counts[result["risk_level"]] += 1
        updated += 1

    if verbose:
        print(f"  ✅ Ensemble complete. {updated:,} bills scored.\n")
        print(f"  Final label distribution:")
        for lbl, n in counts.items():
            pct = n / total * 100 if total else 0
            print(f"    {lbl:12s}: {n:,} ({pct:.1f}%)")
        print(f"\n  Risk levels:")
        for lvl, n in risk_counts.items():
            pct = n / total * 100 if total else 0
            print(f"    {lvl:8s}: {n:,} ({pct:.1f}%)")
        print()

    return {
        "total":   total,
        "updated": updated,
        "labels":  counts,
        "risks":   risk_counts,
    }


# ─────────────────────────────────────────────────────
# STANDALONE TEST
# python -m src.classifier.ensemble
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("=" * 55)
    print("  PHASE 7b — WEIGHTED ENSEMBLE")
    print("=" * 55)

    # ── Quick unit test with synthetic data ──
    print("\n  ── Synthetic test cases ──\n")

    test_cases = [
        {
            "name": "Strong pro-animal (all signals agree)",
            "row": {
                "keyword_score": 0.8, "keyword_match": 1,
                "keywords_found": "animal cruelty|pro|strong",
                "embedding_similarity": 0.75, "embedding_label": "pro_animal",
                "groq_label": "pro_animal", "groq_confidence": 0.92,
                "openpaws_alignment_score": 0.85,
            },
            "expected_label": "pro_animal",
        },
        {
            "name": "Strong anti-animal (all signals agree)",
            "row": {
                "keyword_score": 0.6, "keyword_match": 1,
                "keywords_found": "ag-gag|anti|strong, hunting|anti|weak",
                "embedding_similarity": 0.65, "embedding_label": "anti_animal",
                "groq_label": "anti_animal", "groq_confidence": 0.88,
                "openpaws_alignment_score": -0.75,
            },
            "expected_label": "anti_animal",
        },
        {
            "name": "Neutral (no animal relevance)",
            "row": {
                "keyword_score": 0.0, "keyword_match": 0,
                "keywords_found": "",
                "embedding_similarity": 0.15, "embedding_label": "neutral",
                "groq_label": "neutral", "groq_confidence": 0.95,
                "openpaws_alignment_score": 0.0,
            },
            "expected_label": "neutral",
        },
        {
            "name": "Mixed signals (groq pro, openpaws slightly anti)",
            "row": {
                "keyword_score": 0.3, "keyword_match": 1,
                "keywords_found": "wildlife|pro|weak",
                "embedding_similarity": 0.45, "embedding_label": "pro_animal",
                "groq_label": "pro_animal", "groq_confidence": 0.65,
                "openpaws_alignment_score": -0.1,
            },
            "expected_label": "pro_animal",
        },
    ]

    for tc in test_cases:
        result = compute_ensemble(tc["row"])
        match = "✅" if result["final_label"] == tc["expected_label"] else "❌"
        print(f"  {match} {tc['name']}")
        print(f"     label={result['final_label']:12s}  "
              f"conf={result['final_confidence']:.3f}  "
              f"rel={result['relevance_score']:.3f}  "
              f"risk={result['risk_level']}")
        print()

    # ── Run on real DB ──
    print("\n  ── Running on SQLite database ──")
    summary = run_ensemble(verbose=True)
