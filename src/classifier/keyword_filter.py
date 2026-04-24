# src/classifier/keyword_filter.py

import re
from config.keywords import (
    PRO_ANIMAL_STRONG, PRO_ANIMAL_MODERATE, PRO_ANIMAL_WEAK,
    ANTI_ANIMAL_STRONG, ANTI_ANIMAL_MODERATE, ANTI_ANIMAL_WEAK,
    TIER_WEIGHTS, MAX_RAW_SCORE,
)


# ─────────────────────────────────────────────────────
# TEXT PREPARATION
# ─────────────────────────────────────────────────────

def _build_search_text(bill: dict) -> str:
    """
    Concatenates the bill fields we want to search.
    Lowercases everything so keyword matching is case-insensitive.

    Fields used (in order of importance):
      1. title       — always present, most informative
      2. description — usually present, adds context
      3. subjects    — comma-separated tags from LegiScan

    full_text is NOT used here because:
      - It's empty for most bills (not in the JSON download)
      - Even when present it's huge and slows the filter
      - The embedding stage (Phase 4) will use it instead
    """
    parts = [
        bill.get("title", ""),
        bill.get("description", ""),
        bill.get("subjects", ""),
    ]
    combined = " ".join(parts).lower()

    # Collapse multiple spaces / newlines into single space
    combined = re.sub(r"\s+", " ", combined).strip()

    return combined


# ─────────────────────────────────────────────────────
# KEYWORD MATCHING ENGINE
# ─────────────────────────────────────────────────────

def _find_matches(text: str, keyword_list: list[str]) -> list[str]:
    """
    Checks which keywords from keyword_list appear in text.
    Uses word-boundary matching so 'animal' doesn't match 'animals'
    unless 'animals' is also in the list.

    Returns a list of matched keywords (de-duplicated, order preserved).
    """
    matches = []
    for keyword in keyword_list:
        # \b = word boundary — prevents partial matches
        # re.escape handles special chars in keywords like "ag-gag"
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, text):
            matches.append(keyword)
    return matches


def _score_matches(
    pro_strong:   list[str],
    pro_moderate: list[str],
    pro_weak:     list[str],
    anti_strong:  list[str],
    anti_moderate: list[str],
    anti_weak:    list[str],
) -> float:
    """
    Computes a 0–1 relevance score from keyword matches.

    How it works:
      1. Each matched keyword contributes its tier weight
         (strong=1.0, moderate=0.6, weak=0.3)
      2. Pro and anti matches are combined (we care about
         relevance here, not direction — direction is determined
         by WHICH keywords matched, not the score)
      3. The raw sum is divided by MAX_RAW_SCORE and clamped to [0, 1]

    Examples:
      - 1 strong match        → 1.0 / 3.0 = 0.33
      - 2 strong matches      → 2.0 / 3.0 = 0.67
      - 3+ strong matches     → capped at 1.0
      - 1 moderate + 1 weak   → 0.9 / 3.0 = 0.30
      - Only weak matches     → low score, won't pass threshold alone
    """
    raw = 0.0

    raw += len(pro_strong)   * TIER_WEIGHTS["strong"]
    raw += len(pro_moderate) * TIER_WEIGHTS["moderate"]
    raw += len(pro_weak)     * TIER_WEIGHTS["weak"]
    raw += len(anti_strong)  * TIER_WEIGHTS["strong"]
    raw += len(anti_moderate) * TIER_WEIGHTS["moderate"]
    raw += len(anti_weak)    * TIER_WEIGHTS["weak"]

    # Normalize to 0–1, cap at 1.0
    score = min(raw / MAX_RAW_SCORE, 1.0)

    return round(score, 4)


# ─────────────────────────────────────────────────────
# PUBLIC API
# This is the function you call from the pipeline
# ─────────────────────────────────────────────────────

def classify_bill_keywords(bill: dict) -> dict:
    """
    Runs keyword filtering on a single bill.

    Parameters:
        bill (dict): A bill row from the database.
                     Must have keys: title, description, subjects.

    Returns:
        dict with exactly 3 keys:
          - keyword_match  (bool):  True if ANY keyword was found
          - keywords_found (list):  All matched keywords with their tier/direction
          - keyword_score  (float): 0–1 relevance score

    Example return:
        {
            "keyword_match":  True,
            "keywords_found": ["animal cruelty|pro|strong", "shelter|pro|weak"],
            "keyword_score":  0.4333
        }
    """
    text = _build_search_text(bill)

    if not text:
        return {
            "keyword_match":  False,
            "keywords_found": [],
            "keyword_score":  0.0,
        }

    # ── Run matching against all 6 lists ──
    pro_s  = _find_matches(text, PRO_ANIMAL_STRONG)
    pro_m  = _find_matches(text, PRO_ANIMAL_MODERATE)
    pro_w  = _find_matches(text, PRO_ANIMAL_WEAK)
    anti_s = _find_matches(text, ANTI_ANIMAL_STRONG)
    anti_m = _find_matches(text, ANTI_ANIMAL_MODERATE)
    anti_w = _find_matches(text, ANTI_ANIMAL_WEAK)

    # ── Build the keywords_found list ──
    # Format: "keyword|direction|tier"
    # This makes it easy to parse later in the pipeline
    keywords_found = []
    for kw in pro_s:   keywords_found.append(f"{kw}|pro|strong")
    for kw in pro_m:   keywords_found.append(f"{kw}|pro|moderate")
    for kw in pro_w:   keywords_found.append(f"{kw}|pro|weak")
    for kw in anti_s:  keywords_found.append(f"{kw}|anti|strong")
    for kw in anti_m:  keywords_found.append(f"{kw}|anti|moderate")
    for kw in anti_w:  keywords_found.append(f"{kw}|anti|weak")

    # ── Compute score ──
    score = _score_matches(pro_s, pro_m, pro_w, anti_s, anti_m, anti_w)

    return {
        "keyword_match":  len(keywords_found) > 0,
        "keywords_found": keywords_found,
        "keyword_score":  score,
    }


# ─────────────────────────────────────────────────────
# BATCH PROCESSOR
# Runs keyword filter on all bills from the database
# ─────────────────────────────────────────────────────

def run_keyword_filter(bills: list[dict], verbose: bool = True) -> list[dict]:
    """
    Runs the keyword filter on a list of bills.

    Parameters:
        bills (list[dict]): List of bill dicts from db.get_all_bills()
        verbose (bool):     If True, prints progress every 500 bills

    Returns:
        list[dict] — Each dict has:
            bill_id, keyword_match, keywords_found, keyword_score
    """
    results = []
    total = len(bills)
    matched = 0

    if verbose:
        print(f"\n  🔍 Running keyword filter on {total:,} bills...\n")

    for i, bill in enumerate(bills, 1):
        result = classify_bill_keywords(bill)
        result["bill_id"] = bill["bill_id"]

        if result["keyword_match"]:
            matched += 1

        results.append(result)

        # Progress logging
        if verbose and (i % 500 == 0 or i == 1 or i == total):
            print(
                f"  [{i:>6,}/{total:,}]  "
                f"🎯 {matched:,} matched so far  "
                f"({matched / i * 100:.1f}%)"
            )

    if verbose:
        print(f"\n  ✅ Keyword filter complete.")
        print(f"  📊 {matched:,} / {total:,} bills matched "
              f"({matched / total * 100:.1f}%)\n")

    return results


# ─────────────────────────────────────────────────────
# STANDALONE TEST
# Run this file directly to test against your database:
#   python -m src.classifier.keyword_filter
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.db import get_all_bills

    print("=" * 55)
    print("  PHASE 3 — KEYWORD FILTER TEST")
    print("=" * 55)

    bills = get_all_bills()
    print(f"\n  📦 Loaded {len(bills):,} bills from database")

    results = run_keyword_filter(bills)

    # ── Show some examples ──
    matched_results = [r for r in results if r["keyword_match"]]

    if matched_results:
        print("\n  ── TOP 10 HIGHEST-SCORING MATCHES ──\n")
        top_10 = sorted(matched_results, key=lambda r: r["keyword_score"], reverse=True)[:10]
        for r in top_10:
            # Find the bill title for display
            bill = next(b for b in bills if b["bill_id"] == r["bill_id"])
            title = bill["title"][:70] + "..." if len(bill["title"]) > 70 else bill["title"]
            print(f"  {r['bill_id']:15s}  score={r['keyword_score']:.4f}  {title}")
            # Show first 3 keywords
            kw_display = r["keywords_found"][:3]
            print(f"  {'':15s}  keywords: {', '.join(kw_display)}")
            print()
    else:
        print("\n  ⚠️  No bills matched any keywords!")

    # ── Summary stats ──
    scores = [r["keyword_score"] for r in matched_results]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"  ── SCORE DISTRIBUTION ──")
        print(f"  Avg score (matched only): {avg_score:.4f}")
        print(f"  Score >= 0.5:  {sum(1 for s in scores if s >= 0.5):,}")
        print(f"  Score >= 0.3:  {sum(1 for s in scores if s >= 0.3):,}")
        print(f"  Score >= 0.1:  {sum(1 for s in scores if s >= 0.1):,}")
        print(f"  Score <  0.1:  {sum(1 for s in scores if s <  0.1):,}")
    print()
