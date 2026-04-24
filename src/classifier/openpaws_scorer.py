# src/classifier/openpaws_scorer.py
#
# Phase 7a — Open Paws Alignment Scorer
#
# PURPOSE: Different from Groq classification (Phase 6).
#   Groq asks: "Is this pro/anti/neutral?"
#   Open Paws asks: "How does this bill FRAME animal welfare?
#                    What is the alignment score on a -1 to +1 scale?"
#
# BACKEND: HuggingFace Inference API (free, no GPU needed).
#   Falls back to Groq with alignment-specific prompt if HF fails.

import json
import time
import requests as http_requests
from datetime import datetime, timezone

from config.settings import (
    HF_TOKEN, OPENPAWS_MODEL, OPENPAWS_FALLBACK,
    GROQ_API_KEY, GROQ_MODEL,
)
from src.utils.db import get_connection, utcnow


# ─────────────────────────────────────────────────────
# ALIGNMENT PROMPT (different lens than Groq classifier)
# ─────────────────────────────────────────────────────

ALIGNMENT_PROMPT = """You are an animal welfare policy analyst for an advocacy organization called Open Paws.

Your task: Analyze a bill's FRAMING and ALIGNMENT with animal welfare values.

This is NOT simple classification. Evaluate:
1. Does the bill's language frame animals as sentient beings deserving protection, or as property/resources?
2. Does it strengthen or weaken the legal/regulatory framework protecting animals?
3. What specific policy mechanisms does it use (bans, penalties, exemptions, deregulation)?

Return ONLY valid JSON:
{"alignment_score": 0.75, "framing_summary": "Brief 1-3 sentence analysis of how the bill frames animal welfare."}

Rules for alignment_score:
  +1.0 = Strongest possible pro-animal framing (e.g. animal cruelty felony, factory farm ban)
  +0.5 = Moderately pro-animal (e.g. shelter funding, spay/neuter programs)
   0.0 = Neutral or unrelated to animals
  -0.5 = Moderately anti-animal (e.g. hunting license expansion, ag exemptions)
  -1.0 = Strongest anti-animal framing (e.g. ag-gag laws, cruelty exemptions)

"framing_summary" must be under 80 words. Focus on HOW the bill frames animals, not just WHAT it does."""


# ─────────────────────────────────────────────────────
# HF INFERENCE API
# ─────────────────────────────────────────────────────

HF_API_URL = "https://router.huggingface.co/hf-inference/models"


def _call_hf_api(bill: dict) -> dict:
    """
    Calls HuggingFace Inference API with the Open Paws alignment prompt.
    Uses the serverless inference endpoint (free tier).
    """
    title = bill.get("title", "")
    description = bill.get("description", "") or ""
    subjects = bill.get("subjects", "") or ""

    user_msg = f"Bill Title: {title}"
    if description:
        user_msg += f"\nDescription: {description[:400]}"
    if subjects:
        user_msg += f"\nSubjects: {subjects}"

    # HF chat completions endpoint
    url = f"{HF_API_URL}/{OPENPAWS_MODEL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "model": OPENPAWS_MODEL,
        "messages": [
            {"role": "system", "content": ALIGNMENT_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens":  250,
    }

    try:
        resp = http_requests.post(url, headers=headers, json=payload, timeout=45)
    except http_requests.exceptions.RequestException as e:
        return {"success": False, "error": f"HF network error: {e}"}

    if resp.status_code == 503:
        return {"success": False, "error": "HF model loading (503)"}
    if resp.status_code == 429:
        return {"success": False, "error": "HF rate limited (429)"}
    if resp.status_code != 200:
        return {"success": False, "error": f"HF error {resp.status_code}: {resp.text[:200]}"}

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        # Extract JSON from response — model may wrap in markdown
        json_str = content
        if "```" in content:
            # Strip markdown code fences
            json_str = content.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        parsed = json.loads(json_str.strip())
        return {"success": True, "data": parsed}
    except (KeyError, json.JSONDecodeError, IndexError) as e:
        return {"success": False, "error": f"HF parse error: {e}"}


# ─────────────────────────────────────────────────────
# GROQ FALLBACK (same API, different prompt)
# ─────────────────────────────────────────────────────

def _call_groq_alignment(bill: dict) -> dict:
    """
    Uses Groq API with the alignment-specific prompt as fallback
    when HuggingFace Inference API is unavailable.
    """
    if not GROQ_API_KEY:
        return {"success": False, "error": "No GROQ_API_KEY for fallback"}

    title = bill.get("title", "")
    description = bill.get("description", "") or ""
    subjects = bill.get("subjects", "") or ""

    user_msg = f"Bill Title: {title}"
    if description:
        user_msg += f"\nDescription: {description[:400]}"
    if subjects:
        user_msg += f"\nSubjects: {subjects}"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": ALIGNMENT_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature":     0.1,
        "max_tokens":      250,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=30,
        )
    except http_requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Groq fallback error: {e}"}

    if resp.status_code == 429:
        return {"success": False, "error": "Groq fallback rate limited (429)"}
    if resp.status_code != 200:
        return {"success": False, "error": f"Groq {resp.status_code}"}

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return {"success": True, "data": parsed}
    except (KeyError, json.JSONDecodeError, IndexError) as e:
        return {"success": False, "error": f"Groq parse error: {e}"}


# ─────────────────────────────────────────────────────
# SCORE SINGLE BILL
# ─────────────────────────────────────────────────────

def score_bill_alignment(bill: dict) -> dict:
    """
    Scores a single bill's animal welfare alignment.
    Tries HuggingFace first, falls back to Groq.

    Returns:
        dict:
          openpaws_alignment_score (float):  -1.0 to +1.0
          openpaws_framing_summary (str):    Analysis text
          openpaws_scored_at (str):          ISO timestamp
          success (bool)
          error (str|None)
          backend (str):                     "hf" or "groq"
    """
    # ── Try HuggingFace first ──
    result = _call_hf_api(bill)

    backend = "hf"
    if not result["success"]:
        # ── Fallback to Groq ──
        backend = "groq"
        result = _call_groq_alignment(bill)

    if not result["success"]:
        return {
            "openpaws_alignment_score": 0.0,
            "openpaws_framing_summary": "",
            "openpaws_scored_at":       "",
            "success": False,
            "error":   result.get("error", "Unknown error"),
            "backend": backend,
        }

    # ── Parse and validate ──
    data = result["data"]

    try:
        score = float(data.get("alignment_score", 0.0))
        score = round(max(-1.0, min(1.0, score)), 4)
    except (ValueError, TypeError):
        score = 0.0

    summary = str(data.get("framing_summary", ""))[:500]

    return {
        "openpaws_alignment_score": score,
        "openpaws_framing_summary": summary,
        "openpaws_scored_at":       utcnow(),
        "success": True,
        "error":   None,
        "backend": backend,
    }


# ─────────────────────────────────────────────────────
# DB STORAGE
# ─────────────────────────────────────────────────────

def store_alignment(bill_id: str, result: dict):
    """Writes Open Paws alignment results to classifications table."""
    conn = get_connection()
    cursor = conn.cursor()
    now = utcnow()

    cursor.execute(
        "SELECT id FROM classifications WHERE bill_id = ?", (bill_id,)
    )
    exists = cursor.fetchone()

    if exists:
        cursor.execute("""
            UPDATE classifications
            SET openpaws_alignment_score = :score,
                openpaws_framing_summary = :summary,
                openpaws_scored_at       = :scored_at,
                updated_at               = :updated_at
            WHERE bill_id = :bill_id
        """, {
            "bill_id":    bill_id,
            "score":      result["openpaws_alignment_score"],
            "summary":    result["openpaws_framing_summary"],
            "scored_at":  result["openpaws_scored_at"],
            "updated_at": now,
        })
    else:
        cursor.execute("""
            INSERT INTO classifications (
                bill_id,
                keyword_match, keywords_found, keyword_score,
                embedding_similarity, embedding_label,
                groq_label, groq_confidence, groq_reasoning, groq_classified_at,
                openpaws_alignment_score, openpaws_framing_summary, openpaws_scored_at,
                final_label, final_confidence, relevance_score, risk_level,
                classified_at, updated_at
            ) VALUES (
                :bill_id,
                0, '', 0.0,
                0.0, '',
                '', 0.0, '', '',
                :score, :summary, :scored_at,
                '', 0.0, 0.0, '',
                :classified_at, :updated_at
            )
        """, {
            "bill_id":       bill_id,
            "score":         result["openpaws_alignment_score"],
            "summary":       result["openpaws_framing_summary"],
            "scored_at":     result["openpaws_scored_at"],
            "classified_at": now,
            "updated_at":    now,
        })

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────
# RESUME SUPPORT
# ─────────────────────────────────────────────────────

def get_already_scored() -> set:
    """Returns bill_ids that already have Open Paws alignment scores."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT bill_id FROM classifications
        WHERE openpaws_scored_at IS NOT NULL
          AND openpaws_scored_at != ''
    """).fetchall()
    conn.close()
    return {r["bill_id"] for r in rows}


# ─────────────────────────────────────────────────────
# BATCH PROCESSOR
# ─────────────────────────────────────────────────────

def run_openpaws_scorer(
    bills: list[dict],
    delay: float = 2.5,
    max_retries: int = 2,
    verbose: bool = True,
) -> list[dict]:
    """
    Scores a list of bills through the Open Paws alignment layer.
    Resume-safe, incremental DB writes, rate-limit aware.
    """
    already_done = get_already_scored()
    todo = [b for b in bills if b.get("bill_id") not in already_done]

    if verbose:
        print(f"\n  🐾 Open Paws Alignment Scorer")
        print(f"     Primary backend:  HuggingFace ({OPENPAWS_MODEL})")
        print(f"     Fallback:         Groq ({GROQ_MODEL})")
        print(f"     Candidates: {len(bills):,}  |  Already scored: {len(already_done):,}")
        print(f"     To process: {len(todo):,}")
        est_mins = (len(todo) * delay) / 60
        print(f"     Est. time:  ~{est_mins:.0f} min\n")

    if not todo:
        if verbose:
            print("  ✅ All bills already scored.\n")
        return []

    results   = []
    succeeded = 0
    failed    = 0
    hf_count  = 0
    groq_count = 0

    for i, bill in enumerate(todo, 1):
        bill_id = bill.get("bill_id", "?")

        # Retry loop
        result = None
        for attempt in range(max_retries):
            result = score_bill_alignment(bill)
            if result["success"]:
                break
            if "429" in str(result.get("error", "")):
                wait = delay * (2 ** attempt)
                if verbose:
                    print(f"  ⏳ Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                break

        result["bill_id"] = bill_id

        if result["success"]:
            store_alignment(bill_id, result)
            succeeded += 1
            if result["backend"] == "hf":
                hf_count += 1
            else:
                groq_count += 1
        else:
            failed += 1

        results.append(result)

        if verbose and (i % 10 == 0 or i == 1 or i == len(todo)):
            score = result.get("openpaws_alignment_score", 0.0)
            bk = result.get("backend", "?")
            print(
                f"  Processed {i:>4,} / {len(todo):,}  "
                f"[✅{succeeded} ❌{failed}]  "
                f"score={score:+.2f}  via={bk}"
            )

        if i < len(todo):
            time.sleep(delay)

    if verbose:
        print(f"\n  ✅ Open Paws scoring complete.")
        print(f"     Succeeded: {succeeded:,}  |  Failed: {failed:,}")
        print(f"     HF calls: {hf_count:,}  |  Groq fallback: {groq_count:,}")

        scores = [r["openpaws_alignment_score"] for r in results if r["success"]]
        if scores:
            import statistics
            print(f"\n  Score distribution:")
            print(f"    Mean:  {statistics.mean(scores):+.3f}")
            print(f"    Min:   {min(scores):+.3f}")
            print(f"    Max:   {max(scores):+.3f}")
        print()

    return results


# ─────────────────────────────────────────────────────
# STANDALONE TEST
# python -m src.classifier.openpaws_scorer
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("=" * 55)
    print("  PHASE 7a — OPEN PAWS ALIGNMENT TEST")
    print("=" * 55)

    test_bills = [
        {
            "bill_id": "TEST_ALIGN_PRO",
            "title": "Animal Cruelty Prevention and Puppy Mill Ban Act",
            "description": "Bans commercial puppy mills and increases animal cruelty to felony",
            "subjects": "Animals, Criminal Law",
        },
        {
            "bill_id": "TEST_ALIGN_ANTI",
            "title": "Agricultural Operations Protection Act",
            "description": "Criminalizes unauthorized recording at farms and slaughterhouses",
            "subjects": "Agriculture, Criminal Law",
        },
        {
            "bill_id": "TEST_ALIGN_NEUTRAL",
            "title": "Highway Infrastructure Improvement Act",
            "description": "Allocates funding for bridge repair on state highways",
            "subjects": "Transportation, Budget",
        },
    ]

    print()
    for bill in test_bills:
        result = score_bill_alignment(bill)
        status = "✅" if result["success"] else "❌"
        score = result["openpaws_alignment_score"]
        bk = result.get("backend", "?")
        print(f"  {status} {bill['bill_id']:22s}  score={score:+.2f}  via={bk}")
        if result["openpaws_framing_summary"]:
            print(f"     {result['openpaws_framing_summary'][:80]}")
        if result.get("error"):
            print(f"     ERROR: {result['error']}")
        print()
