# src/classifier/groq_classifier.py
#
# Phase 6 — Groq LLM Classifier
# Reasoning-based classification using llama-3.3-70b-versatile.
#
# Key features:
#   - Skips bills already classified by Groq (resume-safe)
#   - Prioritizes keyword-matched bills, then embedding-ranked
#   - Stores results incrementally to SQLite after each call
#   - Rate-limit aware (2.5s delay, retry on 429)
#   - Accepts keyword_score + embedding_similarity as LLM context

import json
import time
import os
import requests
from datetime import datetime, timezone

from config.settings import GROQ_API_KEY, GROQ_MODEL
from src.utils.db import get_connection, save_classification, get_classification, utcnow


# ─────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_DELAY = 2.5   # seconds between calls — stay under 30 req/min free limit
MAX_RETRIES   = 3     # retry attempts on 429 rate-limit errors

SYSTEM_PROMPT = """You are an expert legislative analyst for an animal advocacy organization.

Classify each bill as EXACTLY one of:
  "pro_animal"  — Bill strengthens animal welfare protections, increases cruelty penalties,
                   bans harmful practices (puppy mills, factory farm confinement, testing),
                   protects wildlife/habitat, or expands animal rights.
  "anti_animal" — Bill weakens animal protections, shields industry from oversight,
                   expands hunting/trapping rights, legalizes cruelty exemptions,
                   criminalizes whistleblower investigations (ag-gag), or preempts
                   local welfare ordinances.
  "neutral"     — Bill is NOT primarily about animals, or has no clear animal
                   welfare impact (e.g. tax bills, infrastructure, human healthcare).

Rules:
- If a bill mentions animals only incidentally, classify as "neutral"
- If uncertain between pro and anti, lean toward the label with stronger evidence
- "confidence" must reflect your certainty: 0.9+ = very clear, 0.5–0.7 = ambiguous

Respond ONLY with valid JSON — no markdown, no explanation outside the JSON:
{"label": "pro_animal", "confidence": 0.87, "reasoning": "One or two sentence explanation under 60 words."}"""


# ─────────────────────────────────────────────────────
# 1. LOAD API KEY
# ─────────────────────────────────────────────────────

def load_api_key() -> str:
    """
    Returns the Groq API key from environment / .env file.
    Raises ValueError if not set so failures are caught early.
    """
    key = GROQ_API_KEY
    if not key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file:\n"
            "  GROQ_API_KEY=gsk_..."
        )
    return key


# ─────────────────────────────────────────────────────
# 2. BUILD PROMPT
# ─────────────────────────────────────────────────────

def build_prompt(bill: dict, kw_score: float = 0.0, emb_score: float = 0.0) -> str:
    """
    Constructs the user-turn message for the LLM.

    Includes keyword_score and embedding_score as additional
    signals so the LLM can weight its decision accordingly.

    Parameters:
        bill (dict):       Bill row from SQLite
        kw_score (float):  keyword_score from Phase 3 (0–1)
        emb_score (float): embedding_similarity from Phase 4 (0–1)

    Returns:
        str: Formatted prompt string
    """
    title       = bill.get("title", "No title")
    description = bill.get("description", "") or ""
    subjects    = bill.get("subjects", "")    or ""
    committee   = bill.get("committee", "")   or ""
    state       = bill.get("state", "")

    lines = [
        f"State: {state}",
        f"Title: {title}",
    ]
    if description:
        lines.append(f"Description: {description[:500]}")   # cap at 500 chars
    if subjects:
        lines.append(f"Subjects: {subjects}")
    if committee:
        lines.append(f"Committee: {committee}")

    # Add prior-stage signals as soft hints
    lines.append(f"\nPrior analysis signals:")
    lines.append(f"  Keyword relevance score: {kw_score:.2f}  (0=no match, 1=strong match)")
    lines.append(f"  Semantic similarity score: {emb_score:.2f}  (0=unrelated, 1=very similar)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# 3. CLASSIFY SINGLE BILL (raw API call)
# ─────────────────────────────────────────────────────

def classify_bill(
    bill: dict,
    kw_score: float = 0.0,
    emb_score: float = 0.0,
    api_key: str = "",
) -> dict:
    """
    Calls Groq API to classify a single bill.

    Parameters:
        bill (dict):       Bill row from SQLite
        kw_score (float):  Keyword score from Phase 3
        emb_score (float): Embedding similarity from Phase 4
        api_key (str):     Groq API key

    Returns:
        dict:
          groq_label, groq_confidence, groq_reasoning,
          groq_classified_at, success, error
    """
    prompt = build_prompt(bill, kw_score, emb_score)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature":     0.1,
        "max_tokens":      220,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        return _error_result(f"Network error: {e}")

    if resp.status_code == 429:
        return _error_result("rate_limited_429")

    if resp.status_code != 200:
        return _error_result(f"API {resp.status_code}: {resp.text[:200]}")

    try:
        raw_content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)
    except (KeyError, json.JSONDecodeError, IndexError) as e:
        return _error_result(f"Parse error: {e}")

    # Validate label
    label = parsed.get("label", "neutral")
    if label not in ("pro_animal", "anti_animal", "neutral"):
        label = "neutral"

    # Validate confidence
    try:
        conf = float(parsed.get("confidence", 0.5))
        conf = round(max(0.0, min(1.0, conf)), 4)
    except (ValueError, TypeError):
        conf = 0.5

    reasoning = str(parsed.get("reasoning", ""))[:500]

    return {
        "groq_label":         label,
        "groq_confidence":    conf,
        "groq_reasoning":     reasoning,
        "groq_classified_at": utcnow(),
        "success":            True,
        "error":              None,
    }


def _error_result(msg: str) -> dict:
    return {
        "groq_label":         "neutral",
        "groq_confidence":    0.0,
        "groq_reasoning":     "",
        "groq_classified_at": "",
        "success":            False,
        "error":              msg,
    }


# ─────────────────────────────────────────────────────
# 4. STORE RESULTS TO DB
# ─────────────────────────────────────────────────────

def store_results(bill_id: str, result: dict):
    """
    Writes (or updates) a Groq classification result to the
    classifications table in SQLite.

    Only stores the Groq-specific fields — leaves all other
    columns (keyword_*, embedding_*, openpaws_*, final_*) untouched
    if a row already exists, or sets them to defaults for new rows.
    """
    now = utcnow()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM classifications WHERE bill_id = ?", (bill_id,)
    )
    exists = cursor.fetchone()

    if exists:
        cursor.execute("""
            UPDATE classifications
            SET groq_label           = :groq_label,
                groq_confidence      = :groq_confidence,
                groq_reasoning       = :groq_reasoning,
                groq_classified_at   = :groq_classified_at,
                updated_at           = :updated_at
            WHERE bill_id = :bill_id
        """, {
            "bill_id":             bill_id,
            "groq_label":          result["groq_label"],
            "groq_confidence":     result["groq_confidence"],
            "groq_reasoning":      result["groq_reasoning"],
            "groq_classified_at":  result["groq_classified_at"],
            "updated_at":          now,
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
                :groq_label, :groq_confidence, :groq_reasoning, :groq_classified_at,
                0.0, '', '',
                '', 0.0, 0.0, '',
                :classified_at, :updated_at
            )
        """, {
            "bill_id":            bill_id,
            "groq_label":         result["groq_label"],
            "groq_confidence":    result["groq_confidence"],
            "groq_reasoning":     result["groq_reasoning"],
            "groq_classified_at": result["groq_classified_at"],
            "classified_at":      now,
            "updated_at":         now,
        })

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────
# 5. RESUME PROGRESS
# ─────────────────────────────────────────────────────

def resume_progress() -> set:
    """
    Returns the set of bill_ids that already have a Groq
    classification stored in the database.

    Used by classify_batch() to skip already-processed bills,
    making the pipeline safely resumable after interruption.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT bill_id FROM classifications
        WHERE groq_label IS NOT NULL
          AND groq_label != ''
          AND groq_classified_at IS NOT NULL
          AND groq_classified_at != ''
    """).fetchall()
    conn.close()
    return {r["bill_id"] for r in rows}


# ─────────────────────────────────────────────────────
# 6. CLASSIFY BATCH (main pipeline entry point)
# ─────────────────────────────────────────────────────

def classify_batch(
    candidate_bills: list[dict],
    delay: float = REQUEST_DELAY,
    verbose: bool = True,
) -> dict:
    """
    Classifies a list of candidate bills through Groq, with:
      - Resume support (skips already-classified bills)
      - Incremental DB writes after each successful API call
      - Rate-limit retries (exponential backoff on 429)
      - Progress logging every 10 bills

    Parameters:
        candidate_bills (list[dict]):
            Each dict MUST have:
              bill_id, title, description, subjects, committee, state
            SHOULD have (from prior phases):
              keyword_score, embedding_similarity

        delay (float):   Seconds between API calls
        verbose (bool):  Print progress

    Returns:
        dict: {
            "processed": int,
            "skipped":   int,
            "succeeded": int,
            "failed":    int,
            "results":   list[dict]
        }
    """
    api_key = load_api_key()

    # ── Load already-done bill_ids ──
    already_done = resume_progress()
    total_candidates = len(candidate_bills)

    if verbose:
        print(f"\n  🤖 Groq Batch Classifier")
        print(f"     Model:       {GROQ_MODEL}")
        print(f"     Candidates:  {total_candidates:,}")
        print(f"     Already done: {len(already_done):,} (will skip)")
        print(f"     Delay:       {delay}s/request")

    # ── Filter out already-done bills ──
    todo = [b for b in candidate_bills if b.get("bill_id") not in already_done]

    if verbose:
        est_mins = (len(todo) * delay) / 60
        print(f"     To process:  {len(todo):,}")
        print(f"     Est. time:   ~{est_mins:.0f} min\n")

    if not todo:
        if verbose:
            print("  ✅ All bills already classified — nothing to do.\n")
        return {
            "processed": 0, "skipped": total_candidates,
            "succeeded": 0, "failed": 0, "results": [],
        }

    # ── Process ──
    results      = []
    succeeded    = 0
    failed       = 0

    for i, bill in enumerate(todo, 1):
        bill_id  = bill.get("bill_id", "?")
        kw_score  = float(bill.get("keyword_score", 0.0) or 0.0)
        emb_score = float(bill.get("embedding_similarity", 0.0) or 0.0)

        # Retry loop (handles 429 rate-limit)
        result = None
        for attempt in range(MAX_RETRIES):
            result = classify_bill(bill, kw_score, emb_score, api_key)

            if result["success"]:
                break

            if result["error"] == "rate_limited_429":
                wait = delay * (2 ** attempt)   # 2.5, 5, 10 sec
                if verbose:
                    print(f"  ⏳ Rate limited — waiting {wait:.0f}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                break   # non-retryable

        result["bill_id"] = bill_id

        # Incremental DB write
        if result["success"]:
            store_results(bill_id, result)
            succeeded += 1
        else:
            failed += 1

        results.append(result)

        # Progress log every 10 bills
        if verbose and (i % 10 == 0 or i == 1 or i == len(todo)):
            label = result.get("groq_label", "?")
            conf  = result.get("groq_confidence", 0.0)
            print(
                f"  Processed {i:>4,} / {len(todo):,} candidate bills  "
                f"[✅{succeeded} ❌{failed}]  "
                f"last → {label} ({conf:.2f})"
            )

        # Rate-limit pause
        if i < len(todo):
            time.sleep(delay)

    # ── Summary ──
    if verbose:
        labels = [r["groq_label"] for r in results if r["success"]]
        print(f"\n  ✅ Batch complete.")
        print(f"     Succeeded: {succeeded:,} | Failed: {failed:,} | "
              f"Skipped: {len(already_done):,}")
        print(f"\n  Label distribution:")
        for lbl in ("pro_animal", "anti_animal", "neutral"):
            n   = labels.count(lbl)
            pct = n / len(labels) * 100 if labels else 0
            print(f"    {lbl:12s}: {n:,} ({pct:.1f}%)")
        print()

    return {
        "processed": len(todo),
        "skipped":   len(already_done),
        "succeeded": succeeded,
        "failed":    failed,
        "results":   results,
    }


# ─────────────────────────────────────────────────────
# CANDIDATE SELECTION HELPER
# ─────────────────────────────────────────────────────

def get_candidate_bills() -> list[dict]:
    """
    Pulls candidate bills from SQLite for Groq classification.

    Priority order:
      1. Keyword-matched bills (keyword_match = 1), sorted by keyword_score DESC
      2. Embedding-ranked bills (embedding_similarity >= 0.4), sorted by score DESC

    De-duplicates so no bill appears twice.
    Returns a flat list of bill dicts enriched with their scores.
    """
    conn = get_connection()

    # ── Priority 1: keyword matches ──
    kw_rows = conn.execute("""
        SELECT b.*, c.keyword_score, c.keyword_match,
               c.embedding_similarity, c.embedding_label
        FROM bills b
        LEFT JOIN classifications c ON b.bill_id = c.bill_id
        WHERE c.keyword_match = 1
        ORDER BY c.keyword_score DESC
    """).fetchall()

    # ── Priority 2: embedding-ranked (not already in list) ──
    kw_ids = {r["bill_id"] for r in kw_rows}

    emb_rows = conn.execute("""
        SELECT b.*, c.keyword_score, c.keyword_match,
               c.embedding_similarity, c.embedding_label
        FROM bills b
        LEFT JOIN classifications c ON b.bill_id = c.bill_id
        WHERE c.embedding_similarity >= 0.4
          AND (c.keyword_match = 0 OR c.keyword_match IS NULL)
        ORDER BY c.embedding_similarity DESC
    """).fetchall()

    conn.close()

    # Merge — keyword-matched first, then embedding-ranked
    candidates = [dict(r) for r in kw_rows]
    candidates += [dict(r) for r in emb_rows if r["bill_id"] not in kw_ids]

    return candidates


# ─────────────────────────────────────────────────────
# STANDALONE EXECUTION
# python -m src.classifier.groq_classifier
# (or: python src/classifier/groq_classifier.py)
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("=" * 55)
    print("  PHASE 6 — GROQ LLM CLASSIFIER")
    print("=" * 55)

    # Validate API key early
    try:
        key = load_api_key()
        print(f"\n  API Key : {key[:8]}...{key[-4:]}")
        print(f"  Model   : {GROQ_MODEL}")
    except ValueError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)

    # ── Load candidate bills ──
    print("\n  Loading candidate bills from database...")
    candidates = get_candidate_bills()
    print(f"  Found {len(candidates):,} candidates")
    print(f"    (keyword-matched first, then embedding-ranked)")

    if not candidates:
        print("\n  ⚠️  No candidates found. Run Phase 3 + 4 first.")
        sys.exit(0)

    # ── Show already-done count ──
    done = resume_progress()
    print(f"  Already classified: {len(done):,}")
    remaining = [b for b in candidates if b["bill_id"] not in done]
    print(f"  Remaining to classify: {len(remaining):,}")

    if not remaining:
        print("\n  ✅ All candidates already classified!")
        sys.exit(0)

    # ── Run ──
    summary = classify_batch(candidates, delay=REQUEST_DELAY, verbose=True)
    print(f"\n  Run complete. Results stored in SQLite classifications table.")
