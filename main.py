# main.py
#
# Paw-dvocate — Full Pipeline Orchestrator
#
# Runs all classification stages in sequence:
#   Phase 3: Keyword filter
#   Phase 4: Embedding similarity
#   Phase 6: Groq LLM classification
#   Phase 7: Open Paws alignment + Ensemble
#   Phase 8: Markdown digest generation
#
# Usage:
#   python main.py --run-all              # Full pipeline (all stages)
#   python main.py --ingest               # Ingest raw JSON data
#   python main.py --classify             # Run all classification stages
#   python main.py --digest               # Generate digest only
#   python main.py --email                # Email the latest digest
#   python main.py --stage keyword        # Run a single stage
#   python main.py --stage embedding
#   python main.py --stage groq
#   python main.py --stage openpaws
#   python main.py --stage ensemble
#   python main.py --stage digest

import os
import sys
import time
import argparse
from datetime import datetime, timezone

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def print_header():
    print()
    print("=" * 55)
    print("  🐾 Paw-dvocate Legislative Intelligence Pipeline")
    print("=" * 55)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"  Started: {now}")
    print()


def print_stage(num: int, name: str):
    print()
    print(f"  {'─' * 50}")
    print(f"  Stage {num}: {name}")
    print(f"  {'─' * 50}")


# ─────────────────────────────────────────────────────
# STAGE 0: INGEST RAW DATA
# ─────────────────────────────────────────────────────

def run_stage_ingest():
    """Ingests raw LegiScan JSON data from data/raw/."""
    print_stage(0, "Data Ingestion")

    from src.api.ingestor import ingest_all_json_files

    result = ingest_all_json_files(verbose=True)
    return result


# ─────────────────────────────────────────────────────
# STAGE 1: KEYWORD FILTER
# ─────────────────────────────────────────────────────

def run_stage_keyword():
    """Runs keyword filter on all bills and stores results."""
    print_stage(1, "Keyword Filter")

    from src.utils.db import get_all_bills, get_connection, utcnow
    from src.classifier.keyword_filter import run_keyword_filter

    bills = get_all_bills()
    print(f"  Loaded {len(bills):,} bills")

    results = run_keyword_filter(bills, verbose=True)

    # Write keyword results to classifications table
    print("  Writing keyword results to database...")
    conn = get_connection()
    cursor = conn.cursor()
    now = utcnow()
    written = 0

    for r in results:
        bill_id = r["bill_id"]
        kw_found_str = ", ".join(r["keywords_found"]) if r["keywords_found"] else ""

        cursor.execute(
            "SELECT id FROM classifications WHERE bill_id = ?", (bill_id,)
        )
        exists = cursor.fetchone()

        if exists:
            cursor.execute("""
                UPDATE classifications
                SET keyword_match = :kw_match,
                    keywords_found = :kw_found,
                    keyword_score = :kw_score,
                    updated_at = :updated_at
                WHERE bill_id = :bill_id
            """, {
                "bill_id":    bill_id,
                "kw_match":   1 if r["keyword_match"] else 0,
                "kw_found":   kw_found_str,
                "kw_score":   r["keyword_score"],
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
                    :kw_match, :kw_found, :kw_score,
                    0.0, '',
                    '', 0.0, '', '',
                    0.0, '', '',
                    '', 0.0, 0.0, '',
                    :now, :now
                )
            """, {
                "bill_id":  bill_id,
                "kw_match": 1 if r["keyword_match"] else 0,
                "kw_found": kw_found_str,
                "kw_score": r["keyword_score"],
                "now":      now,
            })
        written += 1

    conn.commit()
    conn.close()
    print(f"  ✅ {written:,} keyword results saved to DB")

    return results


# ─────────────────────────────────────────────────────
# STAGE 2: EMBEDDING SCORER
# ─────────────────────────────────────────────────────

def run_stage_embedding():
    """Runs embedding similarity on all bills and stores results."""
    print_stage(2, "Embedding Similarity Scorer")

    from src.utils.db import get_all_bills, get_connection, utcnow
    from src.classifier.embedding_scorer import run_embedding_scorer

    bills = get_all_bills()
    results = run_embedding_scorer(bills, verbose=True)

    # Write embedding results to classifications table
    print("  Writing embedding results to database...")
    conn = get_connection()
    cursor = conn.cursor()
    now = utcnow()
    written = 0

    for r in results:
        bill_id = r["bill_id"]
        cursor.execute(
            "SELECT id FROM classifications WHERE bill_id = ?", (bill_id,)
        )
        exists = cursor.fetchone()

        if exists:
            cursor.execute("""
                UPDATE classifications
                SET embedding_similarity = :emb_sim,
                    embedding_label = :emb_label,
                    updated_at = :updated_at
                WHERE bill_id = :bill_id
            """, {
                "bill_id":    bill_id,
                "emb_sim":    r["embedding_similarity"],
                "emb_label":  r["embedding_label"],
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
                    :emb_sim, :emb_label,
                    '', 0.0, '', '',
                    0.0, '', '',
                    '', 0.0, 0.0, '',
                    :now, :now
                )
            """, {
                "bill_id":   bill_id,
                "emb_sim":   r["embedding_similarity"],
                "emb_label": r["embedding_label"],
                "now":       now,
            })
        written += 1

    conn.commit()
    conn.close()
    print(f"  ✅ {written:,} embedding results saved to DB")

    return results


# ─────────────────────────────────────────────────────
# STAGE 3: GROQ LLM CLASSIFIER
# ─────────────────────────────────────────────────────

def run_stage_groq(max_bills: int = 0):
    """Runs Groq classifier on candidate bills."""
    print_stage(3, "Groq LLM Classifier")

    from src.classifier.groq_classifier import (
        get_candidate_bills, classify_batch
    )

    candidates = get_candidate_bills()
    print(f"  Found {len(candidates):,} candidates")

    if max_bills > 0:
        candidates = candidates[:max_bills]
        print(f"  Limited to first {max_bills} (use --max-bills to change)")

    return classify_batch(candidates, verbose=True)


# ─────────────────────────────────────────────────────
# STAGE 4: OPEN PAWS ALIGNMENT
# ─────────────────────────────────────────────────────

def run_stage_openpaws(max_bills: int = 0):
    """Runs Open Paws alignment on candidate bills."""
    print_stage(4, "Open Paws Alignment Scorer")

    from src.classifier.groq_classifier import get_candidate_bills
    from src.classifier.openpaws_scorer import run_openpaws_scorer

    candidates = get_candidate_bills()
    if max_bills > 0:
        candidates = candidates[:max_bills]

    return run_openpaws_scorer(candidates, verbose=True)


# ─────────────────────────────────────────────────────
# STAGE 5: WEIGHTED ENSEMBLE
# ─────────────────────────────────────────────────────

def run_stage_ensemble():
    """Runs the weighted ensemble on all classified bills."""
    print_stage(5, "Weighted Ensemble")

    from src.classifier.ensemble import run_ensemble
    return run_ensemble(verbose=True)


# ─────────────────────────────────────────────────────
# STAGE 6: DIGEST GENERATION
# ─────────────────────────────────────────────────────

def run_stage_digest(days_back: int = 0):
    """Generates the Markdown digest."""
    print_stage(6, "Markdown Digest Generation")

    from src.digest.generator import generate_weekly_digest
    return generate_weekly_digest(days_back=days_back, verbose=True)


# ─────────────────────────────────────────────────────
# STAGE 7: EMAIL DIGEST
# ─────────────────────────────────────────────────────

def run_stage_email(digest_path: str = "", recipient: str = ""):
    """Emails the latest digest."""
    print_stage(7, "Email Digest")

    from src.digest.email_sender import send_digest_email

    result = send_digest_email(digest_path=digest_path, recipient=recipient)
    if result["success"]:
        print(f"  ✅ {result['message']}")
    else:
        print(f"  ❌ {result['message']}")
    return result


# ─────────────────────────────────────────────────────
# COMPOSITE COMMANDS
# ─────────────────────────────────────────────────────

def run_full_pipeline(max_groq_bills: int = 50, days_back: int = 0):
    """Runs the entire pipeline end-to-end: ingest → classify → digest."""
    print_header()
    start = time.time()

    # Stage 0: Ingest (re-ingest any new data)
    try:
        run_stage_ingest()
    except Exception as e:
        print(f"  ⚠️  Ingest skipped: {e}")

    # Stage 1: Keywords (fast, ~1 min)
    run_stage_keyword()

    # Stage 2: Embeddings (medium, ~7 min on CPU)
    run_stage_embedding()

    # Stage 3: Groq API (slow, rate limited)
    run_stage_groq(max_bills=max_groq_bills)

    # Stage 4: Open Paws (slow, rate limited)
    run_stage_openpaws(max_bills=max_groq_bills)

    # Stage 5: Ensemble (instant)
    run_stage_ensemble()

    # Stage 6: Digest (instant)
    path = run_stage_digest(days_back=days_back)

    elapsed = time.time() - start
    print()
    print("=" * 55)
    print(f"  ✅ Pipeline complete in {elapsed / 60:.1f} minutes")
    if path:
        print(f"  📄 Digest: {path}")
    print("=" * 55)
    print()

    return path


def run_classify_only(max_groq_bills: int = 50):
    """Runs only the classification stages (no ingest, no digest)."""
    print_header()
    start = time.time()

    run_stage_keyword()
    run_stage_embedding()
    run_stage_groq(max_bills=max_groq_bills)
    run_stage_openpaws(max_bills=max_groq_bills)
    run_stage_ensemble()

    elapsed = time.time() - start
    print(f"\n  ✅ Classification complete in {elapsed / 60:.1f} minutes\n")


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🐾 Paw-dvocate Legislative Intelligence Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --run-all              Full pipeline (ingest → classify → digest)
  python main.py --ingest               Re-ingest raw JSON data
  python main.py --classify             Run all classification stages
  python main.py --digest               Generate Markdown digest
  python main.py --email                Email the latest digest
  python main.py --stage groq           Run a single stage
  python main.py --run-all --email      Full pipeline + email digest
        """
    )

    # Top-level commands
    group = parser.add_argument_group("Pipeline Commands")
    group.add_argument("--run-all",   action="store_true", help="Run the full pipeline end-to-end")
    group.add_argument("--ingest",    action="store_true", help="Ingest raw LegiScan JSON data")
    group.add_argument("--classify",  action="store_true", help="Run all classification stages")
    group.add_argument("--digest",    action="store_true", help="Generate Markdown digest")
    group.add_argument("--email",     action="store_true", help="Email the latest digest")

    # Single stage
    group2 = parser.add_argument_group("Single Stage")
    group2.add_argument(
        "--stage",
        choices=["keyword", "embedding", "groq", "openpaws", "ensemble", "digest"],
        help="Run a single pipeline stage"
    )

    # Options
    group3 = parser.add_argument_group("Options")
    group3.add_argument("--max-bills", type=int, default=50,
                        help="Max bills for Groq/OpenPaws (default: 50)")
    group3.add_argument("--days-back", type=int, default=0,
                        help="Days of data for digest (0 = all time)")
    group3.add_argument("--email-to",  type=str, default="",
                        help="Override email recipient")

    args = parser.parse_args()

    # ── Determine what to run ──
    has_command = any([args.run_all, args.ingest, args.classify, args.digest,
                       args.email, args.stage])

    if not has_command:
        parser.print_help()
        return

    digest_path = ""

    # --run-all: full pipeline
    if args.run_all:
        digest_path = run_full_pipeline(
            max_groq_bills=args.max_bills,
            days_back=args.days_back,
        )

    # --ingest: data ingestion only
    if args.ingest and not args.run_all:
        print_header()
        run_stage_ingest()

    # --classify: classification only
    if args.classify and not args.run_all:
        run_classify_only(max_groq_bills=args.max_bills)

    # --digest: generate digest only
    if args.digest and not args.run_all:
        print_header()
        digest_path = run_stage_digest(days_back=args.days_back)

    # --stage: single stage
    if args.stage:
        print_header()
        if args.stage == "keyword":
            run_stage_keyword()
        elif args.stage == "embedding":
            run_stage_embedding()
        elif args.stage == "groq":
            run_stage_groq(max_bills=args.max_bills)
        elif args.stage == "openpaws":
            run_stage_openpaws(max_bills=args.max_bills)
        elif args.stage == "ensemble":
            run_stage_ensemble()
        elif args.stage == "digest":
            digest_path = run_stage_digest(days_back=args.days_back)

    # --email: send digest email (can combine with other commands)
    if args.email:
        run_stage_email(digest_path=digest_path or "", recipient=args.email_to)


if __name__ == "__main__":
    main()
