# tests/test_phase1.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from src.utils.db import (
    create_all_tables,
    get_connection,
    insert_bill,
    get_bill,
    save_classification,
    get_classification,
    save_embedding_record,
    save_digest_record,
    DB_PATH
)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def test_schema():
    print("\n" + "="*55)
    print("  PHASE 1 TEST — SQLite Schema")
    print("="*55)

    # ── Create tables ──
    print("\n[1/5] Creating tables...")
    create_all_tables()

    # ── Verify all 4 tables exist ──
    print("\n[2/5] Verifying tables exist...")
    conn = get_connection()
    tables = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """).fetchall()
    conn.close()

    expected = {"bills", "classifications", "embeddings", "digest_history"}
    found = {t["name"] for t in tables}

    for table in sorted(expected):
        icon = "✅" if table in found else "❌"
        print(f"  {icon} Table: {table}")

    # ── Insert a dummy bill ──
    print("\n[3/5] Testing bill insert...")
    dummy_bill = {
        "bill_id":          "CA_TEST_001",
        "state":            "CA",
        "bill_number":      "AB 100",
        "title":            "Animal Cruelty Prevention Act",
        "description":      "A bill to strengthen animal cruelty laws.",
        "status":           "Introduced",
        "status_date":      "2025-01-15",
        "introduced_date":  "2025-01-15",
        "last_action":      "Referred to committee",
        "last_action_date": "2025-01-20",
        "session":          "2025-2026",
        "session_year":     2025,
        "url":              "https://legiscan.com/CA/bill/AB100/2025",
        "sponsors":         "Jane Doe",
        "committee":        "Judiciary",
        "subjects":         "Animals, Welfare",
        "full_text":        "This bill amends section 597 of the Penal Code...",
        "source_file":      "data/raw/CA_AB100.json"
    }

    insert_bill(dummy_bill)
    fetched = get_bill("CA_TEST_001")

    if fetched and fetched["title"] == dummy_bill["title"]:
        print("  ✅ Bill inserted and retrieved correctly")
    else:
        print("  ❌ Bill insert/fetch failed")

    # ── Insert a dummy classification ──
    print("\n[4/5] Testing classification insert...")
    dummy_class = {
        "bill_id":                  "CA_TEST_001",
        "keyword_match":            1,
        "keywords_found":           "animal,cruelty,welfare",
        "keyword_score":            0.85,
        "embedding_similarity":     0.76,
        "embedding_label":          "pro-animal",
        "groq_label":               "pro-animal",
        "groq_confidence":          0.92,
        "groq_reasoning":           "Bill strengthens protections for animals.",
        "groq_classified_at":       utcnow(),
        "openpaws_alignment_score": 0.88,
        "openpaws_framing_summary": "Strong welfare alignment.",
        "openpaws_scored_at":       utcnow(),
        "final_label":              "pro-animal",
        "final_confidence":         0.90,
        "relevance_score":          0.87,
        "risk_level":               "low"
    }

    save_classification(dummy_class)
    fetched_class = get_classification("CA_TEST_001")

    if fetched_class and fetched_class["final_label"] == "pro-animal":
        print("  ✅ Classification saved and retrieved correctly")
    else:
        print("  ❌ Classification save/fetch failed")

    # ── Verify DB file on disk ──
    print("\n[5/5] Verifying database file on disk...")
    if os.path.exists(DB_PATH):
        size_kb = os.path.getsize(DB_PATH) / 1024
        print(f"  ✅ Database file found: {DB_PATH}")
        print(f"     Size: {size_kb:.1f} KB")
    else:
        print(f"  ❌ Database file not found at: {DB_PATH}")

    print("\n" + "="*55)
    print("  Phase 1 complete! Ready for Phase 2.")
    print("="*55 + "\n")


if __name__ == "__main__":
    test_schema()