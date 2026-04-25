# tests/test_phase2.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.db import create_all_tables, get_all_bills, get_connection
from src.api.ingestor import ingest_all_json_files, print_summary


def test_ingestion():
    print("\n" + "=" * 55)
    print("  PHASE 2 TEST — JSON Ingestion")
    print("=" * 55)

    # Always make sure tables exist first
    print("\n[1/4] Ensuring database tables exist...")
    create_all_tables()

    # Run the ingestor
    print("\n[2/4] Ingesting JSON files from data/raw/...")
    counts = ingest_all_json_files()
    print_summary(counts)

    # Verify bills landed in the database
    print("\n[3/4] Verifying bills in database...")
    all_bills = get_all_bills()

    if not all_bills:
        print("  ⚠️  No bills found in database.")
        print("      Check that data/raw/ has .json files")
    else:
        print(f"  ✅ Total bills in DB: {len(all_bills)}")

        # Show breakdown by state
        from collections import Counter
        state_counts = Counter(b["state"] for b in all_bills)
        for state, count in sorted(state_counts.items()):
            print(f"     {state}: {count} bill(s)")

    # Preview first 3 bills
    print("\n[4/4] Preview of first 3 bills:")
    print("─" * 55)
    for bill in all_bills[:3]:
        print(f"  ID:        {bill['bill_id']}")
        print(f"  State:     {bill['state']}")
        print(f"  Number:    {bill['bill_number']}")
        print(f"  Title:     {bill['title'][:70]}")
        print(f"  Status:    {bill['status']}")
        print(f"  Sponsors:  {bill['sponsors'][:60]}")
        print(f"  Committee: {bill['committee']}")
        print(f"  Ingested:  {bill['ingested_at']}")
        print("─" * 55)

    print("\n  Phase 2 complete! Ready for Phase 3.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    test_ingestion()