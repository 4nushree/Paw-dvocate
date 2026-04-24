# src/api/ingestor.py

import os
import json
import glob
from datetime import datetime, timezone

from src.utils.db import insert_bill, get_bill
from config.settings import RAW_DATA_DIR, STATUS_MAP, MONITORED_STATES


def utcnow():
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────
# FIELD EXTRACTORS
# Each function pulls one piece of data safely.
# If the field is missing, it returns "" instead of
# crashing — important for messy real-world JSON.
# ─────────────────────────────────────────────────────

def extract_bill_id(bill: dict, filename: str) -> str:
    """
    Creates a unique ID like CA_1893344.
    Uses the state + numeric bill_id from JSON.
    """
    state   = bill.get("state", "XX")
    numeric = bill.get("bill_id", "0")
    return f"{state}_{numeric}"


def extract_status(bill: dict) -> str:
    """Converts numeric status code to human-readable text."""
    code = bill.get("status", 0)
    return STATUS_MAP.get(int(code), "Unknown")


def extract_introduced_date(bill: dict) -> str:
    """
    Introduced date = first event in progress array.
    Falls back to status_date if progress is empty.
    """
    progress = bill.get("progress", [])
    if progress:
        return progress[0].get("date", "")
    return bill.get("status_date", "")


def extract_last_action(bill: dict) -> tuple[str, str]:
    """
    Returns (last_action_text, last_action_date).
    Pulls from the last item in the history array.
    """
    history = bill.get("history", [])
    if history:
        last = history[-1]
        return last.get("action", ""), last.get("date", "")
    return "", ""


def extract_sponsors(bill: dict) -> str:
    """
    Joins all sponsor names into a comma-separated string.
    Example: "Damon Connolly, Dawn Addis, Benjamin Allen"
    """
    sponsors = bill.get("sponsors", [])
    names = [s.get("name", "") for s in sponsors if s.get("name")]
    return ", ".join(names)


def extract_committee(bill: dict) -> str:
    """
    Gets the most recent committee from referrals array.
    Falls back to checking the committee field directly.
    """
    referrals = bill.get("referrals", [])
    if referrals:
        # Last referral = most current committee
        return referrals[-1].get("name", "")
    
    # Fallback: direct committee field
    committee = bill.get("committee", [])
    if isinstance(committee, list) and committee:
        return committee[0].get("name", "")
    if isinstance(committee, str):
        return committee
    return ""


def extract_subjects(bill: dict) -> str:
    """
    Joins subject tags into a comma-separated string.
    Example: "Animals, Agriculture, Environment"
    """
    subjects = bill.get("subjects", [])
    if not subjects:
        return ""
    if isinstance(subjects[0], dict):
        return ", ".join(s.get("subject_name", "") for s in subjects)
    return ", ".join(str(s) for s in subjects)


def extract_session(bill: dict) -> tuple[str, int]:
    """
    Returns (session_name, session_year).
    Example: ("2025-2026 Regular Session", 2025)
    """
    session = bill.get("session", {})
    if isinstance(session, dict):
        name = session.get("session_name", "")
        year = session.get("year_start", 0)
        return name, int(year)
    return "", 0


def extract_text_url(bill: dict) -> str:
    """
    Pulls the URL of the most recent bill text version.
    LegiScan JSON has a 'texts' array with multiple versions.
    """
    texts = bill.get("texts", [])
    if texts:
        # Last entry = most recent version
        return texts[-1].get("url", "")
    return bill.get("url", "")


# ─────────────────────────────────────────────────────
# CORE PARSER
# Takes the raw dict from one JSON file and returns
# a clean dict ready to insert into SQLite.
# ─────────────────────────────────────────────────────

def parse_bill_json(raw: dict, source_file: str) -> dict | None:
    """
    Parses a LegiScan JSON file into a flat dict
    matching the bills table schema.

    Returns None if the file is malformed or
    the state is not in MONITORED_STATES.
    """
    # LegiScan wraps everything under a "bill" key
    if "bill" not in raw:
        print(f"  ⚠️  No 'bill' key found in {source_file} — skipping")
        return None

    bill = raw["bill"]

    # Only process states we care about
    state = bill.get("state", "")
    if state not in MONITORED_STATES:
        print(f"  ⚠️  State '{state}' not monitored — skipping {source_file}")
        return None

    # Extract all fields
    bill_id          = extract_bill_id(bill, source_file)
    status           = extract_status(bill)
    introduced_date  = extract_introduced_date(bill)
    last_action, last_action_date = extract_last_action(bill)
    sponsors         = extract_sponsors(bill)
    committee        = extract_committee(bill)
    subjects         = extract_subjects(bill)
    session_name, session_year = extract_session(bill)
    text_url         = extract_text_url(bill)

    return {
        "bill_id":          bill_id,
        "state":            state,
        "bill_number":      bill.get("bill_number", ""),
        "title":            bill.get("title", "No title"),
        "description":      bill.get("description", ""),
        "status":           status,
        "status_date":      bill.get("status_date", ""),
        "introduced_date":  introduced_date,
        "last_action":      last_action,
        "last_action_date": last_action_date,
        "session":          session_name,
        "session_year":     session_year,
        "url":              text_url,
        "sponsors":         sponsors,
        "committee":        committee,
        "subjects":         subjects,
        "full_text":        "",       # Text not in JSON, fetched later
        "source_file":      source_file,
    }


# ─────────────────────────────────────────────────────
# BATCH INGESTOR
# Scans data/raw/ for all .json files and processes them
# ─────────────────────────────────────────────────────

def ingest_all_json_files(raw_dir: str = RAW_DATA_DIR) -> dict:
    """
    Scans data/raw/ recursively for all .json files.
    Parses each one and inserts/updates into SQLite.
    Returns a summary dict with counts.
    """
    pattern = os.path.join(raw_dir, "**", "*.json")
    all_files = glob.glob(pattern, recursive=True)

    if not all_files:
        print(f"  ⚠️  No JSON files found in {raw_dir}/")
        return {"found": 0, "inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    total = len(all_files)
    print(f"\n  📂 Found {total:,} JSON file(s) in {raw_dir}/\n")

    counts = {
        "found":    total,
        "inserted": 0,
        "updated":  0,
        "skipped":  0,
        "errors":   0
    }

    for i, filepath in enumerate(sorted(all_files), 1):
        filename = os.path.relpath(filepath)

        # Progress counter — only prints every 100 files
        if i % 100 == 0 or i == 1 or i == total:
            print(f"  [{i:>6,}/{total:,}] Processing... "
                  f"✅{counts['inserted']} inserted  "
                  f"🔄{counts['updated']} updated  "
                  f"❌{counts['errors']} errors")

        # ── Load JSON ──
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            counts["errors"] += 1
            continue
        except Exception:
            counts["errors"] += 1
            continue

        # ── Parse fields ──
        try:
            bill_data = parse_bill_json(raw, filename)
        except Exception:
            counts["errors"] += 1
            continue

        if bill_data is None:
            counts["skipped"] += 1
            continue

        # ── Insert or update ──
        try:
            was_inserted = insert_bill(bill_data)
            if was_inserted:
                counts["inserted"] += 1
            else:
                counts["updated"] += 1
        except Exception:
            counts["errors"] += 1

    return counts


def print_summary(counts: dict):
    """Prints a clean summary after ingestion."""
    print("\n" + "─" * 50)
    print("  INGESTION SUMMARY")
    print("─" * 50)
    print(f"  📂 Files found:   {counts['found']:,}")
    print(f"  ✅ Inserted:      {counts['inserted']:,}")
    print(f"  🔄 Updated:       {counts['updated']:,}")
    print(f"  ⏭️  Skipped:       {counts['skipped']:,}")
    print(f"  ❌ Errors:        {counts['errors']:,}")
    print("─" * 50)


def print_summary(counts: dict):
    """Prints a clean summary after ingestion."""
    print("\n" + "─" * 50)
    print("  INGESTION SUMMARY")
    print("─" * 50)
    print(f"  📂 Files found:   {counts['found']:,}")
    print(f"  ✅ Inserted:      {counts['inserted']:,}")
    print(f"  🔄 Updated:       {counts['updated']:,}")
    print(f"  ⏭️  Skipped:       {counts['skipped']:,}")
    print(f"  ❌ Errors:        {counts['errors']:,}")
    print("─" * 50)