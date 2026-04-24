# src/utils/db.py

import sqlite3
import os
from datetime import datetime, timezone


def utcnow():
    """Single helper — use this everywhere instead of utcnow()"""
    return datetime.now(timezone.utc).isoformat()


# Always store the database in data/db/
DB_PATH = os.path.join("data", "db", "legislation.db")


def get_connection():
    """
    Opens a connection to the SQLite database.
    Creates the file if it doesn't exist yet.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # This makes rows behave like dictionaries
    # so you can do row["bill_id"] instead of row[0]
    conn.row_factory = sqlite3.Row

    # Enables foreign key enforcement
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def create_all_tables():
    """
    Creates all 4 tables if they don't already exist.
    Safe to run multiple times — won't delete existing data.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ─────────────────────────────────────────
    # TABLE 1: bills
    # Core bill data from LegiScan JSON files
    # ─────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bills (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            bill_id             TEXT UNIQUE NOT NULL,
            state               TEXT NOT NULL,
            bill_number         TEXT NOT NULL,
            title               TEXT NOT NULL,
            description         TEXT,
            status              TEXT,
            status_date         TEXT,
            introduced_date     TEXT,
            last_action         TEXT,
            last_action_date    TEXT,
            session             TEXT,
            session_year        INTEGER,
            url                 TEXT,
            sponsors            TEXT,
            committee           TEXT,
            subjects            TEXT,
            full_text           TEXT,
            source_file         TEXT,
            ingested_at         TEXT NOT NULL,
            updated_at          TEXT NOT NULL
        )
    """)

    # ─────────────────────────────────────────
    # TABLE 2: classifications
    # Results from all 4 classifier stages
    # ─────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            bill_id                     TEXT NOT NULL,

            -- Stage 1: Keyword filter
            keyword_match               INTEGER DEFAULT 0,
            keywords_found              TEXT,
            keyword_score               REAL DEFAULT 0.0,

            -- Stage 2: Embedding similarity
            embedding_similarity        REAL DEFAULT 0.0,
            embedding_label             TEXT,

            -- Stage 3: Groq / LLM classifier
            groq_label                  TEXT,
            groq_confidence             REAL DEFAULT 0.0,
            groq_reasoning              TEXT,
            groq_classified_at          TEXT,

            -- Stage 4: Open Paws alignment
            openpaws_alignment_score    REAL DEFAULT 0.0,
            openpaws_framing_summary    TEXT,
            openpaws_scored_at          TEXT,

            -- Final combined output
            final_label                 TEXT,
            final_confidence            REAL DEFAULT 0.0,
            relevance_score             REAL DEFAULT 0.0,
            risk_level                  TEXT,

            classified_at               TEXT NOT NULL,
            updated_at                  TEXT NOT NULL,

            FOREIGN KEY (bill_id) REFERENCES bills(bill_id)
        )
    """)

    # ─────────────────────────────────────────
    # TABLE 3: embeddings
    # Stores sentence-transformer vectors
    # ─────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            bill_id         TEXT UNIQUE NOT NULL,
            model_name      TEXT NOT NULL,
            vector_file     TEXT NOT NULL,
            vector_dim      INTEGER NOT NULL,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,

            FOREIGN KEY (bill_id) REFERENCES bills(bill_id)
        )
    """)

    # ─────────────────────────────────────────
    # TABLE 4: digest_history
    # Tracks every weekly digest generated
    # ─────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS digest_history (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            digest_filename     TEXT NOT NULL,
            digest_filepath     TEXT NOT NULL,
            week_start          TEXT NOT NULL,
            week_end            TEXT NOT NULL,
            states_covered      TEXT NOT NULL,
            total_bills         INTEGER DEFAULT 0,
            new_bills           INTEGER DEFAULT 0,
            updated_bills       INTEGER DEFAULT 0,
            pro_animal_count    INTEGER DEFAULT 0,
            anti_animal_count   INTEGER DEFAULT 0,
            neutral_count       INTEGER DEFAULT 0,
            high_risk_count     INTEGER DEFAULT 0,
            generated_at        TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("✅ All 4 tables created successfully.")


# ─────────────────────────────────────────────────────
# HELPER FUNCTIONS
# Used by every phase that follows
# ─────────────────────────────────────────────────────

def insert_bill(bill_data: dict) -> bool:
    """
    Inserts a new bill or updates it if bill_id already exists.
    Returns True if inserted, False if updated.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = utcnow()

    cursor.execute(
        "SELECT id FROM bills WHERE bill_id = ?",
        (bill_data["bill_id"],)
    )
    exists = cursor.fetchone()

    if exists:
        cursor.execute("""
            UPDATE bills SET
                status           = :status,
                status_date      = :status_date,
                last_action      = :last_action,
                last_action_date = :last_action_date,
                updated_at       = :updated_at
            WHERE bill_id = :bill_id
        """, {**bill_data, "updated_at": now})
        conn.commit()
        conn.close()
        return False  # Updated
    else:
        cursor.execute("""
            INSERT INTO bills (
                bill_id, state, bill_number, title, description,
                status, status_date, introduced_date,
                last_action, last_action_date, session, session_year,
                url, sponsors, committee, subjects,
                full_text, source_file, ingested_at, updated_at
            ) VALUES (
                :bill_id, :state, :bill_number, :title, :description,
                :status, :status_date, :introduced_date,
                :last_action, :last_action_date, :session, :session_year,
                :url, :sponsors, :committee, :subjects,
                :full_text, :source_file, :ingested_at, :updated_at
            )
        """, {**bill_data, "ingested_at": now, "updated_at": now})
        conn.commit()
        conn.close()
        return True  # Inserted


def get_bill(bill_id: str) -> dict | None:
    """Fetch a single bill by its bill_id."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM bills WHERE bill_id = ?", (bill_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_bills(state: str = None) -> list[dict]:
    """
    Fetch all bills, optionally filtered by state.
    Usage: get_all_bills() or get_all_bills("CA")
    """
    conn = get_connection()
    if state:
        rows = conn.execute(
            "SELECT * FROM bills WHERE state = ?", (state,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM bills").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_classification(data: dict):
    """
    Insert or update a classification record for a bill.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = utcnow()

    cursor.execute(
        "SELECT id FROM classifications WHERE bill_id = ?",
        (data["bill_id"],)
    )
    exists = cursor.fetchone()

    if exists:
        cursor.execute("""
            UPDATE classifications SET
                keyword_match            = :keyword_match,
                keywords_found           = :keywords_found,
                keyword_score            = :keyword_score,
                embedding_similarity     = :embedding_similarity,
                embedding_label          = :embedding_label,
                groq_label               = :groq_label,
                groq_confidence          = :groq_confidence,
                groq_reasoning           = :groq_reasoning,
                groq_classified_at       = :groq_classified_at,
                openpaws_alignment_score = :openpaws_alignment_score,
                openpaws_framing_summary = :openpaws_framing_summary,
                openpaws_scored_at       = :openpaws_scored_at,
                final_label              = :final_label,
                final_confidence         = :final_confidence,
                relevance_score          = :relevance_score,
                risk_level               = :risk_level,
                updated_at               = :updated_at
            WHERE bill_id = :bill_id
        """, {**data, "updated_at": now})
    else:
        cursor.execute("""
            INSERT INTO classifications (
                bill_id,
                keyword_match, keywords_found, keyword_score,
                embedding_similarity, embedding_label,
                groq_label, groq_confidence, groq_reasoning,
                groq_classified_at,
                openpaws_alignment_score, openpaws_framing_summary,
                openpaws_scored_at,
                final_label, final_confidence, relevance_score,
                risk_level, classified_at, updated_at
            ) VALUES (
                :bill_id,
                :keyword_match, :keywords_found, :keyword_score,
                :embedding_similarity, :embedding_label,
                :groq_label, :groq_confidence, :groq_reasoning,
                :groq_classified_at,
                :openpaws_alignment_score, :openpaws_framing_summary,
                :openpaws_scored_at,
                :final_label, :final_confidence, :relevance_score,
                :risk_level, :classified_at, :updated_at
            )
        """, {**data, "classified_at": now, "updated_at": now})

    conn.commit()
    conn.close()


def get_classification(bill_id: str) -> dict | None:
    """Fetch classification record for a bill."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM classifications WHERE bill_id = ?",
        (bill_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_embedding_record(bill_id: str, model_name: str,
                           vector_file: str, vector_dim: int):
    """Log that an embedding was saved for this bill."""
    conn = get_connection()
    now = utcnow()
    conn.execute("""
        INSERT OR REPLACE INTO embeddings
            (bill_id, model_name, vector_file, vector_dim,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (bill_id, model_name, vector_file, vector_dim, now, now))
    conn.commit()
    conn.close()


def save_digest_record(record: dict):
    """Log a completed digest to digest_history."""
    conn = get_connection()
    now = utcnow()
    conn.execute("""
        INSERT INTO digest_history (
            digest_filename, digest_filepath,
            week_start, week_end, states_covered,
            total_bills, new_bills, updated_bills,
            pro_animal_count, anti_animal_count, neutral_count,
            high_risk_count, generated_at
        ) VALUES (
            :digest_filename, :digest_filepath,
            :week_start, :week_end, :states_covered,
            :total_bills, :new_bills, :updated_bills,
            :pro_animal_count, :anti_animal_count, :neutral_count,
            :high_risk_count, :generated_at
        )
    """, {**record, "generated_at": now})
    conn.commit()
    conn.close()