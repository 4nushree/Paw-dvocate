# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

# ── States we monitor ──
MONITORED_STATES = ["CA", "TX", "NY"]

# ── LegiScan status codes → human readable ──
STATUS_MAP = {
    0: "NA",
    1: "Introduced",
    2: "Engrossed",
    3: "Enrolled",
    4: "Passed",
    5: "Vetoed",
    6: "Failed",
    7: "Override",
    8: "Chaptered",
    9: "Refer",
    10: "Report Pass",
    11: "Report Fail",
    12: "Draft"
}

# ── Paths ──
RAW_DATA_DIR    = os.path.join("data", "raw")
DB_PATH         = os.path.join("data", "db", "legislation.db")
EMBEDDINGS_DIR  = os.path.join("data", "embeddings")
DIGESTS_DIR     = "digests"
LOGS_DIR        = "logs"

# ── Groq API ──
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = "llama-3.3-70b-versatile"

# ── Embedding model ──
EMBEDDING_MODEL = "all-MiniLM-L6-v2"