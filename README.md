# 🐾 Paw-dvocate — Animal Legislation Intelligence Pipeline

An automated AI-powered system that monitors, classifies, and generates intelligence digests for animal-related legislation across **California**, **Texas**, and **New York**.

Built with Python, SQLite, sentence-transformers, Groq LLM, and Streamlit. **100% free tools.**

---

## 📖 Project Overview

Paw-dvocate is an applied AI pipeline designed to automate the discovery and analysis of animal-related legislation across United States state legislatures. By bridging the gap between raw policy data and actionable insights, the system ensures that advocacy organizations can influence the legislative process before critical deadlines pass.

### The Problem: The "Awareness Gap"

Currently, many advocacy groups rely on manual web searches, word-of-mouth, or high-traffic listservs to find relevant bills. This fragmented approach often leads to "late discovery"—finding out about a bill only after the public comment window has closed or a crucial vote has already taken place. In the fast-moving environment of state politics, missing even a 48-hour window can mean the difference between a policy's success or failure.

### Who Benefits?

- **Advocacy Organizations**: Gain a "first-look" advantage to mobilize supporters early.
- **Policy Researchers**: Save hundreds of hours of manual filtering by accessing pre-categorized bill sets.
- **Engineering Interns**: Access a modular, real-world example of how to apply Natural Language Processing (NLP) to civic tech.

### Why Paw-dvocate?

Manual monitoring is often reactive and inconsistent, relying on human effort to scan thousands of pages of text. Paw-dvocate is proactive and persistent; it doesn't just look for keywords—it understands intent. By leveraging semantic analysis, it can distinguish between a bill that mentions "animals" in a passing agricultural context and one that significantly impacts welfare standards. This precision moves advocacy from a reactive stance to a proactive strategy.

For detailed documentation, see [docs/project_overview.md](docs/project_overview.md).

---

## 📊 What It Does

1. **Ingests** 28,000+ bills from LegiScan bulk exports (CA, TX, NY)
2. **Filters** candidates using tiered keyword matching (strong/moderate/weak)
3. **Scores** semantic similarity using `all-MiniLM-L6-v2` embeddings
4. **Classifies** using Groq's Llama 3.3 70B with structured reasoning
5. **Aligns** against Open Paws advocacy framing (-1 to +1 scale)
6. **Combines** all signals via weighted ensemble (85%+ accuracy target)
7. **Generates** weekly Markdown intelligence digests
8. **Emails** digests automatically (optional Gmail SMTP)
9. **Visualizes** everything in a Streamlit dashboard

---

## 🗂️ Project Structure

```
legislation_monitor/
│
├── main.py                       # Pipeline CLI orchestrator
├── .env                          # API keys (GROQ_API_KEY, etc.)
│
├── config/
│   ├── settings.py               # Paths, models, weights
│   └── keywords.py               # Tiered keyword lists
│
├── data/
│   ├── raw/                      # LegiScan JSON exports
│   │   ├── CA/                   # California session data
│   │   ├── TX/                   # Texas session data
│   │   └── NY/                   # New York session data
│   ├── db/
│   │   └── legislation.db        # SQLite database
│   └── embeddings/               # Cached .npy centroid files
│
├── src/
│   ├── api/
│   │   └── ingestor.py           # JSON → SQLite ingestion
│   ├── classifier/
│   │   ├── keyword_filter.py     # Phase 3: Keyword matching
│   │   ├── embedding_scorer.py   # Phase 4: Semantic similarity
│   │   ├── groq_classifier.py    # Phase 5/6: LLM classification
│   │   ├── openpaws_scorer.py    # Phase 7: Alignment scoring
│   │   └── ensemble.py           # Phase 7: Weighted ensemble
│   ├── digest/
│   │   ├── generator.py          # Phase 8: Markdown digest
│   │   └── email_sender.py       # Phase 12: Email export
│   ├── scheduler/
│   │   └── scheduler.py          # Phase 10: APScheduler automation
│   └── utils/
│       └── db.py                 # SQLite CRUD helpers
│
├── frontend/
│   └── app.py                    # Phase 11: Streamlit dashboard
│
├── tests/
│   ├── test_phase3.py            # Keyword filter tests
│   ├── test_phase4.py            # Embedding scorer tests
│   ├── test_phase5.py            # Groq classifier tests
│   ├── test_phase7.py            # Ensemble tests
│   ├── test_phase8.py            # Digest generator tests
│   ├── evaluation_phase9.py      # Accuracy benchmark
│   └── manual_labels.csv         # Human labels for evaluation
│
├── digests/                      # Generated Markdown reports
└── logs/                         # Scheduler + evaluation logs
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/) (required)

### 2. Clone & Install

```bash
git clone https://github.com/your-repo/legislation_monitor.git
cd legislation_monitor

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
# Required — Groq free tier
GROQ_API_KEY=gsk_your_groq_api_key_here

# Optional — HuggingFace (for Open Paws primary backend)
HF_TOKEN=hf_your_token_here

# Optional — Email digest (Gmail)
EMAIL_FROM=yourname@gmail.com
EMAIL_PASSWORD=xxxx xxxx xxxx xxxx
EMAIL_TO=recipient@example.com
```

> **Gmail Setup:** Enable 2FA → go to [App Passwords](https://myaccount.google.com/apppasswords) → create one for "Mail".

### 4. Add LegiScan Data

Place bulk JSON exports in:
```
data/raw/CA/   ← California session folder
data/raw/TX/   ← Texas session folder
data/raw/NY/   ← New York session folder
```

Each should contain: `bill.json`, `people.json`, `committee.json`, etc.

### 5. Run the Pipeline

```bash
# Full pipeline (ingest → classify → digest)
python main.py --run-all

# First run? Start with fewer bills to test:
python main.py --run-all --max-bills 20
```

---

## 🛠️ Pipeline Commands

### Full Pipeline

```bash
python main.py --run-all                    # Everything: ingest → classify → digest
python main.py --run-all --email            # Full pipeline + email digest
python main.py --run-all --max-bills 100    # Limit Groq/OpenPaws to 100 bills
```

### Individual Operations

```bash
python main.py --ingest                     # Re-ingest raw JSON data
python main.py --classify                   # Run all classification stages
python main.py --digest                     # Generate digest from existing data
python main.py --digest --days-back 7       # Digest for last 7 days only
python main.py --email                      # Email the latest digest
python main.py --email --email-to a@b.com   # Email to specific address
```

### Single Stage

```bash
python main.py --stage keyword              # Keyword filter only
python main.py --stage embedding            # Embedding scorer only
python main.py --stage groq --max-bills 50  # Groq LLM on 50 bills
python main.py --stage openpaws             # Open Paws alignment
python main.py --stage ensemble             # Weighted ensemble
python main.py --stage digest               # Digest generation
```

---

## 📊 Streamlit Dashboard

```bash
streamlit run frontend/app.py
```

Opens at `http://localhost:8501` with:

- **State selector** (CA / TX / NY / All)
- **Risk/label/confidence filters**
- **Anti-animal alerts** (high-risk bills)
- **Pro-animal opportunities** (advocacy targets)
- **Recent movement** (bills in progress)
- **Distribution charts** by state and risk
- **Full-text search** across all bills
- **Raw data table** (expandable)

---

## ⏰ Scheduler (Automation)

```bash
# Start scheduler (runs forever, Ctrl+C to stop)
python src/scheduler/scheduler.py

# Run pipeline now, then start scheduler
python src/scheduler/scheduler.py --run-now

# Test both jobs once, then exit
python src/scheduler/scheduler.py --test
```

**Schedule:**
| Job | When | What |
|-----|------|------|
| Daily Pipeline | Every day at 02:00 | keyword → embedding → groq → openpaws → ensemble |
| Weekly Digest | Every Sunday at 09:00 | Generate Markdown intelligence report |

Logs saved to `logs/scheduler.log`.

---

## 🧪 Testing & Evaluation

### Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific phase
python -m pytest tests/test_phase3.py -v
```

### Accuracy Benchmark (Phase 9)

```bash
# Step 1: Generate sample CSV (20 bills)
python tests/evaluation_phase9.py --generate

# Step 2: Open tests/manual_labels.csv, fill 'human_label' column:
#   pro_animal  |  anti_animal  |  neutral

# Step 3: Compute metrics
python tests/evaluation_phase9.py --evaluate

# View saved report
python tests/evaluation_phase9.py --show
```

**Target:** ≥ 85% classification accuracy

---

## 🔧 Architecture

### Pipeline Stages & Weights

```
   LegiScan JSON → Ingest → SQLite
                      ↓
   ┌─────────────────────────────────┐
   │  Stage 1: Keyword Filter (15%) │  Fast: ~1 min
   │  Stage 2: Embedding (20%)      │  Medium: ~7 min
   │  Stage 3: Groq LLM (45%)      │  Slow: rate-limited
   │  Stage 4: Open Paws (20%)     │  Slow: rate-limited
   └─────────────────────────────────┘
                      ↓
              Weighted Ensemble
                      ↓
         Markdown Digest → Email
                      ↓
           Streamlit Dashboard
```

### Ensemble Weights

| Signal | Weight | What it measures |
|--------|--------|------------------|
| Keyword Filter | 15% | Tiered keyword matching (strong/moderate/weak) |
| Semantic Embedding | 20% | `all-MiniLM-L6-v2` cosine similarity to reference centroids |
| Groq LLM | 45% | Llama 3.3 70B structured reasoning |
| Open Paws Alignment | 20% | Framing analysis: pro-animal ↔ anti-animal scale |

### Risk Levels

| Level | Criteria |
|-------|----------|
| 🔴 High | Non-neutral + ≥60% confidence |
| 🟡 Medium | Non-neutral + ≥30% confidence |
| 🟢 Low | Neutral or < 30% confidence |

### Database Tables

| Table | Purpose |
|-------|---------|
| `bills` | Raw bill metadata (title, status, sponsors, etc.) |
| `classifications` | All pipeline signals + final ensemble results |
| `digest_history` | Log of generated digests |

---

## 📧 Email Setup (Optional)

1. **Enable 2-Factor Auth** on your Gmail
2. Go to [App Passwords](https://myaccount.google.com/apppasswords)
3. Generate a password for "Mail"
4. Add to `.env`:
   ```
   EMAIL_FROM=you@gmail.com
   EMAIL_PASSWORD=xxxx xxxx xxxx xxxx
   EMAIL_TO=team@example.com
   ```
5. Test: `python main.py --email`

**Preview without sending:**
```bash
python -m src.digest.email_sender --preview
# Saves HTML file next to the digest for browser preview
```

---

## 📋 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Free Groq API key |
| `HF_TOKEN` | ❌ No | HuggingFace token (falls back to Groq) |
| `EMAIL_FROM` | ❌ No | Gmail sender address |
| `EMAIL_PASSWORD` | ❌ No | Gmail App Password |
| `EMAIL_TO` | ❌ No | Default email recipient |

---

## 🛡️ Design Decisions

- **Cascading filters**: Low-cost stages (keyword/embedding) run first, reducing expensive LLM calls
- **Resume-safe**: Every API call is immediately persisted; restarts skip completed work
- **Rate-limit aware**: 2.5s delay between Groq calls, exponential backoff on failures
- **No ORM**: Direct SQLite for simplicity and zero dependencies
- **Weighted ensemble**: Reduces single-model volatility by combining 4 independent signals
- **Offline-first**: Works with LegiScan bulk exports, no live API polling required

---

## 📝 License

MIT

---

*Built with 🐾 for animal welfare advocacy*