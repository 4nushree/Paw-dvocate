# 🐾 Paw-dvocate

> An applied AI legislative monitoring system that detects, classifies, and prioritises animal-related legislation across US state legislatures - built for advocacy organisations who can't afford to miss a policy window.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://paw-dvocate-jivv5kdgrryrvsydvuxbav.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SQLite](https://img.shields.io/badge/database-SQLite-lightgrey.svg)](https://www.sqlite.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3--70B-orange.svg)](https://groq.com/)

---

## Why Paw-dvocate Matters

Animal welfare organisations currently rely on listservs, manual legislative searches, and word of mouth to track relevant bills. By the time a harmful bill reaches the floor - or a protective bill dies in committee - the intervention window has already closed.

**Paw-dvocate automates early detection and prioritisation** of animal welfare legislation across three high-volume US state legislatures, giving advocates the intelligence layer they need to act when it matters.

---

## Live Demo

🌐 **Dashboard:** [paw-dvocate-jivv5kdgrryrvsydvuxbav.streamlit.app](https://paw-dvocate-jivv5kdgrryrvsydvuxbav.streamlit.app/)

📁 **Repository:** [github.com/4nushree/Paw-dvocate](https://github.com/4nushree/Paw-dvocate)

> **[Live Dashboard]**<img width="1919" height="911" alt="image" src="https://github.com/user-attachments/assets/92638379-f097-487e-9685-36e683d8c6f9" />


---

## Core Features

- Monitors **28,600+ bills** across California, Texas, and New York
- Detects animal-related legislation using a 4-layer AI classifier stack
- Classifies every bill as **pro-animal**, **anti-animal**, or **neutral**
- Scores each bill for **advocacy relevance** and **risk level**
- Generates **weekly Markdown intelligence digests**
- Exposes results through an **interactive Streamlit dashboard**
- Runs automatically on a daily schedule via **APScheduler**

---

## System Pipeline

```
LegiScan Bulk Dataset (JSON)
            ↓
    Ingestion Pipeline
            ↓
      SQLite Storage
            ↓
    Keyword Filter (Stage 1)
            ↓
  Embedding Similarity (Stage 2)
            ↓
  Groq LLM Classifier (Stage 3)
            ↓
 OpenPaws Alignment Scoring (Stage 4)
            ↓
    Ensemble Relevance Ranking
            ↓
     Weekly Digest Generator
            ↓
   Streamlit Advocacy Dashboard
```
> **[System Architecture]**<img width="1737" height="2949" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/eb154172-64fa-4b6e-b41d-ea1dc887e681" />


## Technical Architecture

### Classification Pipeline

Paw-dvocate uses a **4-stage layered classifier** designed to progressively reduce false positives while maximising recall of genuinely relevant bills.

**Stage 1 - Keyword Filtering**
Fast pattern matching against a curated dictionary of animal-related terms (`config/keywords.py`). Eliminates clearly irrelevant bills before any compute-heavy steps run. Returns a keyword density score.

**Stage 2 - Embedding Similarity Scoring**
Bills passing Stage 1 are encoded using `sentence-transformers` (`all-MiniLM-L6-v2`). Cosine similarity is computed against a set of reference animal welfare sentences. This catches semantically relevant bills that don't use explicit keywords.

**Stage 3 - Groq API Reasoning Classifier**
Bills above the similarity threshold are sent to **Llama-3.3-70B via Groq's free API tier** for zero-shot classification into `pro-animal`, `anti-animal`, or `neutral`. The LLM also returns a confidence score and a plain-language reasoning summary.

**Stage 4 - OpenPaws Alignment Scoring**
The [OpenPaws 8B Instruct](https://huggingface.co/open-paws/8B-instruct-chat) model scores each classified bill for ethical alignment with animal liberation principles and generates an advocacy framing summary for the digest.

**Ensemble Ranking**
All four signal layers are combined into a final `relevance_score` using weighted averaging. The ensemble weighting is: keyword (15%) + embedding (20%) + Groq confidence (45%) + OpenPaws alignment (20%).

---

### Database Architecture

Paw-dvocate uses a **single-file SQLite database** at `data/db/legislation.db`.

SQLite was chosen deliberately for three reasons: it requires zero server infrastructure, it deploys directly to Streamlit Cloud as a file, and it handles 28,000+ records well within this read-heavy workload.

**Schema - 4 tables:**

| Table | Purpose |
|---|---|
| `bills` | Core bill metadata: title, state, sponsors, committee, status, session |
| `classifications` | All 4 classifier outputs + final label, confidence, relevance score, risk level |
| `embeddings` | Embedding vector file paths, model name, vector dimension |
| `digest_history` | Record of every weekly digest generated with counts |

---

### Data Source Strategy

LegiScan API approval was pending during development. Bulk session JSON exports were downloaded directly from the LegiScan website for California, Texas, and New York (2025–2026 session).

The ingestion pipeline (`src/api/ingestor.py`) is designed to parse LegiScan's standard JSON schema, meaning **it will switch to live API ingestion without any redesign** once an API key is available - only the data source path changes.

---

## Project Structure

```
pawdvocate/
│
├── config/
│   ├── settings.py           ← API keys, state codes, paths, model config
│   └── keywords.py           ← Animal keyword dictionaries by category
│
├── data/
│   ├── raw/                  ← LegiScan JSON exports (CA / TX / NY)
│   ├── db/
│   │   └── legislation.db    ← SQLite database (28,600+ bills)
│   └── embeddings/           ← Cached sentence-transformer vectors (.npy)
│
├── src/
│   ├── api/
│   │   └── ingestor.py       ← JSON parser + batch ingestion pipeline
│   │
│   ├── classifier/
│   │   ├── keyword_filter.py ← Stage 1: keyword density scoring
│   │   ├── embedder.py       ← Stage 2: sentence-transformer encoding
│   │   ├── similarity.py     ← Stage 2: cosine similarity scoring
│   │   ├── groq_classifier.py← Stage 3: Llama-3.3-70B classification
│   │   ├── alignment.py      ← Stage 4: OpenPaws alignment scoring
│   │   └── ensemble.py       ← Weighted ensemble + final label
│   │
│   ├── digest/
│   │   └── generator.py      ← Weekly Markdown digest generator
│   │
│   ├── scheduler/
│   │   └── scheduler.py      ← APScheduler automation
│   │
│   └── utils/
│       └── db.py             ← SQLite helpers (CRUD, connection management)
│
├── frontend/
│   └── app.py                ← Streamlit dashboard
│
├── tests/
│   ├── test_phase1.py        ← Schema validation
│   ├── test_phase2.py        ← Ingestion validation
│   └── evaluation_phase9.py  ← Accuracy benchmark (≥85% target)
│
├── digests/                  ← Weekly .md digest output files
├── logs/                     ← Pipeline logs + evaluation reports
├── pipeline.py               ← Full pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Phase-by-Phase Development

Paw-dvocate was built incrementally in 11 phases, each validated before the next began.

**Phase 1 - SQLite Schema + Project Scaffolding**
Designed the 4-table database schema and all CRUD helper functions. Established the project folder structure and logging utilities.

**Phase 2 - JSON Ingestion Pipeline**
Built the LegiScan JSON parser handling real-world schema variations (nested sponsors, referral arrays, status codes). Ingested 51,858 JSON files with progress tracking, inserting 28,604 bills across CA, TX, and NY.

**Phase 3 - Keyword Classifier**
Built a weighted keyword dictionary (`config/keywords.py`) with 200+ terms across categories: species, legislation type, welfare practices, and exploitation industries. Returns a keyword density score per bill.

**Phase 4 - Embedding Similarity Scoring**
Integrated `sentence-transformers` (`all-MiniLM-L6-v2`) to encode bill text. Vectors cached as `.npy` files in `data/embeddings/` to avoid recomputation. Bills scored by cosine similarity against curated reference sentences.

**Phase 5 - Semantic Relevance Detection**
Built the similarity threshold logic to identify which embedding-scored bills warrant LLM classification. Tuned thresholds using precision/recall trade-off analysis on a sample set.

**Phase 6 - Groq LLM Classifier Integration**
Integrated the Groq API (`llama-3.3-70b-versatile`) as the primary reasoning classifier. Returns classification label, confidence score (0–1), and plain-language reasoning stored in the `classifications` table.

**Phase 7 - OpenPaws Alignment Scoring + Ensemble Ranking**
Integrated the OpenPaws 8B Instruct model for ethical alignment scoring. Built the weighted ensemble that combines all four signal layers into a final `relevance_score` and `risk_level`.

**Phase 8 - Markdown Digest Generator + Pipeline Orchestrator**
Built the weekly digest generator producing structured Markdown reports covering new bills, updated bills, high-risk alerts, and welfare opportunities. Built `pipeline.py` as the single-command orchestrator.

**Phase 9 - Evaluation Benchmark**
Built `tests/evaluation_phase9.py` to validate classification accuracy against manually labelled samples. Target: ≥85% accuracy. Results saved to `logs/evaluation_report.txt`.

**Phase 10 - Scheduler Automation**
Integrated APScheduler to run the full pipeline daily at 02:00 and generate digests weekly on Sundays at 09:00.

**Phase 11 - Streamlit Dashboard**
Built the interactive advocacy dashboard with state filtering, risk-level grouping, and per-bill reasoning display.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/4nushree/Paw-dvocate.git
cd Paw-dvocate
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com). No credit card required.

### 5. Add your data

Place LegiScan JSON exports in `data/raw/` following the structure:

```
data/raw/CA/2025-2026_Regular_Session/bill/*.json
data/raw/TX/2025-2026_Regular_Session/bill/*.json
data/raw/NY/2025-2026_General_Assembly/bill/*.json
```

### 6. Run the full pipeline

```bash
python pipeline.py --run-all
```

Or run individual stages:

```bash
python pipeline.py --ingest      # Ingest JSON files only
python pipeline.py --classify    # Run classifier stack only
python pipeline.py --digest      # Generate weekly digest only
```

### 7. Launch the dashboard

```bash
streamlit run frontend/app.py
```

### 8. Run the scheduler (optional)

```bash
python src/scheduler/scheduler.py
```

---

## Deployment

### Streamlit Cloud

Paw-dvocate's frontend is deployed on Streamlit Cloud connected to this GitHub repository.

The SQLite database snapshot (`data/db/legislation.db`) is committed to the repository and read directly by the dashboard - no server infrastructure required.

**To deploy your own instance:**

1. Push this repository to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app** → select your repository
4. Set **Main file path** to `frontend/app.py`
5. Add your `GROQ_API_KEY` in the **Secrets** manager
6. Deploy

### Scheduler Deployment (optional)

To run the pipeline on a schedule in the cloud:

- **GitHub Actions** - add a cron workflow to trigger `pipeline.py` daily
- **Railway / Render / Fly.io** - deploy `src/scheduler/scheduler.py` as a background worker

---

## Evaluation Methodology

Classification accuracy is validated using `tests/evaluation_phase9.py`:

1. 20 bills are sampled randomly from the `classifications` table
2. Bill IDs are exported to `tests/manual_labels.csv` for human labelling
3. The script compares `true_label` (manual) against `final_label` (predicted)
4. Outputs accuracy, precision, recall, F1-score, and a confusion matrix
5. Results saved to `logs/evaluation_report.txt`

**Target: ≥85% classification accuracy**

The ensemble scoring approach reduces false positives by requiring agreement across multiple signal layers before assigning high-confidence labels.

> **[Classifier Output Example Placeholder]**

---

## Sample Digest Output

> **[Digest Output Screenshot Placeholder]**

Weekly digests are saved to `digests/` as Markdown files named `digest_YYYY-MM-DD.md`. Each digest contains:

- **New Bills** - bills ingested since the last digest
- **Updated Bills** - bills with status changes
- **High-Risk Anti-Animal Alerts** - bills classified anti-animal with high relevance score
- **High-Impact Welfare Opportunities** - pro-animal bills in active committee stages

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Database | SQLite |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM Classifier | Groq API - Llama-3.3-70B (free tier) |
| Alignment Scoring | OpenPaws 8B Instruct (HuggingFace) |
| Scheduler | APScheduler |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud + GitHub |

---

## Limitations

- **Bulk JSON used instead of live API** - LegiScan API approval was pending during development. The pipeline is API-ready but currently runs on downloaded session exports.
- **Sponsor-history weighting partially implemented** - sponsor voting history signals are extracted but not yet used as a weighted feature in the ensemble.
- **Committee influence weighting is basic** - committee prestige scoring uses a simple lookup rather than a learned model.
- **OpenPaws 8B requires 16GB+ RAM locally** - the alignment layer falls back to the HuggingFace Inference API free tier when running on limited hardware.

---

## Future Improvements

- **Real-time LegiScan API ingestion** - replace bulk exports with live API polling as sessions update
- **Email digest alerts** - convert weekly Markdown digests to HTML and deliver via SMTP
- **Multi-state comparison dashboard** - side-by-side view of how the same issue tracks across CA, TX, and NY simultaneously
- **Sponsor-history learning model** - train a lightweight model on sponsor voting history to improve pro/anti signal quality
- **Committee influence modelling** - weight bills by committee composition and historical passage rates
- **PostgreSQL migration** - replace SQLite with PostgreSQL for production multi-user deployments
- **Docker containerisation** - package the full pipeline for reproducible deployment

---

## Documentation

The `documentation/` directory contains phase-by-phase explanations of system architecture decisions, classifier design rationale, and evaluation methodology.

---

## Acknowledgements

- [LegiScan](https://legiscan.com/) for legislative data
- [Open Paws](https://openpaws.ai/) for the animal advocacy alignment model
- [Groq](https://groq.com/) for free LLM inference
- [Streamlit](https://streamlit.io/) for the dashboard framework

---

*Built for the animal advocacy community. Every bill that harms animals passes because no one was watching. Paw-dvocate watches.*
