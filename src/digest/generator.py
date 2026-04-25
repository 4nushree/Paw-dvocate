# src/digest/generator.py
#
# Phase 8a — Weekly Markdown Digest Generator
#
# Queries the classifications table for the latest results
# and produces a formatted Markdown intelligence report
# saved to the digests/ directory.

import os
from datetime import datetime, timezone, timedelta

from config.settings import DIGESTS_DIR, MONITORED_STATES
from src.utils.db import get_connection, utcnow, save_digest_record


# ─────────────────────────────────────────────────────
# QUERY: Pull classified bills for the digest
# ─────────────────────────────────────────────────────

def get_digest_bills(
    days_back: int = 7,
    states: list = None,
) -> list[dict]:
    """
    Pulls bills that have a final classification,
    optionally filtered to a recent time window.

    Parameters:
        days_back (int):  How many days back to look (0 = all time)
        states (list):    Filter by state codes (None = all monitored)

    Returns:
        list[dict]: Bills joined with their classification data
    """
    conn = get_connection()

    query = """
        SELECT
            b.bill_id, b.state, b.bill_number, b.title,
            b.description, b.status, b.status_date,
            b.introduced_date, b.last_action, b.last_action_date,
            b.url, b.sponsors, b.committee, b.subjects,
            c.keyword_score, c.keyword_match, c.keywords_found,
            c.embedding_similarity, c.embedding_label,
            c.groq_label, c.groq_confidence, c.groq_reasoning,
            c.openpaws_alignment_score, c.openpaws_framing_summary,
            c.final_label, c.final_confidence, c.relevance_score,
            c.risk_level
        FROM bills b
        INNER JOIN classifications c ON b.bill_id = c.bill_id
        WHERE c.final_label IS NOT NULL
          AND c.final_label != ''
    """
    params = []

    if days_back > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        query += " AND c.updated_at >= ?"
        params.append(cutoff)

    if states:
        placeholders = ",".join(["?"] * len(states))
        query += f" AND b.state IN ({placeholders})"
        params.extend(states)

    query += " ORDER BY c.relevance_score DESC, c.final_confidence DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────
# MARKDOWN GENERATION
# ─────────────────────────────────────────────────────

def _risk_emoji(level: str) -> str:
    return {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(level, "⚪")


def _label_emoji(label: str) -> str:
    return {
        "pro_animal": "🐾",
        "anti_animal": "⚠️",
        "neutral": "➖",
    }.get(label, "❓")


def _label_display(label: str) -> str:
    return {
        "pro_animal": "Pro-Animal",
        "anti_animal": "Anti-Animal",
        "neutral": "Neutral",
    }.get(label, "Unknown")


def generate_digest_markdown(
    bills: list[dict],
    week_start: str = "",
    week_end: str = "",
    states: list = None,
) -> str:
    """
    Generates a formatted Markdown digest from classified bills.

    Returns the full Markdown string.
    """
    if not states:
        states = MONITORED_STATES

    now = datetime.now(timezone.utc)
    if not week_start:
        week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    if not week_end:
        week_end = now.strftime("%Y-%m-%d")

    # ── Compute stats ──
    total = len(bills)
    pro   = [b for b in bills if b.get("final_label") == "pro_animal"]
    anti  = [b for b in bills if b.get("final_label") == "anti_animal"]
    neut  = [b for b in bills if b.get("final_label") == "neutral"]
    high_risk  = [b for b in bills if b.get("risk_level") == "high"]
    med_risk   = [b for b in bills if b.get("risk_level") == "medium"]

    # State breakdown
    state_counts = {}
    for b in bills:
        s = b.get("state", "??")
        state_counts[s] = state_counts.get(s, 0) + 1

    # ── Build Markdown ──
    lines = []

    # Header
    lines.append(f"# 🐾 Paw-dvocate Weekly Intelligence Digest")
    lines.append(f"")
    lines.append(f"**Period:** {week_start} → {week_end}  ")
    lines.append(f"**States:** {', '.join(states)}  ")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M UTC')}  ")
    lines.append(f"")

    # ── Executive Summary ──
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 📊 Executive Summary")
    lines.append(f"")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total classified bills | **{total}** |")
    lines.append(f"| 🐾 Pro-animal | **{len(pro)}** |")
    lines.append(f"| ⚠️ Anti-animal | **{len(anti)}** |")
    lines.append(f"| ➖ Neutral | **{len(neut)}** |")
    lines.append(f"| 🔴 High risk | **{len(high_risk)}** |")
    lines.append(f"| 🟡 Medium risk | **{len(med_risk)}** |")
    lines.append(f"")

    # State breakdown
    lines.append(f"### By State")
    lines.append(f"")
    lines.append(f"| State | Bills |")
    lines.append(f"|-------|-------|")
    for s in sorted(state_counts.keys()):
        lines.append(f"| {s} | {state_counts[s]} |")
    lines.append(f"")

    # ── HIGH PRIORITY: Bills needing advocacy attention ──
    if high_risk:
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## 🔴 High Priority — Needs Immediate Attention")
        lines.append(f"")
        for b in sorted(high_risk, key=lambda x: x.get("final_confidence", 0), reverse=True):
            _append_bill_card(lines, b)

    # ── PRO-ANIMAL BILLS ──
    pro_only = [b for b in pro if b.get("risk_level") != "high"]
    if pro_only:
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## 🐾 Pro-Animal Bills")
        lines.append(f"")
        for b in sorted(pro_only, key=lambda x: x.get("final_confidence", 0), reverse=True)[:20]:
            _append_bill_card(lines, b)

    # ── ANTI-ANIMAL BILLS ──
    anti_only = [b for b in anti if b.get("risk_level") != "high"]
    if anti_only:
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## ⚠️ Anti-Animal Bills")
        lines.append(f"")
        for b in sorted(anti_only, key=lambda x: x.get("final_confidence", 0), reverse=True)[:20]:
            _append_bill_card(lines, b)

    # ── METHODOLOGY ──
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 📐 Methodology")
    lines.append(f"")
    lines.append(f"Bills were classified using a 4-stage weighted ensemble pipeline:")
    lines.append(f"")
    lines.append(f"| Stage | Weight | Method |")
    lines.append(f"|-------|--------|--------|")
    lines.append(f"| Keyword Filter | 15% | Tiered keyword matching (strong/moderate/weak) |")
    lines.append(f"| Semantic Embedding | 20% | all-MiniLM-L6-v2 cosine similarity to reference centroids |")
    lines.append(f"| LLM Classification | 45% | Groq Llama 3.3 70B reasoning-based classification |")
    lines.append(f"| Alignment Scoring | 20% | Open Paws framing analysis (-1 to +1 alignment) |")
    lines.append(f"")
    lines.append(f"**Risk levels:**  ")
    lines.append(f"- 🔴 **High** — Non-neutral bill with ≥60% ensemble confidence  ")
    lines.append(f"- 🟡 **Medium** — Non-neutral bill with ≥30% confidence  ")
    lines.append(f"- 🟢 **Low** — Neutral or low-confidence classification  ")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"*Generated by Paw-dvocate Legislative Intelligence Pipeline*")

    return "\n".join(lines)


def _append_bill_card(lines: list, b: dict):
    """Appends a single bill card to the markdown lines."""
    label   = _label_display(b.get("final_label", ""))
    emoji   = _label_emoji(b.get("final_label", ""))
    risk    = _risk_emoji(b.get("risk_level", ""))
    conf    = b.get("final_confidence", 0.0)
    rel     = b.get("relevance_score", 0.0)
    state   = b.get("state", "")
    number  = b.get("bill_number", "")
    title   = b.get("title", "")
    status  = b.get("status", "")
    url     = b.get("url", "")
    sponsors = b.get("sponsors", "") or ""
    reasoning = b.get("groq_reasoning", "") or ""
    framing   = b.get("openpaws_framing_summary", "") or ""
    alignment = b.get("openpaws_alignment_score", 0.0) or 0.0

    lines.append(f"### {risk} {emoji} {state} {number} — {title[:80]}")
    lines.append(f"")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| **Classification** | {label} |")
    lines.append(f"| **Confidence** | {conf:.0%} |")
    lines.append(f"| **Relevance** | {rel:.0%} |")
    lines.append(f"| **Alignment Score** | {alignment:+.2f} |")
    lines.append(f"| **Status** | {status} |")
    if sponsors:
        lines.append(f"| **Sponsors** | {sponsors[:80]} |")
    if url:
        lines.append(f"| **Link** | [{number}]({url}) |")
    lines.append(f"")

    if reasoning:
        lines.append(f"> **AI Analysis:** {reasoning[:200]}")
        lines.append(f"")
    if framing:
        lines.append(f"> **Framing:** {framing[:200]}")
        lines.append(f"")
    lines.append(f"")


# ─────────────────────────────────────────────────────
# SAVE DIGEST TO DISK + DB
# ─────────────────────────────────────────────────────

def save_digest(
    markdown: str,
    bills: list[dict],
    week_start: str,
    week_end: str,
    states: list,
) -> str:
    """
    Saves the digest Markdown to digests/ directory
    and records it in the digest_history table.

    Returns the file path.
    """
    os.makedirs(DIGESTS_DIR, exist_ok=True)

    now = datetime.now(timezone.utc)
    filename = f"digest_{now.strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(DIGESTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown)

    # Count labels
    pro   = sum(1 for b in bills if b.get("final_label") == "pro_animal")
    anti  = sum(1 for b in bills if b.get("final_label") == "anti_animal")
    neut  = sum(1 for b in bills if b.get("final_label") == "neutral")
    high  = sum(1 for b in bills if b.get("risk_level") == "high")

    save_digest_record({
        "digest_filename":  filename,
        "digest_filepath":  filepath,
        "week_start":       week_start,
        "week_end":         week_end,
        "states_covered":   ",".join(states),
        "total_bills":      len(bills),
        "new_bills":        len(bills),
        "updated_bills":    0,
        "pro_animal_count": pro,
        "anti_animal_count": anti,
        "neutral_count":    neut,
        "high_risk_count":  high,
    })

    return filepath


# ─────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────

def generate_weekly_digest(
    days_back: int = 7,
    states: list = None,
    verbose: bool = True,
) -> str:
    """
    Full digest generation: query → format → save → return path.

    Parameters:
        days_back (int):   Days of data to include (0 = all time)
        states (list):     State filter (None = all monitored)
        verbose (bool):    Print progress

    Returns:
        str: Path to the saved digest file
    """
    if not states:
        states = MONITORED_STATES

    now = datetime.now(timezone.utc)
    week_end = now.strftime("%Y-%m-%d")
    week_start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

    if verbose:
        print(f"\n  📝 Generating digest...")
        print(f"     Period: {week_start} → {week_end}")
        print(f"     States: {', '.join(states)}")

    # Pull bills
    bills = get_digest_bills(days_back=days_back, states=states)

    if verbose:
        print(f"     Bills found: {len(bills):,}")

    if not bills:
        if verbose:
            print("  ⚠️  No classified bills found for this period.")
            print("     Try days_back=0 for all time, or run the pipeline first.")
        # Generate a minimal digest anyway
        bills = get_digest_bills(days_back=0, states=states)
        if bills:
            if verbose:
                print(f"     Using all-time data instead: {len(bills):,} bills")
            week_start = "all-time"

    # Generate markdown
    markdown = generate_digest_markdown(bills, week_start, week_end, states)

    # Save
    filepath = save_digest(markdown, bills, week_start, week_end, states)

    if verbose:
        pro  = sum(1 for b in bills if b.get("final_label") == "pro_animal")
        anti = sum(1 for b in bills if b.get("final_label") == "anti_animal")
        high = sum(1 for b in bills if b.get("risk_level") == "high")
        print(f"\n  ✅ Digest saved: {filepath}")
        print(f"     {len(bills):,} bills | {pro} pro | {anti} anti | {high} high-risk")
        print()

    return filepath


# ─────────────────────────────────────────────────────
# STANDALONE
# python -m src.digest.generator
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("=" * 55)
    print("  PHASE 8a — DIGEST GENERATOR")
    print("=" * 55)

    path = generate_weekly_digest(days_back=0, verbose=True)

    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Print first 60 lines as preview
        preview = "\n".join(content.split("\n")[:60])
        print(f"\n  ── PREVIEW (first 60 lines) ──\n")
        print(preview)
        print(f"\n  ... ({len(content.split(chr(10)))} total lines)")
