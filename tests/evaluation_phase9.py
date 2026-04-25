# tests/evaluation_phase9.py
#
# Phase 9 — Accuracy Benchmark & Evaluation
#
# What this script does:
#   1. Samples 20 classified bills from the database (stratified by label)
#   2. Saves them to tests/manual_labels.csv for YOU to manually label
#   3. After you label them, run this script again to compute metrics
#   4. Outputs: accuracy, precision, recall, F1, confusion matrix
#   5. Saves full report to logs/evaluation_report.txt
#
# Usage:
#   Step 1: python tests/evaluation_phase9.py --generate
#           → Creates tests/manual_labels.csv (fill in "human_label" column)
#
#   Step 2: python tests/evaluation_phase9.py --evaluate
#           → Reads your labels, computes metrics, saves report
#
#   Step 3: python tests/evaluation_phase9.py --show
#           → Prints the saved report
#
# Pass threshold: 85% accuracy

import os
import sys
import csv
import random
import sqlite3
import argparse
from datetime import datetime, timezone
from collections import Counter

# ── Make imports work from project root ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import DB_PATH


# ─────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────

CSV_PATH    = os.path.join(PROJECT_ROOT, "tests", "manual_labels.csv")
REPORT_PATH = os.path.join(PROJECT_ROOT, "logs", "evaluation_report.txt")
SAMPLE_SIZE = 20   # How many bills to sample for evaluation
PASS_THRESHOLD = 0.85   # 85% accuracy target


# ─────────────────────────────────────────────────────
# DATABASE HELPER
# ─────────────────────────────────────────────────────

def get_db():
    """Opens a read-only connection to the legislation database."""
    db_path = os.path.join(PROJECT_ROOT, DB_PATH)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────
# STEP 1: GENERATE SAMPLE CSV
# ─────────────────────────────────────────────────────

def generate_sample_csv():
    """
    Picks 20 bills using stratified sampling:
      - 8 pro_animal  (proportional to high-confidence first)
      - 8 anti_animal
      - 4 neutral
    Prioritizes Groq-classified bills (they have reasoning).
    Saves to tests/manual_labels.csv with a blank human_label column.
    """
    conn = get_db()

    # ── Pull bills with final classifications ──
    # Prefer bills that have Groq reasoning (richer data to evaluate)
    rows = conn.execute("""
        SELECT
            b.bill_id, b.state, b.bill_number, b.title,
            b.description, b.status, b.subjects, b.committee,
            c.final_label, c.final_confidence, c.relevance_score,
            c.risk_level, c.keyword_score,
            c.groq_label, c.groq_confidence, c.groq_reasoning,
            c.openpaws_alignment_score, c.openpaws_framing_summary,
            c.embedding_label, c.embedding_similarity
        FROM bills b
        INNER JOIN classifications c ON b.bill_id = c.bill_id
        WHERE c.final_label IS NOT NULL AND c.final_label != ''
        ORDER BY
            CASE WHEN c.groq_label != '' THEN 0 ELSE 1 END,
            c.final_confidence DESC
    """).fetchall()
    conn.close()

    all_bills = [dict(r) for r in rows]

    # ── Stratified sampling ──
    pro_bills   = [b for b in all_bills if b["final_label"] == "pro_animal"]
    anti_bills  = [b for b in all_bills if b["final_label"] == "anti_animal"]
    neut_bills  = [b for b in all_bills if b["final_label"] == "neutral"]

    # Shuffle each group so we don't just get the top-confidence ones
    # But keep Groq-classified bills at the front (they were sorted first)
    random.seed(42)   # Reproducible sampling

    # Take some from high-confidence and some from lower to test edge cases
    def stratified_pick(pool, n):
        """Pick n bills: half from top confidence, half random."""
        if len(pool) <= n:
            return pool
        top_half = pool[:len(pool) // 3]         # Top third by confidence
        rest     = pool[len(pool) // 3:]          # Bottom two-thirds
        picks = []
        picks += random.sample(top_half, min(n // 2, len(top_half)))
        picks += random.sample(rest, min(n - len(picks), len(rest)))
        return picks[:n]

    sample = []
    sample += stratified_pick(pro_bills, 8)
    sample += stratified_pick(anti_bills, 8)
    sample += stratified_pick(neut_bills, 4)

    # Shuffle final order so labels aren't grouped
    random.shuffle(sample)

    # ── Write CSV ──
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    fieldnames = [
        "bill_id", "state", "bill_number", "title", "description",
        "subjects", "committee", "status",
        "pipeline_label", "pipeline_confidence", "relevance_score", "risk_level",
        "groq_label", "groq_confidence", "groq_reasoning",
        "alignment_score", "framing_summary",
        "keyword_score", "embedding_label", "embedding_similarity",
        "human_label",   # ← THIS IS THE COLUMN YOU FILL IN
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for b in sample:
            writer.writerow({
                "bill_id":              b["bill_id"],
                "state":                b["state"],
                "bill_number":          b["bill_number"],
                "title":                b["title"],
                "description":          (b["description"] or "")[:300],
                "subjects":             b["subjects"] or "",
                "committee":            b["committee"] or "",
                "status":               b["status"] or "",
                "pipeline_label":       b["final_label"],
                "pipeline_confidence":  round(b["final_confidence"], 4),
                "relevance_score":      round(b["relevance_score"], 4),
                "risk_level":           b["risk_level"] or "",
                "groq_label":           b["groq_label"] or "",
                "groq_confidence":      b["groq_confidence"] or 0.0,
                "groq_reasoning":       (b["groq_reasoning"] or "")[:200],
                "alignment_score":      b["openpaws_alignment_score"] or 0.0,
                "framing_summary":      (b["openpaws_framing_summary"] or "")[:200],
                "keyword_score":        b["keyword_score"] or 0.0,
                "embedding_label":      b["embedding_label"] or "",
                "embedding_similarity": b["embedding_similarity"] or 0.0,
                "human_label":          "",   # BLANK — you fill this in!
            })

    print(f"\n  ✅ Sample CSV created: {CSV_PATH}")
    print(f"     {len(sample)} bills sampled ({len([b for b in sample if b['final_label']=='pro_animal'])} pro, "
          f"{len([b for b in sample if b['final_label']=='anti_animal'])} anti, "
          f"{len([b for b in sample if b['final_label']=='neutral'])} neutral)")
    print()
    print("  📝 NEXT STEPS:")
    print("     1. Open tests/manual_labels.csv in Excel/Sheets/VS Code")
    print("     2. Read each bill's title + description")
    print("     3. Fill in the 'human_label' column with one of:")
    print("          pro_animal   — Bill helps/protects animals")
    print("          anti_animal  — Bill harms animals or weakens protections")
    print("          neutral      — Bill is unrelated to animal welfare")
    print("     4. Save the file")
    print("     5. Run: python tests/evaluation_phase9.py --evaluate")
    print()

    return sample


# ─────────────────────────────────────────────────────
# STEP 2: EVALUATE (read CSV, compute metrics)
# ─────────────────────────────────────────────────────

LABELS = ["pro_animal", "anti_animal", "neutral"]


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Computes accuracy, per-class precision/recall/F1,
    macro averages, and confusion matrix.

    We do this manually (no sklearn dependency for this part)
    so the script stays lightweight.
    """
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n > 0 else 0.0

    # ── Per-class metrics ──
    per_class = {}
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    # ── Macro averages ──
    macro_precision = sum(v["precision"] for v in per_class.values()) / len(LABELS)
    macro_recall    = sum(v["recall"]    for v in per_class.values()) / len(LABELS)
    macro_f1        = sum(v["f1"]        for v in per_class.values()) / len(LABELS)

    # ── Confusion matrix ──
    # Rows = true label, Columns = predicted label
    confusion = {}
    for true_label in LABELS:
        confusion[true_label] = {}
        for pred_label in LABELS:
            confusion[true_label][pred_label] = sum(
                1 for t, p in zip(y_true, y_pred)
                if t == true_label and p == pred_label
            )

    return {
        "accuracy":        round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall":    round(macro_recall, 4),
        "macro_f1":        round(macro_f1, 4),
        "per_class":       per_class,
        "confusion":       confusion,
        "n":               n,
        "correct":         correct,
    }


def evaluate():
    """
    Reads manual_labels.csv, computes metrics, prints report,
    and saves to logs/evaluation_report.txt.
    """
    if not os.path.exists(CSV_PATH):
        print(f"\n  ❌ File not found: {CSV_PATH}")
        print("     Run with --generate first.")
        return None

    # ── Read CSV ──
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # ── Validate human labels ──
    y_true = []   # human labels (ground truth)
    y_pred = []   # pipeline predictions
    skipped = 0
    details = []  # for the detailed report

    for row in rows:
        human = row.get("human_label", "").strip().lower()
        pipeline = row.get("pipeline_label", "").strip().lower()

        if not human:
            skipped += 1
            continue

        # Normalize common variations
        human = human.replace("-", "_").replace(" ", "_")
        if human == "pro":
            human = "pro_animal"
        elif human == "anti":
            human = "anti_animal"

        if human not in LABELS:
            print(f"  ⚠️  Skipping invalid label '{human}' for {row.get('bill_id')}")
            print(f"       Valid labels: {LABELS}")
            skipped += 1
            continue

        y_true.append(human)
        y_pred.append(pipeline)
        details.append({
            "bill_id":    row["bill_id"],
            "title":      row["title"][:60],
            "human":      human,
            "pipeline":   pipeline,
            "match":      "✅" if human == pipeline else "❌",
            "confidence": row.get("pipeline_confidence", ""),
        })

    if not y_true:
        print("\n  ❌ No labeled rows found in manual_labels.csv!")
        print("     Fill in the 'human_label' column and try again.")
        return None

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} rows (blank or invalid human_label)")

    # ── Compute metrics ──
    metrics = compute_metrics(y_true, y_pred)

    # ── Build report ──
    report_lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report_lines.append("=" * 60)
    report_lines.append("  PAW-DVOCATE EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"  Date:     {now}")
    report_lines.append(f"  Samples:  {metrics['n']}")
    report_lines.append(f"  Correct:  {metrics['correct']}")
    report_lines.append("")

    # ── Overall metrics ──
    acc = metrics["accuracy"]
    passed = acc >= PASS_THRESHOLD
    status = "✅ PASS" if passed else "❌ FAIL"

    report_lines.append(f"  ACCURACY:  {acc:.1%}  (threshold: {PASS_THRESHOLD:.0%})  {status}")
    report_lines.append("")
    report_lines.append(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    report_lines.append(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    report_lines.append(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    report_lines.append("")

    # ── Per-class breakdown ──
    report_lines.append("  ── PER-CLASS METRICS ──")
    report_lines.append("")
    report_lines.append(f"  {'Label':<14s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'TP':>3s}  {'FP':>3s}  {'FN':>3s}")
    report_lines.append(f"  {'─' * 52}")

    for label in LABELS:
        m = metrics["per_class"][label]
        report_lines.append(
            f"  {label:<14s}  {m['precision']:6.2%}  {m['recall']:6.2%}  {m['f1']:6.2%}  "
            f"{m['tp']:3d}  {m['fp']:3d}  {m['fn']:3d}"
        )
    report_lines.append("")

    # ── Confusion matrix ──
    report_lines.append("  ── CONFUSION MATRIX ──")
    report_lines.append("  (Rows = Human Label, Columns = Pipeline Prediction)")
    report_lines.append("")

    # Header
    header = f"  {'':14s}"
    for pred in LABELS:
        short = pred.replace("_animal", "").replace("al", "")[:6]
        header += f"  {short:>6s}"
    report_lines.append(header)
    report_lines.append(f"  {'─' * 38}")

    for true_label in LABELS:
        row_str = f"  {true_label:<14s}"
        for pred_label in LABELS:
            val = metrics["confusion"][true_label][pred_label]
            row_str += f"  {val:6d}"
        report_lines.append(row_str)
    report_lines.append("")

    # ── Detailed bill-by-bill results ──
    report_lines.append("  ── BILL-BY-BILL RESULTS ──")
    report_lines.append("")
    for d in details:
        report_lines.append(
            f"  {d['match']} {d['bill_id']:15s}  "
            f"human={d['human']:12s}  pipeline={d['pipeline']:12s}  "
            f"conf={d['confidence']}"
        )
        report_lines.append(f"     {d['title']}")
        report_lines.append("")

    # ── Misclassified analysis ──
    misses = [d for d in details if d["match"] == "❌"]
    if misses:
        report_lines.append("  ── MISCLASSIFIED BILLS ──")
        report_lines.append("")
        for d in misses:
            report_lines.append(f"  ❌ {d['bill_id']}: {d['title']}")
            report_lines.append(f"     Human: {d['human']}  |  Pipeline: {d['pipeline']}")
            report_lines.append("")

    report_lines.append("=" * 60)
    report_text = "\n".join(report_lines)

    # ── Print report ──
    print()
    print(report_text)

    # ── Save report ──
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  📄 Report saved: {REPORT_PATH}")
    print()

    return metrics


# ─────────────────────────────────────────────────────
# STEP 3: SHOW SAVED REPORT
# ─────────────────────────────────────────────────────

def show_report():
    """Prints the last saved evaluation report."""
    if not os.path.exists(REPORT_PATH):
        print(f"\n  ❌ No report found at {REPORT_PATH}")
        print("     Run --evaluate first.")
        return

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        print(f.read())


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paw-dvocate Evaluation Benchmark (Phase 9)"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate tests/manual_labels.csv with 20 sampled bills"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Read manual_labels.csv, compute metrics, save report"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Print the last saved evaluation report"
    )

    args = parser.parse_args()

    # Default to --generate if no flag given
    if not any([args.generate, args.evaluate, args.show]):
        print("\n  Usage:")
        print("    python tests/evaluation_phase9.py --generate   # Create CSV")
        print("    python tests/evaluation_phase9.py --evaluate   # Compute metrics")
        print("    python tests/evaluation_phase9.py --show       # Print report")
        print()
        # Auto-generate if CSV doesn't exist
        if not os.path.exists(CSV_PATH):
            print("  No CSV found. Generating sample now...\n")
            generate_sample_csv()
        else:
            print(f"  CSV exists: {CSV_PATH}")
            print("  Run with --evaluate after labeling.\n")
        return

    if args.generate:
        generate_sample_csv()

    if args.evaluate:
        evaluate()

    if args.show:
        show_report()


if __name__ == "__main__":
    main()
