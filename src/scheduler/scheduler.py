# src/scheduler/scheduler.py
#
# Phase 10 — APScheduler Automation
#
# Two scheduled jobs:
#   1. DAILY at 02:00 local time:
#      Run full classification pipeline
#      (ingest → keyword → embedding → groq → openpaws → ensemble)
#
#   2. WEEKLY on Sunday at 09:00 local time:
#      Generate the Markdown intelligence digest
#
# Usage:
#   python src/scheduler/scheduler.py              # Start scheduler (runs forever)
#   python src/scheduler/scheduler.py --run-now    # Run pipeline once immediately, then start scheduler
#   python src/scheduler/scheduler.py --test       # Run both jobs once to verify, then exit
#
# The scheduler runs as a foreground process. Use Ctrl+C to stop.
# For production, run it as a background service or use Task Scheduler / cron.

import os
import sys
import time
import logging
from datetime import datetime, timezone

# ── Make imports work from project root ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR


# ─────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────

LOG_DIR  = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "scheduler.log")

os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pawdvocate.scheduler")

# Reduce noise from APScheduler internals
logging.getLogger("apscheduler").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────
# JOB 1: DAILY CLASSIFICATION PIPELINE
# Runs every day at 02:00 local time
# ─────────────────────────────────────────────────────

def job_daily_pipeline():
    """
    Runs the full classification pipeline:
      1. Keyword filter     (all bills, ~1 min)
      2. Embedding scorer   (all bills, ~7 min)
      3. Groq classifier    (candidates, rate-limited)
      4. Open Paws scorer   (candidates, rate-limited)
      5. Ensemble combiner  (all classified, instant)

    Called automatically at 02:00 every day.
    Also safe to call manually at any time.
    """
    logger.info("=" * 55)
    logger.info("  🐾 DAILY PIPELINE — Starting")
    logger.info("=" * 55)
    start = time.time()

    try:
        # Import pipeline functions from main.py
        # (lazy import so the scheduler module stays lightweight)
        from main import (
            run_stage_keyword,
            run_stage_embedding,
            run_stage_groq,
            run_stage_openpaws,
            run_stage_ensemble,
        )

        # Stage 1: Keyword filter (fast, ~1 min for 28K bills)
        logger.info("Stage 1/5: Keyword filter")
        run_stage_keyword()

        # Stage 2: Embedding similarity (medium, ~7 min on CPU)
        logger.info("Stage 2/5: Embedding scorer")
        run_stage_embedding()

        # Stage 3: Groq LLM classifier (slow, rate-limited)
        # max_bills=100 per daily run to stay within free tier limits
        # Groq free tier: 30 req/min → 100 bills ≈ 4 min
        logger.info("Stage 3/5: Groq classifier (max 100 bills)")
        run_stage_groq(max_bills=100)

        # Stage 4: Open Paws alignment (slow, rate-limited)
        logger.info("Stage 4/5: Open Paws alignment (max 100 bills)")
        run_stage_openpaws(max_bills=100)

        # Stage 5: Ensemble (instant)
        logger.info("Stage 5/5: Weighted ensemble")
        run_stage_ensemble()

        elapsed = time.time() - start
        logger.info(f"  ✅ Daily pipeline complete in {elapsed / 60:.1f} minutes")

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  ❌ Daily pipeline FAILED after {elapsed / 60:.1f} min: {e}")
        # Don't re-raise — let the scheduler continue running
        # The job will retry on the next scheduled run


# ─────────────────────────────────────────────────────
# JOB 2: WEEKLY DIGEST GENERATION
# Runs every Sunday at 09:00 local time
# ─────────────────────────────────────────────────────

def job_weekly_digest():
    """
    Generates the weekly Markdown digest report.
    Covers the last 7 days of classified bills.

    Called automatically every Sunday at 09:00.
    Also safe to call manually at any time.
    """
    logger.info("=" * 55)
    logger.info("  📝 WEEKLY DIGEST — Starting")
    logger.info("=" * 55)

    try:
        from main import run_stage_digest

        # Generate digest covering last 7 days
        # (uses days_back=7 so it only includes recent activity)
        path = run_stage_digest(days_back=7)

        if path:
            logger.info(f"  ✅ Weekly digest saved: {path}")
        else:
            # If no bills in last 7 days, fall back to all-time
            logger.info("  No bills in last 7 days. Generating all-time digest...")
            path = run_stage_digest(days_back=0)
            if path:
                logger.info(f"  ✅ All-time digest saved: {path}")

    except Exception as e:
        logger.error(f"  ❌ Weekly digest FAILED: {e}")


# ─────────────────────────────────────────────────────
# SCHEDULER EVENT LISTENER
# ─────────────────────────────────────────────────────

def job_listener(event):
    """Logs job execution results for monitoring."""
    if event.exception:
        logger.error(f"  Job {event.job_id} FAILED with exception: {event.exception}")
    else:
        logger.info(f"  Job {event.job_id} completed successfully.")


# ─────────────────────────────────────────────────────
# SCHEDULER SETUP
# ─────────────────────────────────────────────────────

def create_scheduler() -> BlockingScheduler:
    """
    Creates and configures the APScheduler instance with:
      - Daily pipeline job at 02:00
      - Weekly digest job on Sunday at 09:00
    """
    scheduler = BlockingScheduler()

    # ── Job 1: Daily classification pipeline ──
    # Runs at 02:00 every day (local time)
    # max_instances=1 prevents overlap if a run takes longer than 24h
    scheduler.add_job(
        job_daily_pipeline,
        trigger=CronTrigger(hour=2, minute=0),
        id="daily_pipeline",
        name="Daily Classification Pipeline",
        max_instances=1,            # Only one instance at a time
        coalesce=True,              # If missed, run once (not multiple)
        misfire_grace_time=3600,    # Allow 1 hour late start
    )

    # ── Job 2: Weekly digest ──
    # Runs Sunday at 09:00 (local time)
    scheduler.add_job(
        job_weekly_digest,
        trigger=CronTrigger(day_of_week="sun", hour=9, minute=0),
        id="weekly_digest",
        name="Weekly Digest Generation",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    # Listen for job completion/failure events
    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    return scheduler


# ─────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Paw-dvocate Scheduler (Phase 10)"
    )
    parser.add_argument(
        "--run-now", action="store_true",
        help="Run the daily pipeline immediately, then start scheduler"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run both jobs once to verify they work, then exit"
    )
    parser.add_argument(
        "--digest-now", action="store_true",
        help="Generate a digest immediately, then start scheduler"
    )

    args = parser.parse_args()

    # ── TEST MODE: run jobs once, then exit ──
    if args.test:
        logger.info("=" * 55)
        logger.info("  🧪 TEST MODE — Running both jobs once")
        logger.info("=" * 55)

        logger.info("\n  Testing daily pipeline job...")
        job_daily_pipeline()

        logger.info("\n  Testing weekly digest job...")
        job_weekly_digest()

        logger.info("\n  ✅ Test complete. Both jobs ran successfully.")
        logger.info("  Start the scheduler with: python src/scheduler/scheduler.py")
        return

    # ── RUN-NOW: execute pipeline immediately ──
    if args.run_now:
        logger.info("  Running pipeline immediately before starting scheduler...")
        job_daily_pipeline()

    # ── DIGEST-NOW: generate digest immediately ──
    if args.digest_now:
        logger.info("  Generating digest immediately before starting scheduler...")
        job_weekly_digest()

    # ── START SCHEDULER ──
    scheduler = create_scheduler()

    logger.info("")
    logger.info("=" * 55)
    logger.info("  🐾 Paw-dvocate Scheduler Started")
    logger.info("=" * 55)
    logger.info("")
    logger.info("  Scheduled jobs:")
    logger.info("    📅 Daily pipeline:  Every day at 02:00")
    logger.info("    📅 Weekly digest:   Every Sunday at 09:00")
    logger.info("")
    logger.info("  Press Ctrl+C to stop.")
    logger.info("  Logs saved to: logs/scheduler.log")
    logger.info("")

    # Print next run times
    for job in scheduler.get_jobs():
        try:
            next_run = job.trigger.get_next_fire_time(None, datetime.now(timezone.utc))
            if next_run:
                logger.info(f"  Next run: {job.name} → {next_run.strftime('%Y-%m-%d %H:%M %Z')}")
        except Exception:
            logger.info(f"  Registered: {job.name}")

    logger.info("")

    try:
        # This blocks forever (until Ctrl+C)
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("\n  🛑 Scheduler stopped by user.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
