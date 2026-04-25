# src/digest/email_sender.py
#
# Phase 12a — Email Digest Export
#
# Converts a Markdown digest file to HTML and sends it
# via Gmail SMTP (free, uses App Password).
#
# Setup (one-time):
#   1. Enable 2-Factor Auth on your Gmail account
#   2. Go to: https://myaccount.google.com/apppasswords
#   3. Generate an App Password for "Mail"
#   4. Add to your .env file:
#        EMAIL_FROM=yourname@gmail.com
#        EMAIL_PASSWORD=xxxx xxxx xxxx xxxx
#        EMAIL_TO=recipient@example.com
#
# Usage:
#   python -m src.digest.email_sender                  # Send latest digest
#   python -m src.digest.email_sender --file digests/digest_20260424.md

import os
import sys
import smtplib
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from pathlib import Path

# ── Make imports work from project root ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
import markdown2

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ─────────────────────────────────────────────────────
# EMAIL CONFIG (loaded from .env)
# ─────────────────────────────────────────────────────

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT   = 587
EMAIL_FROM     = os.getenv("EMAIL_FROM", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_TO       = os.getenv("EMAIL_TO", "")


# ─────────────────────────────────────────────────────
# CONVERT MARKDOWN → HTML EMAIL
# ─────────────────────────────────────────────────────

# Inline CSS for email clients (they strip <style> tags)
EMAIL_CSS = """
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; color: #333; }
    .container { max-width: 700px; margin: 0 auto; background: #fff; padding: 24px; border-radius: 8px; }
    h1 { color: #1a1a2e; border-bottom: 3px solid #667eea; padding-bottom: 8px; }
    h2 { color: #16213e; margin-top: 24px; }
    h3 { color: #0f3460; margin-top: 16px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 14px; }
    th { background: #667eea; color: white; }
    tr:nth-child(even) { background: #f8f9fa; }
    blockquote { border-left: 4px solid #667eea; margin: 8px 0; padding: 4px 16px; color: #555; background: #f0f0ff; }
    a { color: #667eea; }
    code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 13px; }
    .footer { text-align: center; color: #999; font-size: 12px; margin-top: 24px; padding-top: 16px; border-top: 1px solid #eee; }
</style>
"""


def markdown_to_html(md_content: str) -> str:
    """
    Converts Markdown text to a styled HTML email body.
    Uses markdown2 for conversion, then wraps in email-safe CSS.
    """
    # Convert markdown to HTML
    html_body = markdown2.markdown(
        md_content,
        extras=["tables", "fenced-code-blocks", "header-ids", "break-on-newline"]
    )

    # Wrap in full HTML document with inline styles
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {EMAIL_CSS}
    </head>
    <body>
        <div class="container">
            {html_body}
            <div class="footer">
                Sent by Paw-dvocate Legislative Intelligence Pipeline<br>
                Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
            </div>
        </div>
    </body>
    </html>
    """
    return html


# ─────────────────────────────────────────────────────
# SEND EMAIL
# ─────────────────────────────────────────────────────

def send_digest_email(
    digest_path: str = "",
    recipient: str = "",
    subject: str = "",
) -> dict:
    """
    Sends a digest Markdown file as an HTML email.

    Parameters:
        digest_path (str): Path to .md file. If empty, uses latest digest.
        recipient (str):   Email address. If empty, uses EMAIL_TO from .env.
        subject (str):     Email subject. Auto-generated if empty.

    Returns:
        dict: {"success": bool, "message": str}
    """
    # ── Validate config ──
    sender = EMAIL_FROM
    password = EMAIL_PASSWORD
    to_addr = recipient or EMAIL_TO

    if not sender:
        return {"success": False, "message": "EMAIL_FROM not set in .env"}
    if not password:
        return {"success": False, "message": "EMAIL_PASSWORD not set in .env (use Gmail App Password)"}
    if not to_addr:
        return {"success": False, "message": "No recipient. Set EMAIL_TO in .env or pass --to"}

    # ── Find digest file ──
    if not digest_path:
        digest_path = get_latest_digest()

    if not digest_path or not os.path.exists(digest_path):
        return {"success": False, "message": f"Digest file not found: {digest_path}"}

    # ── Read and convert ──
    with open(digest_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    html_content = markdown_to_html(md_content)

    # ── Build email subject ──
    if not subject:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        subject = f"🐾 Paw-dvocate Weekly Digest — {now}"

    # ── Compose MIME message ──
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = to_addr

    # Plain text fallback (for email clients that don't render HTML)
    plain_part = MIMEText(md_content, "plain", "utf-8")
    html_part  = MIMEText(html_content, "html", "utf-8")

    msg.attach(plain_part)   # Fallback
    msg.attach(html_part)    # Preferred

    # ── Send via SMTP ──
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()       # Encrypt the connection
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, [to_addr], msg.as_string())

        return {
            "success": True,
            "message": f"Digest emailed to {to_addr} ({os.path.basename(digest_path)})",
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "message": "SMTP auth failed. Check EMAIL_FROM and EMAIL_PASSWORD in .env. "
                       "Use a Gmail App Password, not your regular password.",
        }
    except smtplib.SMTPException as e:
        return {"success": False, "message": f"SMTP error: {e}"}
    except Exception as e:
        return {"success": False, "message": f"Email error: {e}"}


# ─────────────────────────────────────────────────────
# HELPER: Find latest digest
# ─────────────────────────────────────────────────────

def get_latest_digest() -> str:
    """Returns the path to the most recent digest file."""
    digests_dir = os.path.join(PROJECT_ROOT, "digests")
    if not os.path.exists(digests_dir):
        return ""

    files = sorted(
        [f for f in os.listdir(digests_dir) if f.endswith(".md")],
        reverse=True   # Newest first (filenames are timestamped)
    )
    if files:
        return os.path.join(digests_dir, files[0])
    return ""


# ─────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email digest sender")
    parser.add_argument("--file", help="Path to digest .md file (default: latest)")
    parser.add_argument("--to", help="Recipient email (default: EMAIL_TO from .env)")
    parser.add_argument("--subject", help="Email subject line")
    parser.add_argument("--preview", action="store_true", help="Convert to HTML and save locally (no send)")
    args = parser.parse_args()

    if args.preview:
        path = args.file or get_latest_digest()
        if not path:
            print("  ❌ No digest found.")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            md = f.read()
        html = markdown_to_html(md)
        out_path = path.replace(".md", ".html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  ✅ HTML preview saved: {out_path}")
        print(f"     Open in browser to see the email format.")
        sys.exit(0)

    print("\n  📧 Sending digest email...")
    result = send_digest_email(
        digest_path=args.file or "",
        recipient=args.to or "",
        subject=args.subject or "",
    )
    if result["success"]:
        print(f"  ✅ {result['message']}")
    else:
        print(f"  ❌ {result['message']}")
