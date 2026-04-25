# frontend/app.py
#
# Phase 11 — Streamlit Dashboard for Paw-dvocate
#
# Launch:  streamlit run frontend/app.py
#
# Features:
#   🚨 Anti-animal alerts (high-risk bills)
#   🌱 Pro-animal opportunities
#   📋 Recent movement (status changes)
#   📊 Stats bar (counts, distributions)
#   🔍 State selector dropdown (CA / TX / NY / All)
#   📄 Bill detail cards with AI reasoning

import os
import sys
import sqlite3
from datetime import datetime, timezone

# ── Make imports work from project root ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd

from config.settings import DB_PATH, MONITORED_STATES


# ─────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Paw-dvocate Dashboard",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────
# CUSTOM CSS for premium look
# ─────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.8;
        font-size: 0.95rem;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stat-card.pro {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .stat-card.anti {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .stat-card.neutral {
        background: linear-gradient(135deg, #606c88 0%, #3f4c6b 100%);
    }
    .stat-card.high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    .stat-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .stat-card p {
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }

    /* Bill cards */
    .bill-card {
        background: #1e1e2f;
        border: 1px solid #2d2d44;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .bill-card:hover {
        border-color: #667eea;
    }
    .bill-card .bill-title {
        font-weight: 600;
        font-size: 1rem;
        color: #e0e0e0;
        margin-bottom: 0.5rem;
    }
    .bill-card .bill-meta {
        font-size: 0.82rem;
        color: #999;
        margin-bottom: 0.4rem;
    }
    .bill-card .bill-reasoning {
        font-size: 0.85rem;
        color: #b0b0b0;
        font-style: italic;
        border-left: 3px solid #667eea;
        padding-left: 0.8rem;
        margin-top: 0.5rem;
    }

    /* Labels */
    .label-pro {
        background: #11998e;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .label-anti {
        background: #eb3349;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .label-neutral {
        background: #606c88;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .label-high {
        background: #ff416c;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .label-medium {
        background: #f5a623;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2d2d44;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# DATABASE CONNECTION
# ─────────────────────────────────────────────────────

@st.cache_resource
def get_db_path():
    """Returns the absolute path to the SQLite database."""
    return os.path.join(PROJECT_ROOT, DB_PATH)


def query_db(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Runs a SQL query and returns a pandas DataFrame."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# ─────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # Cache for 5 minutes
def load_bills(state: str = "All") -> pd.DataFrame:
    """
    Loads classified bills from SQLite.
    Joins bills + classifications tables.
    Filters by state if not "All".
    """
    sql = """
        SELECT
            b.bill_id, b.state, b.bill_number, b.title,
            b.description, b.status, b.status_date,
            b.introduced_date, b.last_action, b.last_action_date,
            b.url, b.sponsors, b.committee, b.subjects,
            c.keyword_score, c.keyword_match,
            c.embedding_similarity, c.embedding_label,
            c.groq_label, c.groq_confidence, c.groq_reasoning,
            c.openpaws_alignment_score, c.openpaws_framing_summary,
            c.final_label, c.final_confidence, c.relevance_score,
            c.risk_level
        FROM classifications c
        INNER JOIN bills b ON b.bill_id = c.bill_id
        WHERE c.final_label IS NOT NULL AND c.final_label != ''
    """
    params = ()

    if state != "All":
        sql += " AND b.state = ?"
        params = (state,)

    sql += " ORDER BY c.relevance_score DESC, c.final_confidence DESC"

    return query_db(sql, params)


# ─────────────────────────────────────────────────────
# HELPER: Render a bill card
# ─────────────────────────────────────────────────────

def render_bill_card(row):
    """Renders a single bill as a styled card."""
    label = row.get("final_label", "neutral")
    risk = row.get("risk_level", "low")
    conf = row.get("final_confidence", 0)
    rel = row.get("relevance_score", 0)
    align = row.get("openpaws_alignment_score", 0) or 0

    # Label badge
    label_class = {"pro_animal": "label-pro", "anti_animal": "label-anti"}.get(label, "label-neutral")
    label_text = {"pro_animal": "PRO-ANIMAL", "anti_animal": "ANTI-ANIMAL"}.get(label, "NEUTRAL")

    # Risk badge
    risk_class = {"high": "label-high", "medium": "label-medium"}.get(risk, "")
    risk_html = f'<span class="{risk_class}">{risk.upper()}</span> ' if risk in ("high", "medium") else ""

    # Build card HTML
    title = (row.get("title") or "No title")[:100]
    state = row.get("state", "")
    number = row.get("bill_number", "")
    status = row.get("status", "")
    committee = row.get("committee", "") or ""
    sponsors = (row.get("sponsors", "") or "")[:80]
    url = row.get("url", "") or ""
    reasoning = (row.get("groq_reasoning", "") or "")[:200]
    framing = (row.get("openpaws_framing_summary", "") or "")[:150]

    link_html = f'<a href="{url}" target="_blank" style="color:#667eea;text-decoration:none;">View on LegiScan ↗</a>' if url else ""

    reasoning_html = ""
    if reasoning:
        reasoning_html = f'<div class="bill-reasoning">🤖 {reasoning}</div>'
    elif framing:
        reasoning_html = f'<div class="bill-reasoning">🐾 {framing}</div>'

    card_html = f"""
    <div class="bill-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
            <span style="font-weight:600;color:#b0b0b0;font-size:0.85rem;">{state} {number}</span>
            <span>{risk_html}<span class="{label_class}">{label_text}</span></span>
        </div>
        <div class="bill-title">{title}</div>
        <div class="bill-meta">
            📊 Confidence: {conf:.0%} &nbsp;|&nbsp;
            🎯 Relevance: {rel:.0%} &nbsp;|&nbsp;
            ⚖️ Alignment: {align:+.2f} &nbsp;|&nbsp;
            📌 Status: {status}
        </div>
        <div class="bill-meta">
            🏛️ {committee} &nbsp;|&nbsp; 👤 {sponsors}
        </div>
        {reasoning_html}
        <div style="margin-top:0.4rem;">{link_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🐾 Paw-dvocate")
    st.markdown("*Legislative Intelligence Dashboard*")
    st.markdown("---")

    # State selector
    state_options = ["All"] + MONITORED_STATES
    selected_state = st.selectbox(
        "🗺️ Select State",
        state_options,
        index=0,
        help="Filter bills by state"
    )

    # Risk filter
    risk_filter = st.multiselect(
        "⚠️ Risk Level",
        ["high", "medium", "low"],
        default=["high", "medium", "low"],
        help="Filter by risk level"
    )

    # Label filter
    label_filter = st.multiselect(
        "🏷️ Classification",
        ["pro_animal", "anti_animal", "neutral"],
        default=["pro_animal", "anti_animal", "neutral"],
        help="Filter by classification label"
    )

    # Confidence threshold
    conf_threshold = st.slider(
        "📊 Min Confidence",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="Only show bills above this confidence"
    )

    st.markdown("---")

    # Database info
    db_path = get_db_path()
    if os.path.exists(db_path):
        db_size = os.path.getsize(db_path) / (1024 * 1024)
        st.markdown(f"💾 Database: **{db_size:.1f} MB**")

    st.markdown(f"🕐 Updated: **{datetime.now().strftime('%H:%M')}**")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────

df = load_bills(selected_state)

# Apply filters
if risk_filter:
    df = df[df["risk_level"].isin(risk_filter)]
if label_filter:
    df = df[df["final_label"].isin(label_filter)]
if conf_threshold > 0:
    df = df[df["final_confidence"] >= conf_threshold]


# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────

state_display = selected_state if selected_state != "All" else "CA, TX, NY"
st.markdown(f"""
<div class="main-header">
    <h1>🐾 Paw-dvocate Dashboard</h1>
    <p>Animal Legislation Intelligence — {state_display} — {len(df):,} bills classified</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# 📊 STATS BAR
# ─────────────────────────────────────────────────────

total = len(df)
pro_count = len(df[df["final_label"] == "pro_animal"])
anti_count = len(df[df["final_label"] == "anti_animal"])
neutral_count = len(df[df["final_label"] == "neutral"])
high_count = len(df[df["risk_level"] == "high"])

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <h3>{total:,}</h3>
        <p>Total Bills</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card pro">
        <h3>{pro_count:,}</h3>
        <p>🌱 Pro-Animal</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card anti">
        <h3>{anti_count:,}</h3>
        <p>⚠️ Anti-Animal</p>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-card neutral">
        <h3>{neutral_count:,}</h3>
        <p>➖ Neutral</p>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="stat-card high">
        <h3>{high_count:,}</h3>
        <p>🔴 High Risk</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# 🚨 ANTI-ANIMAL ALERTS (high risk)
# ─────────────────────────────────────────────────────

alerts = df[(df["final_label"] == "anti_animal") & (df["risk_level"] == "high")]
if len(alerts) > 0:
    st.markdown('<div class="section-header">🚨 Anti-Animal Alerts — High Risk</div>',
                unsafe_allow_html=True)
    st.markdown(f"*{len(alerts)} bills that may weaken animal protections — needs immediate attention*")

    for _, row in alerts.head(10).iterrows():
        render_bill_card(row)


# ─────────────────────────────────────────────────────
# 🌱 PRO-ANIMAL OPPORTUNITIES
# ─────────────────────────────────────────────────────

pro_bills = df[(df["final_label"] == "pro_animal") & (df["final_confidence"] > 0.3)]
if len(pro_bills) > 0:
    st.markdown('<div class="section-header">🌱 Pro-Animal Opportunities</div>',
                unsafe_allow_html=True)
    st.markdown(f"*{len(pro_bills)} bills that strengthen animal welfare — support these*")

    for _, row in pro_bills.head(15).iterrows():
        render_bill_card(row)


# ─────────────────────────────────────────────────────
# 📋 RECENT MOVEMENT (any bill with recent activity)
# ─────────────────────────────────────────────────────

# Show bills with non-"Introduced" status (they've moved through the process)
moved = df[~df["status"].isin(["Introduced", "NA", ""])]
if len(moved) > 0:
    st.markdown('<div class="section-header">📋 Recent Movement — Bills in Progress</div>',
                unsafe_allow_html=True)
    st.markdown(f"*{len(moved)} bills that have moved beyond introduction*")

    # Sort by status_date descending
    moved_sorted = moved.sort_values("status_date", ascending=False)
    for _, row in moved_sorted.head(10).iterrows():
        render_bill_card(row)


# ─────────────────────────────────────────────────────
# 📊 DISTRIBUTION CHARTS
# ─────────────────────────────────────────────────────

st.markdown('<div class="section-header">📊 Classification Distribution</div>',
            unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Label distribution by state
    if total > 0:
        label_counts = df.groupby(["state", "final_label"]).size().reset_index(name="count")
        chart_df = label_counts.pivot(index="state", columns="final_label", values="count").fillna(0)
        st.bar_chart(chart_df, color=["#eb3349", "#606c88", "#11998e"])
        st.caption("Bills by State and Classification")

with chart_col2:
    # Risk distribution
    if total > 0:
        risk_counts = df["risk_level"].value_counts()
        st.bar_chart(risk_counts, color="#667eea")
        st.caption("Bills by Risk Level")


# ─────────────────────────────────────────────────────
# 🔍 SEARCH & DATA TABLE
# ─────────────────────────────────────────────────────

st.markdown('<div class="section-header">🔍 Search All Bills</div>',
            unsafe_allow_html=True)

search_query = st.text_input(
    "Search by title, bill number, or keyword",
    placeholder="e.g. 'animal cruelty' or 'AB100'"
)

if search_query:
    mask = (
        df["title"].str.contains(search_query, case=False, na=False) |
        df["bill_number"].str.contains(search_query, case=False, na=False) |
        df["subjects"].str.contains(search_query, case=False, na=False)
    )
    results = df[mask]
    st.markdown(f"**{len(results)}** bills found for *'{search_query}'*")

    for _, row in results.head(20).iterrows():
        render_bill_card(row)

# Data table toggle
with st.expander("📋 View Raw Data Table"):
    display_cols = [
        "state", "bill_number", "title", "final_label",
        "final_confidence", "relevance_score", "risk_level",
        "status", "committee", "groq_reasoning",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available_cols].head(100),
        width='stretch',
        hide_index=True,
    )


# ─────────────────────────────────────────────────────
# 📧 EMAIL DIGEST SECTION
# ─────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">📧 Get the Weekly Digest</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
            padding:1.5rem 2rem;border-radius:12px;margin-bottom:1rem;">
    <p style="color:#e0e0e0;margin:0 0 0.3rem 0;font-size:1rem;">
        📬 Enter your email to receive the latest intelligence digest — including
        high-risk alerts, pro-animal opportunities, and full AI analysis.
    </p>
    <p style="color:#999;margin:0;font-size:0.8rem;">
        Your email is used only to send the digest. We don't store it or subscribe you to anything.
    </p>
</div>
""", unsafe_allow_html=True)

email_col1, email_col2 = st.columns([3, 1])

with email_col1:
    user_email = st.text_input(
        "Your email address",
        placeholder="name@example.com",
        label_visibility="collapsed",
        key="digest_email_input",
    )

with email_col2:
    send_clicked = st.button("📧 Send Digest", type="primary", use_container_width=True)

if send_clicked:
    if not user_email or "@" not in user_email or "." not in user_email:
        st.error("⚠️ Please enter a valid email address.")
    else:
        with st.spinner("Sending digest..."):
            try:
                from src.digest.email_sender import send_digest_email
                result = send_digest_email(recipient=user_email)
                if result["success"]:
                    st.success(f"✅ Digest sent to **{user_email}**! Check your inbox (and spam folder).")
                else:
                    st.error(f"❌ Failed to send: {result['message']}")
            except Exception as e:
                st.error(f"❌ Error: {e}")


# ─────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.8rem;'>"
    "🐾 Paw-dvocate — Animal Legislation Intelligence Pipeline<br>"
    "Built with Streamlit • Powered by Groq + sentence-transformers"
    "</div>",
    unsafe_allow_html=True,
)
