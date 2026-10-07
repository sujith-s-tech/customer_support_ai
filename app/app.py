"""
app.py -- Phase 15: Streamlit Application
==========================================
Title: AI-Powered Banking Customer Support Intelligence

Pages:
  1. Customer Support     -- Query input -> intent + response
  2. Conversation Intelligence -- Deep analysis with sentiment, key info
  3. Analytics Dashboard  -- KPI cards + charts
  4. Model Performance    -- ML vs DL comparison
  5. Escalation Queue     -- Cases requiring human attention

Run:
    cd banking-customer-support-ai
    streamlit run app/app.py
"""

import sys
import time
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ──────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT    = APP_DIR.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Imports from src ────────────────────────────────────────
from data_preprocessing     import clean_text
from priority               import PriorityDetector
from confidence             import get_confidence_tier, escalation_reason, load_thresholds
from conversation_intelligence import ConversationAnalyzer, detect_sentiment
from response_generator     import ResponseGenerator
from analytics              import AnalyticsEngine
from utils                  import load_json, ensure_dir

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI-Powered Banking Customer Support Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ── KPI cards ── */
.kpi-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.kpi-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #06d6a0, #118ab2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Intent badge ── */
.intent-badge {
    display: inline-block;
    background: linear-gradient(90deg, #06d6a0, #118ab2);
    color: white;
    padding: 6px 18px;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.95rem;
}

/* ── Priority badges ── */
.badge-high   { background:#ef476f; color:white; padding:4px 14px; border-radius:50px; font-weight:600; font-size:0.85rem; }
.badge-medium { background:#ffd166; color:#1a1a2e; padding:4px 14px; border-radius:50px; font-weight:600; font-size:0.85rem; }
.badge-low    { background:#06d6a0; color:#1a1a2e; padding:4px 14px; border-radius:50px; font-weight:600; font-size:0.85rem; }

/* ── Response card ── */
.response-card {
    background: rgba(6,214,160,0.08);
    border: 1px solid rgba(6,214,160,0.3);
    border-radius: 12px;
    padding: 20px;
    margin-top: 12px;
}

/* ── Section header ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #06d6a0;
    border-bottom: 2px solid rgba(6,214,160,0.3);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* ── Escalation banner ── */
.escalation-banner {
    background: linear-gradient(90deg, rgba(239,71,111,0.2), rgba(239,71,111,0.05));
    border: 1px solid rgba(239,71,111,0.5);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
}

/* ── Top3 confidence bars ── */
.top3-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #06d6a0, #118ab2);
    transition: width 0.5s ease;
}

/* ── Info box ── */
.info-box {
    background: rgba(17,138,178,0.12);
    border: 1px solid rgba(17,138,178,0.3);
    border-radius: 10px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cached resource loading
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models...")
def load_dl_predictor():
    try:
        from predict import DLPredictor
        return DLPredictor.from_artifacts()
    except Exception as e:
        st.warning(f"DL model not available: {e}")
        return None


@st.cache_resource(show_spinner="Loading ML baseline...")
def load_ml_predictor():
    try:
        from predict import MLPredictor
        return MLPredictor.from_artifacts()
    except Exception as e:
        return None


@st.cache_resource
def load_priority_detector():
    return PriorityDetector()


@st.cache_resource
def load_response_generator():
    return ResponseGenerator()


@st.cache_resource
def load_thresholds_cached():
    return load_thresholds()


@st.cache_resource
def load_ml_metrics():
    p = ROOT / "artifacts" / "ml_metrics.json"
    return load_json(p) if p.exists() else {}


@st.cache_resource
def load_dl_metrics():
    p = ROOT / "artifacts" / "dl_metrics.json"
    return load_json(p) if p.exists() else {}


@st.cache_resource
def load_model_comparison():
    p = ROOT / "results" / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────

if "analytics" not in st.session_state:
    st.session_state.analytics = AnalyticsEngine()
    # Try to load previous session data
    try:
        st.session_state.analytics.load()
    except Exception:
        pass

if "escalation_queue" not in st.session_state:
    st.session_state.escalation_queue = []


# ─────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0e0", family="Inter"),
    margin=dict(t=40, l=10, r=10, b=10),
)

COLOUR_PALETTE = ["#06d6a0", "#118ab2", "#ffd166", "#ef476f", "#8338ec",
                  "#3a86ff", "#fb5607", "#ffbe0b", "#ff006e", "#8ecae6"]


# ─────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px'>
        <div style='font-size:2.5rem'>🏦</div>
        <div style='font-size:1.05rem; font-weight:700; color:#06d6a0; margin-top:4px'>
            Banking AI Support
        </div>
        <div style='font-size:0.75rem; color:rgba(255,255,255,0.4)'>
            Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "Customer Support",
            "Conversation Intelligence",
            "Analytics Dashboard",
            "Model Performance",
            "Escalation Queue",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    model_choice = st.selectbox(
        "Active Model",
        ["Deep Learning (BiLSTM/CNN)", "ML Baseline (TF-IDF)"],
        help="Select which model to use for intent prediction",
    )

    st.divider()
    n_queries = len(st.session_state.analytics)
    st.markdown(f"""
    <div style='text-align:center; color:rgba(255,255,255,0.5); font-size:0.8rem'>
        Session Queries: <b style='color:#06d6a0'>{n_queries}</b>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Clear Session", use_container_width=True):
        st.session_state.analytics.clear()
        st.session_state.escalation_queue = []
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────

dl_predictor = load_dl_predictor()
ml_predictor = load_ml_predictor()
priority_det = load_priority_detector()
response_gen = load_response_generator()
thresholds   = load_thresholds_cached()

# Choose active predictor
if "DL" in model_choice and dl_predictor is not None:
    predictor    = dl_predictor
    model_label  = "Deep Learning"
elif ml_predictor is not None:
    predictor    = ml_predictor
    model_label  = "ML Baseline"
else:
    predictor    = None
    model_label  = "No model loaded"

analyzer = ConversationAnalyzer(predictor, priority_det, thresholds) if predictor else None


# ─────────────────────────────────────────────────────────────
# Helper: run prediction pipeline
# ─────────────────────────────────────────────────────────────

def run_prediction(query: str) -> dict:
    """Run the full prediction pipeline for a customer query."""
    if predictor is None:
        return None

    t0 = time.perf_counter()

    # Prediction
    pred_result = predictor.predict(query)
    intent     = pred_result["intent"]
    confidence = pred_result["confidence"]
    top3       = pred_result["top_3"]

    # Confidence tier + escalation
    tier     = get_confidence_tier(confidence, thresholds)
    escalate = (tier == "LOW")

    # Priority
    prio_result = priority_det.detect(query, intent)
    if prio_result.priority == "HIGH":
        escalate = True

    # Sentiment
    sentiment = detect_sentiment(query)

    # GenAI response
    resp = response_gen.generate(
        query=query,
        intent=intent,
        confidence=confidence,
        priority=prio_result.priority,
        escalate=escalate,
        sentiment=sentiment,
    )

    elapsed = time.perf_counter() - t0

    # Record in analytics
    st.session_state.analytics.record(
        raw_text=query,
        cleaned_text=clean_text(query),
        intent=intent,
        confidence=confidence,
        confidence_tier=tier,
        priority=prio_result.priority,
        sentiment=sentiment,
        escalated=escalate,
        response_time_s=elapsed,
        model_used=model_label,
    )

    # Add to escalation queue if needed
    if escalate:
        st.session_state.escalation_queue.append({
            "Query":          query[:100],
            "Intent":         intent.replace("_", " ").title(),
            "Confidence":     f"{confidence:.0%}",
            "Confidence_val": confidence,
            "Priority":       prio_result.priority,
            "Sentiment":      sentiment,
            "Escalation Reason": escalation_reason(confidence, intent, thresholds),
            "Recommended Action": (
                "Assign to specialist agent" if prio_result.priority == "HIGH"
                else "Standard escalation"
            ),
        })

    return {
        "intent":       intent,
        "confidence":   confidence,
        "confidence_tier": tier,
        "top3":         top3,
        "priority":     prio_result.priority,
        "priority_reason": prio_result.reason,
        "escalate":     escalate,
        "sentiment":    sentiment,
        "response":     resp.text,
        "response_time": elapsed,
        "model_used":   resp.model_used,
    }


# ═════════════════════════════════════════════════════════════
# PAGE 1 — CUSTOMER SUPPORT
# ═════════════════════════════════════════════════════════════

if page == "Customer Support":
    st.markdown("<h1 style='color:#06d6a0; margin-bottom:4px'>🏦 Customer Support AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5); margin-bottom:24px'>Powered by Custom Deep Learning NLP + Generative AI</p>", unsafe_allow_html=True)

    if predictor is None:
        st.error("No model is loaded. Please train the models first (run train_baseline.py and train_deep_learning.py).")
        st.stop()

    # Query input
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_area(
            "Customer Query",
            placeholder="e.g. I tried to withdraw money from the ATM but no cash came out...",
            height=110,
            key="query_input",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_btn = st.button("Sample Query", use_container_width=True)
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    # Sample queries
    SAMPLE_QUERIES = [
        "I tried to withdraw money from the ATM but no cash came out and my account was debited.",
        "Why was I charged a fee for my card payment?",
        "My card has not arrived yet. It's been 2 weeks since I ordered it.",
        "My transfer is still pending after 3 days. What should I do?",
        "I don't recognise a transaction on my statement from yesterday.",
        "I can't remember my PIN. What should I do?",
        "What is the exchange rate for converting GBP to EUR?",
    ]

    if sample_btn:
        import random
        st.session_state.query_input = random.choice(SAMPLE_QUERIES)
        st.rerun()

    if analyze_btn and query.strip():
        with st.spinner("Analysing query..."):
            result = run_prediction(query.strip())

        if result:
            st.divider()

            # ── Row 1: Intent + Confidence + Priority + Escalation ──
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Predicted Intent</div>
                    <div style='margin-top:8px'>
                        <span class='intent-badge'>{result['intent'].replace('_',' ').title()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                conf_pct = f"{result['confidence']:.0%}"
                tier_color = {"HIGH": "#06d6a0", "MEDIUM": "#ffd166", "LOW": "#ef476f"}.get(result['confidence_tier'], "#e0e0e0")
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Confidence</div>
                    <div class='kpi-value' style='color:{tier_color}'>{conf_pct}</div>
                    <div style='font-size:0.8rem; color:{tier_color}'>{result['confidence_tier']} confidence</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                prio_cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(result['priority'], "badge-low")
                prio_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(result['priority'], "⚪")
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Priority</div>
                    <div style='margin-top:8px'>
                        <span class='{prio_cls}'>{prio_icon} {result['priority']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                esc_color = "#ef476f" if result['escalate'] else "#06d6a0"
                esc_label = "Escalated to Human" if result['escalate'] else "Automated Response"
                esc_icon  = "⚠️" if result['escalate'] else "✅"
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Resolution</div>
                    <div style='margin-top:8px; font-size:0.9rem; color:{esc_color}; font-weight:600'>
                        {esc_icon} {esc_label}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Row 2: Top-3 Predictions + Escalation Banner ──
            col_top3, col_esc = st.columns([1, 1])

            with col_top3:
                st.markdown("<div class='section-header'>Top-3 Predictions</div>", unsafe_allow_html=True)
                for i, item in enumerate(result["top3"]):
                    bar_width = int(item["confidence"] * 100)
                    rank_color = ["#06d6a0", "#118ab2", "#8338ec"][i]
                    st.markdown(f"""
                    <div style='margin-bottom:12px'>
                        <div style='display:flex; justify-content:space-between; margin-bottom:4px'>
                            <span style='font-size:0.9rem; color:#e0e0e0'>
                                {'🥇' if i==0 else '🥈' if i==1 else '🥉'} {item['intent'].replace('_',' ').title()}
                            </span>
                            <span style='font-size:0.9rem; color:{rank_color}; font-weight:600'>
                                {item['confidence']:.0%}
                            </span>
                        </div>
                        <div style='background:rgba(255,255,255,0.1); border-radius:4px; height:8px'>
                            <div style='width:{bar_width}%; height:8px; border-radius:4px;
                                        background:linear-gradient(90deg, {rank_color}, {rank_color}80)'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_esc:
                st.markdown("<div class='section-header'>Analysis Details</div>", unsafe_allow_html=True)
                sent_icon = {"Positive": "😊", "Neutral": "😐", "Negative": "😟"}.get(result["sentiment"], "😐")
                st.markdown(f"""
                <div class='info-box'>
                    <div style='margin-bottom:8px'><b>Sentiment:</b> {sent_icon} {result['sentiment']}</div>
                    <div style='margin-bottom:8px'><b>Priority Reason:</b><br>
                        <span style='color:rgba(255,255,255,0.7); font-size:0.85rem'>{result['priority_reason']}</span>
                    </div>
                    <div><b>Response Time:</b> {result['response_time']*1000:.0f}ms</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Escalation banner ──
            if result["escalate"]:
                st.markdown(f"""
                <div class='escalation-banner'>
                    <b style='color:#ef476f'>⚠️ ESCALATION RECOMMENDED</b><br>
                    <span style='color:rgba(255,255,255,0.7); font-size:0.9rem'>
                        This query has been flagged for human agent review.
                        Confidence: {result['confidence']:.0%} ({result['confidence_tier']}) |
                        Priority: {result['priority']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # ── AI Response ──
            st.markdown("<div class='section-header' style='margin-top:20px'>AI-Generated Response</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='response-card'>
                <div style='color:#e0e0e0; line-height:1.7'>{result['response']}</div>
                <div style='margin-top:12px; font-size:0.75rem; color:rgba(255,255,255,0.3)'>
                    Generated by: {result['model_used']} | Model: {model_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif analyze_btn and not query.strip():
        st.warning("Please enter a customer query.")


# ═════════════════════════════════════════════════════════════
# PAGE 2 — CONVERSATION INTELLIGENCE
# ═════════════════════════════════════════════════════════════

elif page == "Conversation Intelligence":
    st.markdown("<h1 style='color:#06d6a0'>🔍 Conversation Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5)'>Deep analysis of customer conversations</p>", unsafe_allow_html=True)

    if predictor is None:
        st.error("No model loaded. Train the models first.")
        st.stop()

    query = st.text_area(
        "Customer Query",
        placeholder="Enter a customer query for deep analysis...",
        height=120,
    )
    analyze_btn = st.button("Run Deep Analysis", type="primary")

    if analyze_btn and query.strip():
        with st.spinner("Running conversation intelligence..."):
            result = run_prediction(query.strip())
            report = analyzer.analyze(query.strip()) if analyzer else None

        if result and report:
            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='section-header'>Issue Summary</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='info-box' style='margin-bottom:16px'>
                    <b>Issue:</b> {report.issue}<br><br>
                    <b>Predicted Intent:</b> {report.predicted_intent.replace('_',' ').title()}<br>
                    <b>Confidence:</b> {report.confidence:.0%} ({report.confidence_tier})<br>
                    <b>Sentiment:</b> {report.sentiment}<br>
                    <b>Priority:</b> {report.priority}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='section-header'>Key Information Extracted</div>", unsafe_allow_html=True)
                for info in report.key_information:
                    st.markdown(f"- {info}")

            with col2:
                st.markdown("<div class='section-header'>Recommended Action</div>", unsafe_allow_html=True)
                st.info(report.recommended_action)

                st.markdown("<div class='section-header'>Escalation Status</div>", unsafe_allow_html=True)
                if report.escalation_required:
                    st.error(f"⚠️ **ESCALATION REQUIRED**\n\n{report.escalation_reason}")
                else:
                    st.success(f"✅ **Automated Response Appropriate**\n\n{report.escalation_reason}")

                st.markdown("<div class='section-header'>Top-3 Intent Predictions</div>", unsafe_allow_html=True)
                for item in result["top3"]:
                    col_a, col_b = st.columns([4, 1])
                    col_a.markdown(f"**{item['intent'].replace('_',' ').title()}**")
                    col_b.markdown(f"`{item['confidence']:.0%}`")

            # Full JSON report (expandable)
            with st.expander("View Full Intelligence Report (JSON)"):
                st.json(report.to_dict())


# ═════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS DASHBOARD
# ═════════════════════════════════════════════════════════════

elif page == "Analytics Dashboard":
    st.markdown("<h1 style='color:#06d6a0'>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5)'>Real-time support intelligence metrics</p>", unsafe_allow_html=True)

    engine  = st.session_state.analytics
    summary = engine.get_summary()
    total   = summary["total_queries"]

    # ── KPI Cards ──
    cols = st.columns(5)
    kpis = [
        ("Total Queries",    str(total),                             "Processed this session"),
        ("Automation Rate",  f"{summary['automation_rate']:.0%}",    "Queries auto-handled"),
        ("Escalation Rate",  f"{summary['escalation_rate']:.0%}",    "Sent to human agents"),
        ("Avg Confidence",   f"{summary['avg_confidence']:.0%}",     "Mean model confidence"),
        ("High Priority %",  f"{summary['high_priority_pct']:.0%}",  "Urgent queries"),
    ]
    for col, (label, value, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value'>{value}</div>
                <div class='kpi-label'>{label}</div>
                <div style='font-size:0.7rem; color:rgba(255,255,255,0.3); margin-top:4px'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if total == 0:
        st.info("No queries analysed yet. Go to the Customer Support page to start.")
    else:
        # ── Row 1: Intent + Confidence ──
        col1, col2 = st.columns(2)

        with col1:
            intent_df = engine.get_intent_distribution(top_n=12)
            if not intent_df.empty:
                fig = px.bar(
                    intent_df, x="count", y="intent", orientation="h",
                    color="count", color_continuous_scale="teal",
                    title="Top Intent Distribution",
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            conf_series = engine.get_confidence_distribution()
            if not conf_series.empty:
                fig = px.histogram(
                    conf_series, nbins=20,
                    color_discrete_sequence=["#118ab2"],
                    title="Confidence Distribution",
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Priority + Sentiment + Escalation ──
        col3, col4, col5 = st.columns(3)

        with col3:
            prio_df = engine.get_priority_distribution()
            if not prio_df.empty:
                fig = px.pie(
                    prio_df, names="priority", values="count",
                    color_discrete_map={"HIGH": "#ef476f", "MEDIUM": "#ffd166", "LOW": "#06d6a0"},
                    title="Priority Distribution",
                    hole=0.4,
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            sent_df = engine.get_sentiment_distribution()
            if not sent_df.empty:
                fig = px.pie(
                    sent_df, names="sentiment", values="count",
                    color_discrete_map={"Positive": "#06d6a0", "Neutral": "#ffd166", "Negative": "#ef476f"},
                    title="Sentiment Distribution",
                    hole=0.4,
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        with col5:
            esc_df = engine.get_escalation_distribution()
            if not esc_df.empty:
                fig = px.pie(
                    esc_df, names="status", values="count",
                    color_discrete_map={"Automated": "#06d6a0", "Escalated": "#ef476f"},
                    title="Escalation Distribution",
                    hole=0.4,
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════

elif page == "Model Performance":
    st.markdown("<h1 style='color:#06d6a0'>📈 Model Performance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5)'>ML Baseline vs Custom Deep Learning comparison</p>", unsafe_allow_html=True)

    ml_metrics = load_ml_metrics()
    dl_metrics = load_dl_metrics()
    comp_df    = load_model_comparison()

    # ── ML vs DL summary ──
    st.markdown("### ML Baseline Results (Validation Set)")
    if ml_metrics:
        best_ml = ml_metrics.get("model_metrics", {})
        for model_name, m in best_ml.items():
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Model", model_name.split("+")[-1].strip()[:20])
            col2.metric("Accuracy",   f"{m.get('val_accuracy',0):.2%}")
            col3.metric("Precision",  f"{m.get('val_precision_macro',0):.2%}")
            col4.metric("Macro F1",   f"{m.get('val_f1_macro',0):.2%}")
            col5.metric("Weighted F1", f"{m.get('val_f1_weighted',0):.2%}")
    else:
        st.info("ML metrics not found. Run train_baseline.py first.")

    st.divider()

    # ── DL model comparison ──
    st.markdown("### Deep Learning Model Comparison (Validation Set)")
    if not comp_df.empty:
        # Metrics bar chart
        metrics_to_show = ["val_accuracy", "val_f1_macro", "val_f1_weighted"]
        available = [c for c in metrics_to_show if c in comp_df.columns]
        if available:
            fig = go.Figure()
            colors = ["#06d6a0", "#118ab2", "#ffd166"]
            for col, color in zip(available, colors):
                fig.add_trace(go.Bar(
                    name=col.replace("val_", "").replace("_", " ").title(),
                    x=comp_df["model_name"],
                    y=comp_df[col],
                    marker_color=color,
                    opacity=0.85,
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="DL Model Comparison",
                barmode="group",
                yaxis_range=[0, 1.05],
            )
            st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        display_cols = [c for c in [
            "model_name", "val_accuracy", "val_f1_macro", "val_f1_weighted",
            "n_params", "training_time"
        ] if c in comp_df.columns]
        display_df = comp_df[display_cols].copy()
        for c in display_cols:
            if c.startswith("val_"):
                display_df[c] = display_df[c].apply(
                    lambda x: f"{x:.2%}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
                )
        st.dataframe(display_df, use_container_width=True)

        # Best model highlight
        if "val_f1_macro" in comp_df.columns:
            best = comp_df.loc[comp_df["val_f1_macro"].idxmax()]
            st.success(
                f"**Selected Best Model: {best['model_name']}** — "
                f"Val Accuracy: {best.get('val_accuracy',0):.2%}, "
                f"Val Macro F1: {best.get('val_f1_macro',0):.2%}"
            )
    else:
        st.info("Deep learning results not found. Run train_deep_learning.py first.")

    # ── Training curves (if available) ──
    st.divider()
    st.markdown("### Training Curves")
    charts_dir = ROOT / "results" / "charts"
    curve_files = list(charts_dir.glob("training_curves_*.png")) if charts_dir.exists() else []
    if curve_files:
        cols = st.columns(min(len(curve_files), 2))
        for col, f in zip(cols, curve_files):
            col.image(str(f), caption=f.stem.replace("training_curves_", "").title(), use_container_width=True)
    else:
        st.info("No training curves found yet.")


# ═════════════════════════════════════════════════════════════
# PAGE 5 — ESCALATION QUEUE
# ═════════════════════════════════════════════════════════════

elif page == "Escalation Queue":
    st.markdown("<h1 style='color:#ef476f'>⚠️ Escalation Queue</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5)'>Cases requiring human agent attention</p>", unsafe_allow_html=True)

    queue = st.session_state.escalation_queue

    if not queue:
        st.info("No escalated queries yet. Cases will appear here when confidence is low or priority is HIGH.")
    else:
        queue_df = pd.DataFrame(queue)

        # ── Filters ──
        col1, col2, col3 = st.columns(3)
        with col1:
            priority_filter = st.multiselect(
                "Filter by Priority",
                options=["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"],
            )
        with col2:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=["Positive", "Neutral", "Negative"],
                default=["Positive", "Neutral", "Negative"],
            )
        with col3:
            conf_threshold = st.slider("Max Confidence", 0.0, 1.0, 1.0, 0.05)

        filtered = queue_df[
            (queue_df["Priority"].isin(priority_filter)) &
            (queue_df["Sentiment"].isin(sentiment_filter)) &
            (queue_df["Confidence_val"] <= conf_threshold)
        ].drop(columns=["Confidence_val"], errors="ignore")

        st.markdown(f"**Showing {len(filtered)} of {len(queue)} escalated queries**")

        if not filtered.empty:
            # Colour code by priority
            def priority_style(row):
                colors = {"HIGH": "background-color:#3d0000", "MEDIUM": "#3d3000", "LOW": "#003d00"}
                return [f"background-color:{colors.get(row['Priority'], '')}" for _ in row]

            try:
                st.dataframe(
                    filtered.style.apply(priority_style, axis=1),
                    use_container_width=True,
                    height=400,
                )
            except Exception:
                st.dataframe(
                    filtered,
                    use_container_width=True,
                    height=400,
                )

            # Download
            csv = filtered.to_csv(index=False)
            st.download_button(
                "Download Escalation Queue",
                data=csv,
                file_name="escalation_queue.csv",
                mime="text/csv",
            )
        else:
            st.warning("No records match the current filters.")
