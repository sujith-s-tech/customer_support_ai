"""
analytics.py -- Phase 14: Analytics Engine
===========================================
Objective:
  Compute, store, and visualise analytics from processed support queries.
  Used by the Streamlit dashboard to display KPIs and charts.

Usage:
    from analytics import AnalyticsEngine
    engine = AnalyticsEngine()
    engine.record(query, result)       # Record a processed query
    summary = engine.get_summary()     # Get KPI summary
    engine.save("artifacts/analytics.json")
    engine.load("artifacts/analytics.json")
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import get_logger, ensure_dir, save_json, load_json

logger = get_logger("analytics")

ANALYTICS_PATH = ROOT / "artifacts" / "analytics.json"


# ─────────────────────────────────────────────────────────────
# Query record
# ─────────────────────────────────────────────────────────────

@dataclass
class QueryRecord:
    """A single processed query and its analysis results."""
    timestamp:     float
    raw_text:      str
    cleaned_text:  str
    intent:        str
    confidence:    float
    confidence_tier: str
    priority:      str
    sentiment:     str
    escalated:     bool
    response_time_s: float
    model_used:    str  # "dl" or "ml"


# ─────────────────────────────────────────────────────────────
# Analytics Engine
# ─────────────────────────────────────────────────────────────

class AnalyticsEngine:
    """
    Collects and aggregates analytics from processed queries.

    All data is stored in-memory and can be persisted to JSON.
    """

    def __init__(self):
        self._records: list[QueryRecord] = []

    def record(
        self,
        raw_text:      str,
        cleaned_text:  str,
        intent:        str,
        confidence:    float,
        confidence_tier: str,
        priority:      str,
        sentiment:     str,
        escalated:     bool,
        response_time_s: float = 0.0,
        model_used:    str = "dl",
    ) -> None:
        """Record a single processed query."""
        rec = QueryRecord(
            timestamp=time.time(),
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            intent=intent,
            confidence=confidence,
            confidence_tier=confidence_tier,
            priority=priority,
            sentiment=sentiment,
            escalated=escalated,
            response_time_s=response_time_s,
            model_used=model_used,
        )
        self._records.append(rec)

    def __len__(self) -> int:
        return len(self._records)

    # ─────────────────────────────────────────────────────────
    # KPI computation
    # ─────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Compute KPI summary from all recorded queries."""
        if not self._records:
            return self._empty_summary()

        total = len(self._records)
        confs = np.array([r.confidence for r in self._records])

        n_escalated  = sum(1 for r in self._records if r.escalated)
        n_automated  = total - n_escalated
        n_high_conf  = sum(1 for r in self._records if r.confidence_tier == "HIGH")
        n_medium_conf = sum(1 for r in self._records if r.confidence_tier == "MEDIUM")
        n_low_conf   = sum(1 for r in self._records if r.confidence_tier == "LOW")
        n_high_prio  = sum(1 for r in self._records if r.priority == "HIGH")
        n_medium_prio = sum(1 for r in self._records if r.priority == "MEDIUM")
        n_low_prio   = sum(1 for r in self._records if r.priority == "LOW")

        avg_response = np.mean([r.response_time_s for r in self._records])

        return {
            "total_queries":       total,
            "automation_rate":     round(n_automated / total, 4),
            "escalation_rate":     round(n_escalated / total, 4),
            "avg_confidence":      round(float(confs.mean()), 4),
            "high_conf_pct":       round(n_high_conf / total, 4),
            "medium_conf_pct":     round(n_medium_conf / total, 4),
            "low_conf_pct":        round(n_low_conf / total, 4),
            "high_priority_pct":   round(n_high_prio / total, 4),
            "medium_priority_pct": round(n_medium_prio / total, 4),
            "low_priority_pct":    round(n_low_prio / total, 4),
            "avg_response_time_s": round(float(avg_response), 3),
            "n_escalated":         n_escalated,
            "n_automated":         n_automated,
        }

    def _empty_summary(self) -> dict:
        return {k: 0 for k in [
            "total_queries", "automation_rate", "escalation_rate",
            "avg_confidence", "high_conf_pct", "medium_conf_pct", "low_conf_pct",
            "high_priority_pct", "medium_priority_pct", "low_priority_pct",
            "avg_response_time_s", "n_escalated", "n_automated",
        ]}

    # ─────────────────────────────────────────────────────────
    # Distribution getters (for charts)
    # ─────────────────────────────────────────────────────────

    def get_intent_distribution(self, top_n: int = 15) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=["intent", "count"])
        counts = defaultdict(int)
        for r in self._records:
            counts[r.intent] += 1
        df = pd.DataFrame(list(counts.items()), columns=["intent", "count"])
        return df.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)

    def get_confidence_distribution(self) -> pd.Series:
        if not self._records:
            return pd.Series(dtype=float)
        return pd.Series([r.confidence for r in self._records], name="confidence")

    def get_sentiment_distribution(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=["sentiment", "count"])
        counts = defaultdict(int)
        for r in self._records:
            counts[r.sentiment] += 1
        return pd.DataFrame(list(counts.items()), columns=["sentiment", "count"])

    def get_priority_distribution(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=["priority", "count"])
        counts = defaultdict(int)
        for r in self._records:
            counts[r.priority] += 1
        return pd.DataFrame(list(counts.items()), columns=["priority", "count"])

    def get_escalation_distribution(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=["status", "count"])
        n_esc = sum(1 for r in self._records if r.escalated)
        n_auto = len(self._records) - n_esc
        return pd.DataFrame([
            {"status": "Automated", "count": n_auto},
            {"status": "Escalated", "count": n_esc},
        ])

    def get_escalated_records(self) -> pd.DataFrame:
        """Return a DataFrame of all escalated queries (for the Escalation Queue page)."""
        escalated = [r for r in self._records if r.escalated]
        if not escalated:
            return pd.DataFrame()
        rows = []
        for r in escalated:
            rows.append({
                "Query":         r.raw_text[:100],
                "Intent":        r.intent.replace("_", " ").title(),
                "Confidence":    f"{r.confidence:.0%}",
                "Confidence_val": r.confidence,
                "Priority":      r.priority,
                "Sentiment":     r.sentiment,
                "Escalation Reason": (
                    "Low confidence" if r.confidence_tier == "LOW"
                    else "High-risk intent"
                ),
            })
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return all records as a DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self._records])

    def save(self, path: str | Path = ANALYTICS_PATH) -> None:
        ensure_dir(Path(path).parent)
        data = [asdict(r) for r in self._records]
        save_json({"records": data}, path)

    def load(self, path: str | Path = ANALYTICS_PATH) -> None:
        if not Path(path).exists():
            logger.warning(f"Analytics file not found: {path}")
            return
        data = load_json(path)
        self._records = [QueryRecord(**r) for r in data.get("records", [])]
        logger.info(f"Loaded {len(self._records)} records from {path}")

    def clear(self) -> None:
        """Clear all records (for demo reset)."""
        self._records.clear()
