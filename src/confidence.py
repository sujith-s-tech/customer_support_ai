"""
confidence.py -- Phase 10: Confidence Calibration and Escalation System
=========================================================================
Objective:
  Empirically determine confidence thresholds using the validation set
  and implement an escalation policy for uncertain predictions.

Key design decisions:
  - Thresholds selected from validation data, never from test data.
  - Three tiers: HIGH (automated), MEDIUM (verify), LOW (escalate).
  - High-confidence errors are explicitly analysed (silent failure risk).
  - Thresholds saved as JSON for use in inference and Streamlit.

Run standalone to calibrate thresholds:
    python src/confidence.py
"""

import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import get_logger, ensure_dir, save_json, load_json

logger = get_logger("confidence")

# ─────────────────────────────────────────────────────────────
# Default operational thresholds
# (overridden if calibrate() has been run)
# ─────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "high":   0.70,   # >= high -> automated response
    "medium": 0.45,   # >= medium && < high -> verify with customer
    # < medium -> escalate to human agent
}


# ─────────────────────────────────────────────────────────────
# Threshold analysis
# ─────────────────────────────────────────────────────────────

def analyse_thresholds(
    confidences:  np.ndarray,
    correct_mask: np.ndarray,
    thresholds:   list[float] = None,
) -> pd.DataFrame:
    """
    For each candidate threshold, compute coverage, accuracy, and escalation rate.

    Args:
        confidences  : Array of max-class probabilities (shape N,).
        correct_mask : Boolean array -- True if prediction == ground truth (shape N,).
        thresholds   : List of confidence thresholds to evaluate.
    Returns:
        DataFrame with per-threshold statistics.
    """
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                      0.75, 0.80, 0.85, 0.90, 0.95]

    rows = []
    n    = len(confidences)

    for t in thresholds:
        high_conf_mask = confidences >= t
        low_conf_mask  = ~high_conf_mask

        n_high = int(high_conf_mask.sum())
        n_low  = int(low_conf_mask.sum())

        # Among high-confidence predictions
        hc_correct = int((high_conf_mask & correct_mask).sum())
        hc_wrong   = n_high - hc_correct
        hc_acc     = hc_correct / n_high if n_high > 0 else np.nan

        rows.append({
            "threshold":         t,
            "coverage":          round(n_high / n, 4),
            "escalation_rate":   round(n_low  / n, 4),
            "hc_accuracy":       round(hc_acc, 4) if not np.isnan(hc_acc) else None,
            "hc_errors":         hc_wrong,
            "hc_error_rate":     round(hc_wrong / n_high, 4) if n_high > 0 else None,
            "n_high_conf":       n_high,
            "n_low_conf":        n_low,
        })

    return pd.DataFrame(rows)


def select_thresholds(
    analysis_df: pd.DataFrame,
    target_hc_accuracy: float = 0.95,
    target_coverage:    float = 0.70,
) -> dict:
    """
    Select HIGH and MEDIUM thresholds based on the threshold analysis.

    High threshold policy:
        Find the LOWEST threshold that achieves >= target_hc_accuracy.
        This maximises coverage while maintaining high accuracy.

    Medium threshold:
        Set to 60% of the high threshold (heuristic; guarantees a meaningful
        mid-tier before escalation).

    Args:
        analysis_df        : Output of analyse_thresholds().
        target_hc_accuracy : Target accuracy for the high-confidence tier.
        target_coverage    : Minimum desired coverage for high-confidence tier.
    Returns:
        dict with "high" and "medium" threshold values.
    """
    # Filter rows that meet the accuracy target
    viable = analysis_df[
        (analysis_df["hc_accuracy"] >= target_hc_accuracy) &
        (analysis_df["coverage"]    >= target_coverage)
    ]

    if viable.empty:
        # Relax accuracy target slightly
        viable = analysis_df[analysis_df["hc_accuracy"] >= 0.90]

    if viable.empty:
        logger.warning("No threshold meets targets; using defaults.")
        return DEFAULT_THRESHOLDS.copy()

    # Choose the row with the best coverage (highest coverage among viable)
    best_row = viable.sort_values("coverage", ascending=False).iloc[0]
    high_thresh   = float(best_row["threshold"])
    medium_thresh = max(0.30, round(high_thresh * 0.65, 2))

    logger.info(f"  Selected HIGH   threshold: {high_thresh:.2f}  "
                f"(coverage={best_row['coverage']:.2%}, hc_acc={best_row['hc_accuracy']:.4f})")
    logger.info(f"  Selected MEDIUM threshold: {medium_thresh:.2f}")

    return {"high": high_thresh, "medium": medium_thresh}


# ─────────────────────────────────────────────────────────────
# Escalation decision
# ─────────────────────────────────────────────────────────────

def get_confidence_tier(confidence: float, thresholds: dict = None) -> str:
    """
    Classify a confidence score into a tier.

    Returns:
        "HIGH"   -> automated response is appropriate
        "MEDIUM" -> response + request customer verification
        "LOW"    -> escalate to human agent
    """
    if thresholds is None:
        thresholds = load_thresholds()

    if confidence >= thresholds["high"]:
        return "HIGH"
    elif confidence >= thresholds["medium"]:
        return "MEDIUM"
    else:
        return "LOW"


def should_escalate(confidence: float, thresholds: dict = None) -> bool:
    """Return True if the query should be escalated to a human agent."""
    tier = get_confidence_tier(confidence, thresholds)
    return tier == "LOW"


def escalation_reason(confidence: float, intent: str, thresholds: dict = None) -> str:
    """Return a human-readable reason for the escalation decision."""
    tier = get_confidence_tier(confidence, thresholds)
    if tier == "HIGH":
        return "High confidence prediction -- automated response."
    elif tier == "MEDIUM":
        return f"Medium confidence ({confidence:.0%}). Recommend customer verification."
    else:
        return (
            f"Low confidence ({confidence:.0%}). Intent unclear -- "
            "escalating to human agent for accurate resolution."
        )


# ─────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────

THRESHOLDS_PATH = ROOT / "artifacts" / "thresholds.json"


def save_thresholds(thresholds: dict) -> None:
    save_json(thresholds, THRESHOLDS_PATH)
    logger.info(f"Thresholds saved -> {THRESHOLDS_PATH}")


def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        data = load_json(THRESHOLDS_PATH)
        return {"high": data["high"], "medium": data["medium"]}
    logger.warning("thresholds.json not found -- using defaults.")
    return DEFAULT_THRESHOLDS.copy()


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

CHART_STYLE = {
    "axes.facecolor":   "#1a1a2e",
    "figure.facecolor": "#16213e",
    "text.color":       "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
}


def plot_threshold_analysis(analysis_df: pd.DataFrame, selected: dict, charts_dir: Path):
    """Plot coverage vs accuracy trade-off across thresholds."""
    plt.rcParams.update(CHART_STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confidence Threshold Analysis (Validation Set)", color="#e0e0e0", fontsize=12)

    thresholds = analysis_df["threshold"].values

    # Coverage vs Accuracy
    axes[0].plot(thresholds, analysis_df["coverage"],    "o-", color="#06d6a0", label="Coverage")
    axes[0].plot(thresholds, analysis_df["hc_accuracy"], "s-", color="#ffd166", label="HC Accuracy")
    axes[0].axvline(selected["high"],   color="#ef476f", linestyle="--", lw=1.5, label="HIGH thresh")
    axes[0].axvline(selected["medium"], color="#118ab2", linestyle="--", lw=1.5, label="MED thresh")
    axes[0].set_xlabel("Confidence Threshold")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Coverage vs HC Accuracy", color="#e0e0e0")
    axes[0].legend(fontsize=8)

    # Escalation rate
    axes[1].bar(thresholds, analysis_df["escalation_rate"],
                width=0.04, color="#ef476f", alpha=0.8, label="Escalation Rate")
    axes[1].axvline(selected["high"],   color="#06d6a0", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Confidence Threshold")
    axes[1].set_ylabel("Escalation Rate")
    axes[1].set_title("Escalation Rate per Threshold", color="#e0e0e0")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = charts_dir / "threshold_analysis.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Threshold analysis chart saved -> {out}")


# ─────────────────────────────────────────────────────────────
# Main calibration routine
# ─────────────────────────────────────────────────────────────

def calibrate(
    confidences:  np.ndarray,
    correct_mask: np.ndarray,
    charts_dir:   Path = ROOT / "results" / "charts",
) -> dict:
    """
    Full threshold calibration pipeline.

    1. Analyse all candidate thresholds.
    2. Select optimal HIGH and MEDIUM thresholds.
    3. Save thresholds to JSON.
    4. Plot and save analysis chart.

    Args:
        confidences  : Array of max predicted probabilities from val set.
        correct_mask : Boolean array (True = correct prediction).
        charts_dir   : Directory to save charts.
    Returns:
        Selected threshold dict.
    """
    ensure_dir(charts_dir)

    logger.info("\n--- Confidence Threshold Calibration ---")
    logger.info(f"  N validation samples : {len(confidences)}")
    logger.info(f"  Mean confidence      : {confidences.mean():.4f}")
    logger.info(f"  Overall accuracy     : {correct_mask.mean():.4f}")

    analysis_df = analyse_thresholds(confidences, correct_mask)
    logger.info("\nThreshold Analysis Table:")
    logger.info("\n" + analysis_df.to_string(index=False))

    selected = select_thresholds(analysis_df)
    save_thresholds(selected)
    plot_threshold_analysis(analysis_df, selected, charts_dir)

    # Save full analysis CSV
    analysis_df.to_csv(ROOT / "results" / "threshold_analysis.csv", index=False)

    return selected


if __name__ == "__main__":
    # For testing: generate synthetic confidence distribution
    np.random.seed(42)
    n = 2001
    confs = np.clip(np.random.beta(5, 2, n), 0, 1)   # skewed toward high confidence
    correct = (confs > 0.5) & (np.random.rand(n) > 0.15)  # simulate some errors
    selected = calibrate(confs, correct)
    print(f"\nSelected thresholds: {selected}")
