"""
train_baseline.py -- Phase 3: Traditional ML Baseline
======================================================
Objective:
  Build a strong TF-IDF + LinearSVC baseline before any Deep Learning.
  Also compare Logistic Regression for reference.

Key rules:
  - Use ONLY train_split.csv and val_split.csv (never test.csv).
  - Tune parameters on validation set only.
  - Save models and metrics.

Run:
    cd banking-customer-support-ai
    python src/train_baseline.py
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

SRC_DIR = Path(__file__).resolve().parent
ROOT    = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import get_logger, ensure_dir, save_json, set_seed

logger = get_logger("train_baseline")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "train_path":  ROOT / "data" / "train_split.csv",
    "val_path":    ROOT / "data" / "val_split.csv",
    "models_dir":  ROOT / "models" / "tfidf",
    "results_dir": ROOT / "results",
    "charts_dir":  ROOT / "results" / "charts",
    "artifacts_dir": ROOT / "artifacts",
    "random_seed": 42,

    # TF-IDF parameters (tuned on validation set)
    "tfidf_params": {
        "analyzer":       "word",
        "ngram_range":    (1, 2),      # unigrams + bigrams
        "max_features":   80_000,
        "sublinear_tf":   True,        # log-normalised TF
        "min_df":         2,           # ignore very rare tokens
        "strip_accents":  "unicode",
    },

    # LinearSVC parameters
    "svm_params": {
        "C":            0.5,
        "max_iter":     2000,
        "random_state": 42,
    },

    # Logistic Regression parameters (comparison model)
    "lr_params": {
        "C":            4.0,
        "max_iter":     1000,
        "solver":       "lbfgs",
        "random_state": 42,
        "n_jobs":       -1,
    },
}


# ─────────────────────────────────────────────────────────────
# Metric computation helper
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """Return a dict of classification metrics."""
    return {
        f"{prefix}accuracy":        round(accuracy_score(y_true, y_pred), 4),
        f"{prefix}precision_macro": round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        f"{prefix}recall_macro":    round(recall_score(   y_true, y_pred, average="macro",    zero_division=0), 4),
        f"{prefix}f1_macro":        round(f1_score(       y_true, y_pred, average="macro",    zero_division=0), 4),
        f"{prefix}f1_weighted":     round(f1_score(       y_true, y_pred, average="weighted", zero_division=0), 4),
    }


# ─────────────────────────────────────────────────────────────
# Chart: Confusion matrix (top N classes)
# ─────────────────────────────────────────────────────────────

CHART_STYLE = {
    "axes.facecolor":   "#1a1a2e",
    "figure.facecolor": "#16213e",
    "text.color":       "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
}


def plot_confusion_matrix(y_true, y_pred, labels, title: str, save_path: Path):
    """Plot and save a confusion matrix heatmap (top-N classes for readability)."""
    plt.rcParams.update(CHART_STYLE)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Limit to top-30 classes for readability
    if len(labels) > 30:
        class_totals = cm.sum(axis=1)
        top_idx      = np.argsort(class_totals)[-30:][::-1]
        cm           = cm[np.ix_(top_idx, top_idx)]
        labels       = [labels[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        linecolor="#2a2a4a",
    )
    ax.set_title(title, color="#e0e0e0", fontsize=12, pad=15)
    ax.set_xlabel("Predicted", color="#e0e0e0")
    ax.set_ylabel("True",      color="#e0e0e0")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"  Confusion matrix saved -> {save_path}")


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Path):
    """Bar chart comparing ML models on key metrics."""
    plt.rcParams.update(CHART_STYLE)

    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    x = np.arange(len(comparison_df))
    width = 0.25
    colors = ["#06d6a0", "#118ab2", "#ffd166"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        col = f"val_{metric}" if f"val_{metric}" in comparison_df.columns else metric
        ax.bar(x + i * width, comparison_df[col], width=width, label=metric, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(comparison_df["model"], fontsize=10, color="#e0e0e0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", color="#e0e0e0")
    ax.set_title("ML Baseline Model Comparison (Validation Set)", color="#e0e0e0", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"  Model comparison chart saved -> {save_path}")


# ─────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────

def build_svm_pipeline() -> Pipeline:
    """TF-IDF + calibrated LinearSVC pipeline."""
    tfidf = TfidfVectorizer(**CONFIG["tfidf_params"])
    svm   = LinearSVC(**CONFIG["svm_params"])
    # CalibratedClassifierCV wraps LinearSVC to provide probability estimates
    # via cross-validation (Platt scaling). We use cv=3 here because training set
    # is ~8000 samples -- enough for reliable calibration.
    cal   = CalibratedClassifierCV(svm, cv=3, method="isotonic")
    return Pipeline([("tfidf", tfidf), ("clf", cal)])


def build_lr_pipeline() -> Pipeline:
    """TF-IDF + Logistic Regression pipeline."""
    tfidf = TfidfVectorizer(**CONFIG["tfidf_params"])
    lr    = LogisticRegression(**CONFIG["lr_params"])
    return Pipeline([("tfidf", tfidf), ("clf", lr)])


def train_and_evaluate(
    pipeline: Pipeline,
    X_train, y_train,
    X_val,   y_val,
    label_names: list[str],
    model_name: str,
) -> dict:
    """Fit pipeline, evaluate on val, return metrics dict."""
    logger.info(f"\n--- Training: {model_name} ---")

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    logger.info(f"  Training time: {train_time:.1f}s")

    y_val_pred  = pipeline.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred, prefix="val_")

    for key, val in val_metrics.items():
        logger.info(f"  {key}: {val:.4f}")

    # Full classification report
    report = classification_report(
        y_val, y_val_pred,
        target_names=label_names,
        zero_division=0,
    )
    logger.info(f"\nClassification Report ({model_name}):\n{report}")

    return {**val_metrics, "training_time_s": round(train_time, 2)}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run():
    """Execute Phase 3: Traditional ML Baseline."""
    set_seed(CONFIG["random_seed"])
    ensure_dir(CONFIG["models_dir"])
    ensure_dir(CONFIG["charts_dir"])
    ensure_dir(CONFIG["artifacts_dir"])

    logger.info("=" * 58)
    logger.info("  PHASE 3 - Traditional ML Baseline")
    logger.info("=" * 58)

    # ── 1. Load splits ──
    df_train = pd.read_csv(CONFIG["train_path"])
    df_val   = pd.read_csv(CONFIG["val_path"])
    logger.info(f"Train split: {len(df_train):,}  |  Val split: {len(df_val):,}")

    X_train, y_train = df_train["clean_text"].astype(str), df_train["category"]
    X_val,   y_val   = df_val["clean_text"].astype(str),   df_val["category"]

    label_names = sorted(y_train.unique())

    # ── 2. Build and train models ──
    models = {
        "TF-IDF + LinearSVC (calibrated)": build_svm_pipeline(),
        "TF-IDF + Logistic Regression":    build_lr_pipeline(),
    }

    all_metrics = {}
    pipelines   = {}

    for name, pipeline in models.items():
        metrics = train_and_evaluate(
            pipeline, X_train, y_train, X_val, y_val, label_names, name
        )
        all_metrics[name] = metrics
        pipelines[name]   = pipeline

    # ── 3. Select best model based on val Macro F1 ──
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["val_f1_macro"])
    logger.info(f"\nBest ML model (val macro F1): {best_name}")
    best_pipeline = pipelines[best_name]

    # ── 4. Save best model ──
    svm_path  = CONFIG["models_dir"] / "baseline_model.joblib"
    tfidf_path = CONFIG["models_dir"] / "tfidf_vectorizer.joblib"

    # Save the full pipeline (TF-IDF + classifier together)
    joblib.dump(best_pipeline, svm_path)
    logger.info(f"Best ML pipeline saved -> {svm_path}")

    # Also expose the TF-IDF vectorizer separately for downstream use
    tfidf_step = best_pipeline.named_steps["tfidf"]
    joblib.dump(tfidf_step, tfidf_path)
    logger.info(f"TF-IDF vectorizer saved -> {tfidf_path}")

    # Save all pipelines for reference
    for name, pipeline in pipelines.items():
        safe_name = name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
        joblib.dump(pipeline, CONFIG["models_dir"] / f"pipeline_{safe_name}.joblib")

    # ── 5. Confusion matrix for best model ──
    y_val_pred_best = best_pipeline.predict(X_val)
    cm_path = CONFIG["charts_dir"] / "confusion_matrix_ml_baseline.png"
    plot_confusion_matrix(y_val, y_val_pred_best, label_names,
                          f"Confusion Matrix - {best_name}", cm_path)

    # ── 6. Save comparison table ──
    comparison_rows = []
    for name, metrics in all_metrics.items():
        comparison_rows.append({"model": name, **metrics})
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(CONFIG["results_dir"] / "ml_comparison.csv", index=False)
    logger.info(f"ML comparison CSV saved -> {CONFIG['results_dir'] / 'ml_comparison.csv'}")

    # ── 7. Model comparison chart ──
    plot_model_comparison(comparison_df, CONFIG["charts_dir"] / "ml_model_comparison.png")

    # ── 8. Per-class metric analysis ──
    report_dict = {}
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_val)
        rep = classification_report(y_val, y_pred, target_names=label_names,
                                    zero_division=0, output_dict=True)
        safe_name = name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
        pd.DataFrame(rep).T.to_csv(
            CONFIG["results_dir"] / f"classification_report_{safe_name}.csv"
        )

    # ── 9. Save metrics to artifacts ──
    ml_metrics = {
        "best_model": best_name,
        "model_metrics": all_metrics,
    }
    save_json(ml_metrics, CONFIG["artifacts_dir"] / "ml_metrics.json")
    logger.info(f"ML metrics saved -> {CONFIG['artifacts_dir'] / 'ml_metrics.json'}")

    # ── 10. Summary ──
    logger.info("\n" + "=" * 58)
    logger.info("ML Baseline Summary (Validation Set)")
    logger.info("=" * 58)
    for name, m in all_metrics.items():
        logger.info(f"\n  Model: {name}")
        logger.info(f"    Accuracy   : {m['val_accuracy']:.4f}")
        logger.info(f"    Macro F1   : {m['val_f1_macro']:.4f}")
        logger.info(f"    Weighted F1: {m['val_f1_weighted']:.4f}")
        logger.info(f"    Train time : {m['training_time_s']:.1f}s")

    logger.info("\n[OK] Phase 3 complete.")
    return best_pipeline, ml_metrics


if __name__ == "__main__":
    run()
